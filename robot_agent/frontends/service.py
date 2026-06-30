"""通道无关门面 `AgentService`：把常驻 agent 的能力收成稳定 API，供任意通道复用。

**可拔插接口的核心**。本仓库的天然分界线是 `Driver`（driver/loop.py）的
`submit(Event)`（输入）与 `on_turn(TurnResult)`（输出）；Web / IM / 麦克风扬声器都只是这条
边界上的不同「通道」。`AgentService` 把这条边界封装成与通道无关的门面：

- 输入：`submit_user_text` / `resume` —— 翻译成 `Event` 投递给常驻 `Driver`。
- 输出：`subscribe` —— 把每个回合（`TurnResult`）精简后**广播（fan-out）**给所有订阅者；
  SSE、未来的 IM/语音都订阅同一输出流。
- 只读视图：`history`（短期记忆/会话）、`memory`（长期记忆）、`tools`（可调用工具）、
  `health`（健康度）——分别复用 `graph.aget_state` / `store.asearch` / 持有的 tools /
  `ops.collect_health`，不重写解析。

装配交给 `build_default_service`：一行起一个离线可跑（mock 模型 + 内存存储）的实例。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from robot_agent.driver import (
    Driver,
    PriorityInbox,
    PromptIdlePolicy,
    StandbyPolicy,
    resume_event,
    user_message,
)
from robot_agent.driver.loop import TurnResult
from robot_agent.graph import build_robot_agent
from robot_agent.memory import P1_KINDS, ns
from robot_agent.ops.health import collect_health
from robot_agent.ops.journal import journal_entry_from_turn

DEFAULT_USER_THREAD = "user_msg"

# store.asearch 分页步长（默认 limit=10 会静默截断，按此步长拉满）。
_PAGE = 100


class AgentService:
    """常驻 agent 的通道无关门面。装配好 graph + driver + 存储后，对外只暴露稳定能力。

    用法：`AgentService(...)` 构造 → `bind_driver(driver)` 接上常驻引擎 → `await start()`。
    通道（web/im/语音）只依赖本类的方法，不直接碰 graph/driver 内部。
    """

    def __init__(
        self,
        *,
        graph: Any,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        tools: Sequence[Any] = (),
        robot_id: str = "robot-1",
    ) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self.store = store
        self._tools = list(tools)
        self.robot_id = robot_id
        self._driver: Driver | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue] = set()
        # 订阅集合被 driver 事件循环线程（_broadcast 遍历）与 HTTP 工作线程
        # （subscribe/unsubscribe 增删）并发访问，用锁护住增删与快照，避免
        # 「Set changed size during iteration」崩掉常驻 driver。
        self._sub_lock = threading.Lock()

    # —— 装配衔接 ————————————————————————————————————————————————

    def bind_driver(self, driver: Driver) -> None:
        """接上常驻 `Driver`（其 `on_turn` 应为本服务的 `on_turn`，见 build_default_service）。"""
        self._driver = driver

    async def on_turn(self, turn: TurnResult) -> None:
        """driver 回合后钩子：把回合精简成事件 dict，广播给所有订阅者。"""
        self._broadcast(self._turn_to_event(turn))

    # —— 生命周期 ————————————————————————————————————————————————

    async def start(self) -> None:
        """后台跑 `driver.run()`（常驻主循环）。记录运行 loop 供跨线程通道桥接。"""
        if self._driver is None:
            raise RuntimeError("调用 start() 前必须先 bind_driver()。")
        if self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._driver.run())

    async def stop(self) -> None:
        """请求停止主循环并等待后台任务收束。"""
        if self._driver is not None:
            self._driver.stop()
        if self._task is not None:
            # start() 紧邻 stop() 时，刚 create_task 的 run() 可能尚未执行，
            # 而 run() 启动时会把 _running 重置为 True，覆盖上面的 stop()。让出一次
            # 调度让 run() 真正进入循环，再 stop 一次，确保它能在下一轮检查时退出。
            await asyncio.sleep(0)
            if self._driver is not None:
                self._driver.stop()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, TimeoutError):
                self._task.cancel()
            self._task = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """常驻 driver 所在事件循环（同步通道用 run_coroutine_threadsafe 跨线程调用）。"""
        return self._loop

    # —— 输入侧 ————————————————————————————————————————————————

    async def submit_user_text(
        self, text: str, *, thread_id: str = DEFAULT_USER_THREAD, priority: int = 0
    ) -> None:
        """把用户文本包成 `user_msg` 事件投递给常驻 driver（复用 driver.user_message）。"""
        if self._driver is None:
            raise RuntimeError("服务未绑定 driver。")
        await self._driver.submit(
            user_message(text, thread_id=thread_id, priority=priority)
        )

    async def resume(self, thread_id: str, decision: Any) -> None:
        """对被安全门控暂停的线程投递 resume 确认（复用 driver.resume_event）。"""
        if self._driver is None:
            raise RuntimeError("服务未绑定 driver。")
        await self._driver.submit(resume_event(thread_id, decision))

    # —— 输出侧（广播 fan-out）————————————————————————————————————

    def subscribe(self) -> asyncio.Queue:
        """注册一个订阅队列，接收后续每个回合的精简事件 dict（SSE/IM/语音共用）。"""
        q: asyncio.Queue = asyncio.Queue()
        with self._sub_lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """注销订阅队列（通道断开时调用，避免泄漏与无谓堆积）。"""
        with self._sub_lock:
            self._subscribers.discard(q)

    def _broadcast(self, event: dict) -> None:
        """把事件无阻塞地推给所有订阅者（慢消费者满了即丢，不阻塞 driver 主循环）。"""
        with self._sub_lock:
            subscribers = list(self._subscribers)  # 锁内取快照，遍历在锁外
        for q in subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - 默认无界，仅防御
                pass

    @staticmethod
    def _turn_to_event(turn: TurnResult) -> dict:
        """把一个 `TurnResult` 抽成精简、可序列化的事件 dict（复用 journal 的消息解析）。"""
        entry = journal_entry_from_turn(turn)
        return {
            "type": "turn",
            "index": turn.index,
            "thread_id": turn.thread_id,
            "event_kind": turn.event.kind,
            "from_idle": turn.from_idle,
            "interrupted": turn.interrupted,
            "reply": entry.outcome,
            "tools": entry.decisions,
        }

    # —— 只读视图 ————————————————————————————————————————————————

    async def history(self, thread_id: str = DEFAULT_USER_THREAD) -> list[dict]:
        """读某线程的会话历史（短期记忆）：复用编译图的 `aget_state` 快照里的 messages。"""
        if self.checkpointer is None:
            return []
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.graph.aget_state(config)
        values = getattr(snapshot, "values", None) or {}
        messages = values.get("messages", []) if isinstance(values, dict) else []
        return [self._message_to_dict(m) for m in messages]

    async def memory(self) -> dict[str, list[dict]]:
        """读长期记忆三类（facts/episodic/prefs）：复用 memory.ns + store.asearch。"""
        out: dict[str, list[dict]] = {kind: [] for kind in P1_KINDS}
        if self.store is None:
            return out
        for kind in P1_KINDS:
            # asearch 默认 limit=10；分页拉满，避免记录数 >10 时静默丢失（同 GoalStore.list）。
            namespace = ns(self.robot_id, kind)
            offset = 0
            items: list = []
            while True:
                batch = await self.store.asearch(namespace, limit=_PAGE, offset=offset)
                items.extend(batch)
                if len(batch) < _PAGE:
                    break
                offset += _PAGE
            out[kind] = [
                {"key": getattr(it, "key", None), "value": getattr(it, "value", None)}
                for it in items
            ]
        return out

    def tools(self) -> list[dict]:
        """列出可调用工具（名称 + 描述）：直接读装配时持有的工具列表。"""
        return [
            {
                "name": getattr(t, "name", str(t)),
                "description": getattr(t, "description", ""),
            }
            for t in self._tools
        ]

    async def health(self) -> dict:
        """健康度快照：复用 ops.collect_health（含暂停线程数等运行态）。"""
        report = collect_health(driver=self._driver)
        mapping = report.to_dict()
        if self._driver is not None:
            # 覆盖 int 计数为线程名列表，前端可直接展示是哪些线程在等确认。
            mapping["pending_threads"] = sorted(self._driver.pending_threads)
        return mapping

    @staticmethod
    def _message_to_dict(msg: Any) -> dict:
        """把一条 langchain 消息序列化为前端可读 dict（role/content/tool_calls）。"""
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)
        tool_calls = getattr(msg, "tool_calls", None) or []
        return {
            "role": getattr(msg, "type", "unknown"),
            "content": content,
            "tool_calls": [tc.get("name", "") for tc in tool_calls],
        }


class _OfflineEchoModel(BaseChatModel):
    """离线兜底模型：每次都回一句固定话术（无工具调用），永不耗尽。

    用作 `build_default_service` 的默认大脑，让 `python -m robot_agent.frontends.web`
    无密钥、无网络也能把控制台**交互式**点亮（可对话、看历史/记忆/工具）。要真实推理，
    传 `model=make_model("fast")` 或经仓库根 `.env` 的 `LLM_*` 配置真实模型。
    """

    @property
    def _llm_type(self) -> str:
        return "offline-echo"

    def bind_tools(self, tools, **kwargs):
        # 兜底模型从不发起工具调用，忽略绑定、原样返回自身（满足 create_react_agent 装配）。
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        reply = (
            "（离线兜底模型）我已收到你的消息。要让我真正思考与下发动作，"
            "请用 model=make_model('fast') 或在 .env 配置 LLM_* 接入真实模型。"
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=reply))]
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)


def build_default_service(
    *,
    model: Any | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    robot_id: str = "robot-1",
    idle_prompt: str | None = None,
    idle_tick: float = 1.0,
) -> AgentService:
    """一行装配一个离线可跑的 `AgentService`（默认离线兜底模型 + 内存存储 + 常驻 driver）。

    - `model` 缺省用 `_OfflineEchoModel`：无密钥/无网络即可**交互式**点亮控制台；
      真实推理传 `model=make_model("fast")` 或经 `.env` 的 `LLM_*` 配置。
    - `checkpointer`/`store` 缺省用内存实现；嵌入式落盘可传 sqlite 版。
    - `idle_prompt` 非空时空闲自发开回合（体现「自主个体」），否则待机最省电。

    返回已 `bind_driver` 的服务；调用方 `await service.start()` 即常驻。
    """
    if model is None:
        model = _OfflineEchoModel()
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
    if store is None:
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore()

    graph = build_robot_agent(
        model=model,
        checkpointer=checkpointer,
        store=store,
        robot_id=robot_id,
    )
    # graph 内部不暴露 tools 列表，这里重建一份等价工具视图供 tools() 只读展示。
    from robot_agent.hal import build_effectors
    from robot_agent.memory import build_memory_tools
    from robot_agent.tools import build_robot_tools

    tools = list(build_robot_tools(build_effectors("mock")))
    tools += build_memory_tools(robot_id)

    service = AgentService(
        graph=graph,
        checkpointer=checkpointer,
        store=store,
        tools=tools,
        robot_id=robot_id,
    )
    idle_policy = PromptIdlePolicy(idle_prompt) if idle_prompt else StandbyPolicy()
    driver = Driver(
        graph,
        PriorityInbox(),
        idle_policy=idle_policy,
        idle_tick=idle_tick,
        on_turn=service.on_turn,
    )
    service.bind_driver(driver)
    return service
