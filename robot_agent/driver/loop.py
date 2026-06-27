"""常驻 driver 主循环（设计 §8.1，实现计划 §P4 任务③）：被动库 → 自主个体。

`Driver` 把「收件箱 + 空闲策略 + 回合编排 + 已编译图」组装成一个能**自己转起来**的循环：

    while running:
        e = await inbox.get(timeout=idle_tick)   # 有事件则醒，超时进空闲策略
        if e is None: e = await idle_policy.on_idle()   # 待机 / 自发回合
        if e is None: continue                   # 纯待机，本 tick 不动
        thread_id = decide_thread(e)             # 决定开哪个回合（衔接 P2 恢复）
        await graph.ainvoke(make_input(e), {thread_id})
        await on_turn(turn)                      # 回合后钩子（复盘 P6 / 日记 P10 挂这里）

设计上 driver 不关心图内部——传入任意已编译的 `create_react_agent`（含 checkpointer/store/
safety）即可。一个回合内的失败（工具/interrupt）由图自身处理；driver 只负责「下一个回合」。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

from robot_agent.driver.events import Event, Inbox
from robot_agent.driver.policy import (
    IdlePolicy,
    StandbyPolicy,
    default_decide_thread,
    default_make_input,
)

# 默认空闲心跳：无事件时每 1s 醒一次评估空闲策略（嵌入式可按需放大以省电）。
DEFAULT_IDLE_TICK = 1.0


@dataclass
class TurnResult:
    """一个回合的产出：触发事件、所开线程、图返回值、是否由空闲策略发起、是否被安全门控暂停。"""

    index: int
    event: Event
    thread_id: str
    result: Any
    from_idle: bool
    interrupted: bool = False


class Driver:
    """常驻自主引擎：消费收件箱、按空闲策略自发开回合、编排 `thread_id` 并调图。

    - `graph`：已编译的 agent（需带 checkpointer 以支持多回合/恢复）。
    - `inbox`：事件来源（外部经 `submit`/`pump_sensor` 投递）。
    - `idle_policy`：空闲时的行为（默认待机）。
    - `decide_thread` / `make_input`：回合编排（默认见 policy.py）。
    - `on_turn`：每回合后的 async 钩子（复盘/日记/指标的挂载点）。
    - `on_error`：回合抛错时的 async 钩子；未提供则原样抛出（resident 部署应提供以续命）。
    """

    def __init__(
        self,
        graph: Any,
        inbox: Inbox,
        *,
        idle_policy: IdlePolicy | None = None,
        idle_tick: float = DEFAULT_IDLE_TICK,
        decide_thread: Callable[[Event], str] = default_decide_thread,
        make_input: Callable[[Event], dict] = default_make_input,
        on_turn: Callable[[TurnResult], Awaitable[None]] | None = None,
        on_error: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        self.graph = graph
        self.inbox = inbox
        self.idle_policy: IdlePolicy = idle_policy or StandbyPolicy()
        self.idle_tick = idle_tick
        self.decide_thread = decide_thread
        self.make_input = make_input
        self.on_turn = on_turn
        self.on_error = on_error
        self._running = False
        self._turn = 0
        # 被安全门控（§7 interrupt）暂停、等待 resume 的线程。对这些线程，下一个事件
        # 应作为 Command(resume=...) 路由，而非灌入新消息（否则会撞 INVALID_CHAT_HISTORY）。
        self._pending: set[str] = set()

    async def submit(self, event: Event) -> None:
        """外部投递一个事件到收件箱（用户指令、传感事件、目标到期…）。"""
        await self.inbox.put(event)

    @staticmethod
    def _is_interrupted(result: Any) -> bool:
        """图返回值是否含未决 interrupt（安全门控暂停，等待 Command(resume)）。"""
        return bool(isinstance(result, dict) and result.get("__interrupt__"))

    async def run_once(self) -> TurnResult | None:
        """跑一个 tick：取事件（或空闲策略产出），有则开一个回合。

        返回 `TurnResult`（执行了一个回合）或 `None`（本 tick 纯待机）。
        若线程处于安全门控暂停态，则把事件作为 `Command(resume=...)` 路由以续跑该回合。
        """
        event = await self.inbox.get(timeout=self.idle_tick)
        from_idle = False
        if event is None:
            idle_event = await self.idle_policy.on_idle()
            # 复查收件箱：on_idle 的 await 期间可能有外部事件入队（空闲事件绕开了优先级队列），
            # 应让真实事件优先；空闲事件让位，下个 tick 再生。
            event = await self.inbox.get(timeout=0)
            if event is None:
                event = idle_event
                from_idle = True
            if event is None:
                return None  # 待机：本 tick 不开回合

        thread_id = self.decide_thread(event)
        config = {"configurable": {"thread_id": thread_id}}

        if thread_id in self._pending:
            # 线程在等安全确认：仅**显式 resume 事件**（payload 带 "resume"）可续跑；
            # 其它事件（含空闲重发的同线程目标事件）不能应用——否则会把未决 interrupt
            # 当成畸形 resume 而 fail-closed 否决，或灌入新消息撞 INVALID_CHAT_HISTORY。跳过留待确认。
            if "resume" not in event.payload:
                return None
            graph_input: Any = Command(resume=event.payload["resume"])
        else:
            graph_input = self.make_input(event)

        result = await self.graph.ainvoke(graph_input, config)

        interrupted = self._is_interrupted(result)
        if interrupted:
            self._pending.add(thread_id)  # 仍未决：等下一个 resume 事件
        else:
            self._pending.discard(thread_id)  # 已收束：解除暂停态

        self._turn += 1
        turn = TurnResult(
            index=self._turn,
            event=event,
            thread_id=thread_id,
            result=result,
            from_idle=from_idle,
            interrupted=interrupted,
        )
        if self.on_turn is not None:
            await self.on_turn(turn)
        return turn

    async def run(self) -> None:
        """常驻主循环，直到 `stop()`。回合抛错经 `on_error` 续命（未配则抛出）。"""
        self._running = True
        while self._running:
            try:
                await self.run_once()
            except Exception as exc:
                if self.on_error is None:
                    raise
                await self.on_error(exc)

    def stop(self) -> None:
        """请求停止主循环（下一次循环检查时退出；可在 `on_turn` 内调用）。"""
        self._running = False

    @property
    def turns(self) -> int:
        """已执行的回合数（不含纯待机 tick）。"""
        return self._turn

    @property
    def pending_threads(self) -> frozenset[str]:
        """当前被安全门控暂停、等待 resume 的线程集合（只读快照）。"""
        return frozenset(self._pending)
