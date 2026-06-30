"""P4 验收：自主引擎 driver + 收件箱 + 事件总线（FR-7 / AC-4）。

对应 docs/IMPLEMENTATION_PLAN.md §P4 与 docs/ROBOT_AGENT_DESIGN.md §8.1。覆盖：

- **收件箱**：优先级出队（高优先级先出、同级 FIFO）、超时返回 None。
- **AC-4**：无事件时 driver 在 idle_tick 到点后按空闲策略自发开回合；
  高优先级事件能打断空闲、被优先处理。
- **回合编排**：decide_thread / make_input 默认行为；事件型观测经 pump_sensor 汇入。
- **常驻循环**：run() 可被 on_turn 内 stop()；on_error 续命。

全部离线（Mock LLM + Mock HAL + 内存 SQLite），用确定性手段（计数/预载）避免依赖 wall-clock。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from robot_agent import (
    Driver,
    Event,
    PriorityInbox,
    PromptIdlePolicy,
    SafetyPolicy,
    StandbyPolicy,
    build_effectors,
    build_robot_agent,
    make_model,
    user_message,
)
from robot_agent.driver import KIND_TIMER, pump_sensor, resume_event
from robot_agent.driver.policy import default_decide_thread, default_make_input
from robot_agent.hal import ScriptedSensor


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# --------------------------------------------------------------------------- #
# 收件箱：优先级 + 超时
# --------------------------------------------------------------------------- #


async def test_inbox_priority_then_fifo():
    inbox = PriorityInbox()
    await inbox.put(Event(kind="a", priority=0))
    await inbox.put(Event(kind="b", priority=5))  # 更高优先级
    await inbox.put(Event(kind="c", priority=0))
    await inbox.put(Event(kind="d", priority=5))

    got = [(await inbox.get(timeout=None)).kind for _ in range(4)]
    # 先出两个高优先级（b,d 按到达 FIFO），再出两个低优先级（a,c FIFO）。
    assert got == ["b", "d", "a", "c"]


async def test_inbox_get_timeout_returns_none():
    inbox = PriorityInbox()
    assert await inbox.get(timeout=0.01) is None


async def test_inbox_zero_timeout_is_nonblocking_read():
    # timeout=0 必须非阻塞读已入队项，而非误判为空（codex review）。
    inbox = PriorityInbox()
    assert await inbox.get(timeout=0.0) is None
    await inbox.put(Event(kind="x"))
    e = await inbox.get(timeout=0.0)
    assert e is not None and e.kind == "x"


async def test_pump_sensor_routes_event_observations():
    inbox = PriorityInbox()
    cam = ScriptedSensor(
        "camera", [{"ts": 1.0, "obj": "cup"}, {"ts": 2.0, "obj": "ball"}]
    )
    n = await pump_sensor(
        cam, inbox, priority=3, predicate=lambda o: o.payload["obj"] == "cup"
    )
    assert n == 1
    e = await inbox.get(timeout=None)
    assert e.kind == "sensor" and e.payload["obj"] == "cup" and e.priority == 3


async def test_pump_sensor_preserves_source_and_frame():
    # source/frame 在 Observation 字段而非 payload 时不能丢（codex review）。
    inbox = PriorityInbox()

    class _Sensor:
        name = "lidar"

        async def stream(self):
            from robot_agent.hal import Observation

            yield Observation(
                source="lidar", ts=1.0, frame="base_link", payload={"d": 0.4}
            )

    await pump_sensor(_Sensor(), inbox)
    e = await inbox.get(timeout=None)
    assert e.payload["source"] == "lidar" and e.payload["frame"] == "base_link"
    assert e.payload["d"] == 0.4


# --------------------------------------------------------------------------- #
# 回合编排：decide_thread / make_input
# --------------------------------------------------------------------------- #


def test_decide_thread_defaults_to_kind():
    assert default_decide_thread(Event(kind="user_msg")) == "user_msg"
    assert (
        default_decide_thread(Event(kind="timer", payload={"thread_id": "t9"})) == "t9"
    )


def test_make_input_text_and_world_state():
    e = Event(
        kind="sensor",
        payload={"text": "去充电", "battery": 12.0, "pose": {"x": 1.0}},
    )
    inp = default_make_input(e)
    assert isinstance(inp["messages"][0], HumanMessage)
    assert inp["messages"][0].content == "去充电"
    assert inp["battery"] == 12.0 and inp["pose"] == {"x": 1.0}


def test_make_input_falls_back_to_serialized_payload():
    e = Event(kind="sensor", payload={"obj": "cup", "dist": 0.3})
    inp = default_make_input(e)
    assert "cup" in inp["messages"][0].content


# --------------------------------------------------------------------------- #
# AC-4：无事件自发开回合 + 高优先级打断空闲
# --------------------------------------------------------------------------- #


async def test_ac4_idle_policy_opens_turn_without_events():
    """不投递任何事件，driver 在 idle_tick 到点后按空闲策略自发开一个回合。"""
    effectors = build_effectors("mock")
    # 空闲自发回合：模型回一句播报。
    model = make_model(
        responses=[
            _tool_call("speak", {"text": "一切正常"}, "i1"),
            AIMessage("巡视完毕"),
        ]
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(model=model, effectors=effectors, checkpointer=saver)
        driver = Driver(
            agent,
            PriorityInbox(),
            idle_policy=PromptIdlePolicy("请巡视环境并报告"),
            idle_tick=0.01,
        )
        turn = await driver.run_once()

    assert turn is not None and turn.from_idle is True
    assert turn.thread_id == KIND_TIMER
    assert effectors["speaker"].log == [{"action": "speak", "text": "一切正常"}]


async def test_ac4_standby_does_nothing_when_idle():
    """默认待机策略：空闲 tick 不开回合（run_once 返回 None）。"""
    model = make_model(responses=[AIMessage("不该被调用")])
    agent = build_robot_agent(model=model)
    driver = Driver(agent, PriorityInbox(), idle_policy=StandbyPolicy(), idle_tick=0.01)
    assert await driver.run_once() is None
    assert driver.turns == 0
    assert model.received == []  # 图未被调用


async def test_ac4_high_priority_event_preempts_idle():
    """收件箱里有高优先级事件时，driver 先处理它（而非进入空闲策略）。"""
    effectors = build_effectors("mock")
    model = make_model(
        responses=[
            _tool_call("move_to", {"x": 5.0, "y": 0.0}, "u1"),
            AIMessage("已到达"),
        ]
    )
    inbox = PriorityInbox()
    await inbox.put(
        Event(kind="sensor", payload={"text": "有人呼救，去现场"}, priority=10)
    )

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(model=model, effectors=effectors, checkpointer=saver)
        driver = Driver(
            agent, inbox, idle_policy=PromptIdlePolicy("低优先级巡视"), idle_tick=1.0
        )
        turn = await driver.run_once()

    # 处理的是高优先级事件而非空闲巡视。
    assert turn is not None and turn.from_idle is False
    assert turn.thread_id == "sensor"
    assert effectors["base"].log == [{"action": "move_to", "x": 5.0, "y": 0.0}]


# --------------------------------------------------------------------------- #
# 安全门控衔接：被 interrupt 暂停的线程保持未决，下个事件作为 resume 路由
# --------------------------------------------------------------------------- #


async def test_driver_routes_resume_to_interrupted_thread():
    """safety 开启时：危险动作回合被门控暂停 → 线程标记未决 → 下个事件按 resume 续跑。

    回归 codex review [P1]：避免把 __interrupt__ 当完成回合、向暂停 checkpoint 灌新消息
    导致 INVALID_CHAT_HISTORY。
    """
    effectors = build_effectors("mock")
    model = make_model(
        responses=[_tool_call("grasp", {"obj": "cup"}, "g1"), AIMessage("已抓取")]
    )
    inbox = PriorityInbox()
    await inbox.put(user_message("抓杯子"))  # thread_id 默认归并到 "user_msg"

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(
            model=model, effectors=effectors, checkpointer=saver, safety=SafetyPolicy()
        )
        driver = Driver(agent, inbox, idle_tick=0.01)

        # 回合 1：被安全门控暂停。
        turn1 = await driver.run_once()
        assert turn1.interrupted is True
        assert driver.pending_threads == frozenset({"user_msg"})
        assert effectors["arm"].log == []  # 尚未下发

        # 投递确认（高优先级 resume 事件，路由到同一线程）→ 回合 2 续跑并放行。
        await driver.submit(resume_event("user_msg", {"approved": True}))
        turn2 = await driver.run_once()

    assert turn2.interrupted is False
    assert driver.pending_threads == frozenset()
    assert effectors["arm"].log == [{"action": "grasp", "target": "cup"}]
    assert turn2.result["messages"][-1].content == "已抓取"


# --------------------------------------------------------------------------- #
# 常驻循环：run() / stop() / on_error
# --------------------------------------------------------------------------- #


async def test_run_loop_stops_via_on_turn():
    """run() 常驻循环处理预载事件，on_turn 内 stop() 后退出（确定性，无 wall-clock 依赖）。"""
    effectors = build_effectors("mock")
    model = make_model(
        responses=[AIMessage("收到1"), AIMessage("收到2")]  # 两个回合各一句终态
    )
    inbox = PriorityInbox()
    await inbox.put(user_message("任务一"))
    await inbox.put(user_message("任务二"))

    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(model=model, effectors=effectors, checkpointer=saver)
        seen: list[str] = []

        driver = Driver(agent, inbox, idle_tick=0.01)

        async def on_turn(turn):
            seen.append(turn.result["messages"][-1].content)
            if turn.index >= 2:
                driver.stop()

        driver.on_turn = on_turn
        await asyncio.wait_for(driver.run(), timeout=5.0)

    assert seen == ["收到1", "收到2"]
    assert driver.turns == 2


async def test_run_loop_on_error_keeps_alive():
    """回合抛错时 on_error 续命：第一个事件触发错误，第二个仍被处理。"""
    inbox = PriorityInbox()
    await inbox.put(user_message("boom"))
    await inbox.put(user_message("ok"))

    errors: list[Exception] = []
    processed: list[str] = []

    class FakeGraph:
        """首个回合抛错、其后正常的极简假图（直接当 graph 用，绕过真实图）。"""

        def __init__(self):
            self.n = 0

        async def ainvoke(self, inp, config):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("第一个回合炸了")
            processed.append(inp["messages"][0].content)
            return {"messages": [AIMessage("ok")]}

    driver = Driver(FakeGraph(), inbox, idle_tick=0.01)

    async def on_turn(turn):
        if turn.index >= 1:
            driver.stop()

    async def on_error(exc):
        errors.append(exc)

    driver.on_turn = on_turn
    driver.on_error = on_error
    await asyncio.wait_for(driver.run(), timeout=5.0)

    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert processed == ["ok"]
