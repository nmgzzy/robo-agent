"""P5 验收：目标系统（FR-8）。

对应 docs/IMPLEMENTATION_PLAN.md §P5 与 docs/ROBOT_AGENT_DESIGN.md §8.2。覆盖：

- **Goal 持久化**：GoalStore add/get/list/update/mark/set_plan/remove（含状态过滤、隔离）。
- **多目标仲裁**：按 priority > deadline > created_ts 选当前目标，只在可推进目标中选。
- **规划/重规划**：plan_goal 把意图分解为步骤（去编号/项目符号，限步数）。
- **打断恢复（验收②）**：driver 空闲推进高优先目标 → 紧急事件打断 → 处理完恢复到原目标。

全部离线（Mock LLM + Mock HAL + 内存 SQLite）。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from robot_agent import (
    Driver,
    Event,
    Goal,
    GoalDrivenIdlePolicy,
    GoalStore,
    PriorityInbox,
    SafetyPolicy,
    arbitrate,
    build_effectors,
    build_robot_agent,
    make_model,
    plan_goal,
)
from robot_agent.driver import resume_event
from robot_agent.goals import STATUS_ACTIVE, STATUS_DONE, order_goals


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# --------------------------------------------------------------------------- #
# 仲裁（纯函数）
# --------------------------------------------------------------------------- #


def test_arbitrate_priority_then_deadline_then_age():
    goals = [
        Goal(id="lowpri", intent="a", priority=1, created_ts=1.0),
        Goal(id="highpri", intent="b", priority=9, created_ts=2.0),
        Goal(id="samepri_late", intent="c", priority=9, deadline=100.0, created_ts=3.0),
        Goal(id="samepri_early", intent="d", priority=9, deadline=10.0, created_ts=4.0),
    ]
    # 最高优先级 + 最早 deadline 胜出。
    assert arbitrate(goals).id == "samepri_early"
    # 全序：两个 pri=9（按 deadline 早→晚），再 pri=1。
    assert [g.id for g in order_goals(goals)] == [
        "samepri_early",
        "samepri_late",
        "highpri",
        "lowpri",
    ]


def test_arbitrate_skips_non_actionable_and_empty():
    goals = [
        Goal(id="done", intent="x", priority=99, status=STATUS_DONE),
        Goal(id="blocked", intent="y", priority=99, status="blocked"),
        Goal(id="ok", intent="z", priority=1),
    ]
    assert arbitrate(goals).id == "ok"  # 高优先但终态/受阻者被跳过
    assert arbitrate([]) is None
    assert arbitrate([Goal(id="d", intent="x", status=STATUS_DONE)]) is None


# --------------------------------------------------------------------------- #
# GoalStore CRUD
# --------------------------------------------------------------------------- #


async def test_goal_store_crud_roundtrip():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        await gs.add(Goal(id="g1", intent="充电", priority=5))
        await gs.add(Goal(id="g2", intent="巡逻", priority=3))

        assert (await gs.get("g1")).intent == "充电"
        assert {g.id for g in await gs.list()} == {"g1", "g2"}

        await gs.mark("g1", STATUS_ACTIVE)
        assert (await gs.get("g1")).status == STATUS_ACTIVE
        assert [g.id for g in await gs.list(status=STATUS_ACTIVE)] == ["g1"]

        await gs.set_plan("g2", ["去A区", "去B区"])
        assert (await gs.get("g2")).plan == ["去A区", "去B区"]

        await gs.remove("g2")
        assert await gs.get("g2") is None


async def test_goal_store_update_unknown_raises():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        with pytest.raises(KeyError, match="未知目标"):
            await gs.update("missing", status=STATUS_ACTIVE)


async def test_goal_store_get_is_isolated():
    # 改 get 返回的 plan（嵌套 list）不得回灌污染已存目标。
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        await gs.add(Goal(id="g1", intent="x", plan=["s1"]))
        got = await gs.get("g1")
        got.plan.append("篡改")
        again = await gs.get("g1")
    assert again.plan == ["s1"]


async def test_goal_store_add_stamps_created_ts():
    # 未给 created_ts 时由 add 补 UTC epoch，使年龄 tiebreak 生效（codex review P2）。
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        g = await gs.add(Goal(id="g1", intent="x"))
        assert g.created_ts > 0.0
        assert (await gs.get("g1")).created_ts == g.created_ts
        # 显式给定则尊重调用方。
        g2 = await gs.add(Goal(id="g2", intent="y", created_ts=5.0))
        assert g2.created_ts == 5.0


async def test_goal_store_lists_more_than_ten_goals():
    # asearch 默认 limit=10；分页必须拉满，否则会漏选最高优先目标（codex review P2）。
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        for i in range(25):
            await gs.add(Goal(id=f"g{i:02d}", intent=f"任务{i}", priority=i))
        goals = await gs.list()
        assert len(goals) == 25
        # 仲裁应能看到第 24 号（最高优先）而非被分页截断。
        assert arbitrate(goals).id == "g24"


# --------------------------------------------------------------------------- #
# 规划 / 重规划
# --------------------------------------------------------------------------- #


async def test_plan_goal_parses_steps():
    model = make_model(
        responses=[AIMessage("1. 走到充电桩\n2) 对接\n- 开始充电\n\n  ")]
    )
    steps = await plan_goal(model, Goal(id="g1", intent="去充电"))
    assert steps == ["走到充电桩", "对接", "开始充电"]


async def test_plan_goal_respects_max_steps():
    model = make_model(responses=[AIMessage("a\nb\nc\nd\ne")])
    steps = await plan_goal(model, Goal(id="g1", intent="x"), max_steps=3)
    assert steps == ["a", "b", "c"]


async def test_plan_goal_rejects_resilience_fallback():
    """决策大脑降级话术不得被当作步骤（codex review）。"""
    from robot_agent.reliability import DEFAULT_FALLBACK_TEXT

    model = make_model(responses=[AIMessage(DEFAULT_FALLBACK_TEXT)])
    assert await plan_goal(model, Goal(id="g1", intent="x")) == []


async def test_idle_planning_does_not_persist_fallback():
    """planner 降级时不持久化 plan，保持可重规划（codex review）。"""
    from robot_agent.reliability import DEFAULT_FALLBACK_TEXT

    planner = make_model(responses=[AIMessage(DEFAULT_FALLBACK_TEXT)])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        await gs.add(Goal(id="g1", intent="拿杯子"))
        event = await GoalDrivenIdlePolicy(gs, planner_model=planner).on_idle()
        assert (await gs.get("g1")).plan == []  # 降级未被持久化，可重试
    assert event.payload["text"] == "拿杯子"  # 回退到纯 intent


async def test_plan_goal_handles_structured_content():
    # content 为内容块列表（如 Anthropic 扩展思考）时应抽取文本而非 str(list)（codex review）。
    model = make_model(
        responses=[
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "略"},
                    {"type": "text", "text": "步骤一\n步骤二"},
                ]
            )
        ]
    )
    steps = await plan_goal(model, Goal(id="g1", intent="x"))
    assert steps == ["步骤一", "步骤二"]


# --------------------------------------------------------------------------- #
# 与 driver 衔接：空闲推进目标 + 打断恢复（验收①②）
# --------------------------------------------------------------------------- #


async def test_ac_idle_pursues_highest_priority_goal():
    """注入两个竞争目标，driver 空闲时按优先级选对当前目标并标记 active（验收①）。"""
    robot_id = "robot-1"
    model = make_model(responses=[AIMessage("正在推进充电")])
    async with (
        AsyncSqliteSaver.from_conn_string(":memory:") as saver,
        AsyncSqliteStore.from_conn_string(":memory:") as store,
    ):
        gs = GoalStore(store, robot_id)
        await gs.add(Goal(id="charge", intent="去充电", priority=10))
        await gs.add(Goal(id="tidy", intent="整理房间", priority=5))

        agent = build_robot_agent(
            model=model, store=store, checkpointer=saver, robot_id=robot_id
        )
        driver = Driver(
            agent, PriorityInbox(), idle_policy=GoalDrivenIdlePolicy(gs), idle_tick=0.01
        )
        turn = await driver.run_once()

        assert turn.from_idle is True
        assert turn.thread_id == "goal-charge"  # 选了高优先目标
        assert (await gs.get("charge")).status == STATUS_ACTIVE


async def test_ac_urgent_event_preempts_then_recovers_to_goal():
    """空闲推进目标 A → 紧急事件打断优先处理 → 处理完下个空闲 tick 恢复到 A（验收②）。"""
    robot_id = "robot-1"
    effectors = build_effectors("mock")
    model = make_model(
        responses=[
            AIMessage("推进充电-1"),  # turn1：空闲推进 A
            AIMessage("处理呼救"),  # turn2：紧急事件
            AIMessage("推进充电-2"),  # turn3：恢复 A
        ]
    )
    inbox = PriorityInbox()
    async with (
        AsyncSqliteSaver.from_conn_string(":memory:") as saver,
        AsyncSqliteStore.from_conn_string(":memory:") as store,
    ):
        gs = GoalStore(store, robot_id)
        await gs.add(Goal(id="charge", intent="去充电", priority=10))
        await gs.add(Goal(id="tidy", intent="整理", priority=5))

        agent = build_robot_agent(
            model=model,
            effectors=effectors,
            store=store,
            checkpointer=saver,
            robot_id=robot_id,
        )
        driver = Driver(
            agent, inbox, idle_policy=GoalDrivenIdlePolicy(gs), idle_tick=0.01
        )

        t1 = await driver.run_once()  # 空闲 → 推进 A
        await driver.submit(
            Event(kind="sensor", payload={"text": "有人呼救"}, priority=100)
        )
        t2 = await driver.run_once()  # 紧急事件打断
        t3 = await driver.run_once()  # 空闲 → 恢复 A

    assert (t1.from_idle, t1.thread_id) == (True, "goal-charge")
    assert (t2.from_idle, t2.thread_id) == (False, "sensor")  # 紧急优先，非空闲
    assert (t3.from_idle, t3.thread_id) == (True, "goal-charge")  # 恢复到被打断目标


# --------------------------------------------------------------------------- #
# 规划接线：分解 → 持久化 plan → 注入回合 → 逐步执行
# --------------------------------------------------------------------------- #


async def test_idle_planning_decomposes_and_persists():
    """配 planner_model 时，首次推进目标会分解并持久化 plan，步骤注入回合指令。"""
    planner = make_model(responses=[AIMessage("1. 走到桌前\n2. 抓起杯子")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        await gs.add(Goal(id="g1", intent="拿杯子", priority=5))
        policy = GoalDrivenIdlePolicy(gs, planner_model=planner)
        event = await policy.on_idle()

        # plan 已分解并持久化。
        assert (await gs.get("g1")).plan == ["走到桌前", "抓起杯子"]
    # 步骤注入了回合指令。
    assert "计划步骤" in event.payload["text"]
    assert "走到桌前" in event.payload["text"] and "抓起杯子" in event.payload["text"]


async def test_idle_planning_skips_when_already_planned():
    """已有 plan 的目标不重复分解（planner 不被调用）。"""
    planner = make_model(responses=[])  # 若误调用会因响应耗尽抛错
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        await gs.add(Goal(id="g1", intent="x", plan=["已有步骤A", "已有步骤B"]))
        policy = GoalDrivenIdlePolicy(gs, planner_model=planner)
        event = await policy.on_idle()
    assert "已有步骤A" in event.payload["text"]  # 沿用已存 plan


async def test_idle_without_planner_keeps_p5_behavior():
    """未配 planner_model 时行为同 P5：只注入 intent，不分解。"""
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        await gs.add(Goal(id="g1", intent="纯意图", priority=1))
        event = await GoalDrivenIdlePolicy(gs).on_idle()
        assert event.payload["text"] == "纯意图"
        assert (await gs.get("g1")).plan == []


async def test_planned_goal_drives_execution_in_driver():
    """完整闭环：目标 → 分解 → 注入回合 → agent 按计划下发动作。"""
    robot_id = "robot-1"
    effectors = build_effectors("mock")
    planner = make_model(responses=[AIMessage("1. 移动到目标\n2. 抓取")])
    turn_model = make_model(
        responses=[
            _tool_call("move_to", {"x": 1.0, "y": 1.0}, "a"),
            _tool_call("grasp", {"obj": "box"}, "b"),
            AIMessage("完成"),
        ]
    )
    async with (
        AsyncSqliteSaver.from_conn_string(":memory:") as saver,
        AsyncSqliteStore.from_conn_string(":memory:") as store,
    ):
        gs = GoalStore(store, robot_id)
        await gs.add(Goal(id="fetch", intent="取箱子", priority=10))
        agent = build_robot_agent(
            model=turn_model,
            effectors=effectors,
            store=store,
            checkpointer=saver,
            robot_id=robot_id,
        )
        driver = Driver(
            agent,
            PriorityInbox(),
            idle_policy=GoalDrivenIdlePolicy(gs, planner_model=planner),
            idle_tick=0.01,
        )
        turn = await driver.run_once()
        planned = await gs.get("fetch")

    assert turn.thread_id == "goal-fetch"
    assert planned.plan == ["移动到目标", "抓取"]  # 目标被分解持久化
    # 计划注入了回合输入。
    assert any("计划步骤" in getattr(m, "content", "") for m in turn_model.received[0])
    # agent 按计划执行了底层动作。
    assert effectors["base"].log == [{"action": "move_to", "x": 1.0, "y": 1.0}]
    assert effectors["arm"].log == [{"action": "grasp", "target": "box"}]


async def test_idle_does_not_auto_resolve_pending_safety_interrupt():
    """空闲重发的同线程目标事件不得把待安全确认的 interrupt 当 resume 否决（codex review P1）。"""
    robot_id = "robot-1"
    effectors = build_effectors("mock")
    model = make_model(
        responses=[_tool_call("grasp", {"obj": "cup"}, "g1"), AIMessage("已抓取")]
    )
    async with (
        AsyncSqliteSaver.from_conn_string(":memory:") as saver,
        AsyncSqliteStore.from_conn_string(":memory:") as store,
    ):
        gs = GoalStore(store, robot_id)
        await gs.add(Goal(id="grab", intent="抓杯子", priority=10))
        agent = build_robot_agent(
            model=model,
            effectors=effectors,
            store=store,
            checkpointer=saver,
            robot_id=robot_id,
            safety=SafetyPolicy(),
        )
        driver = Driver(
            agent, PriorityInbox(), idle_policy=GoalDrivenIdlePolicy(gs), idle_tick=0.01
        )

        t1 = await driver.run_once()  # 空闲推进 grab → grasp 被门控暂停
        assert t1.interrupted is True and driver.pending_threads == frozenset(
            {"goal-grab"}
        )
        assert effectors["arm"].log == []

        # 下个空闲 tick 会重发 goal-grab 事件：必须被跳过，不得自动否决/清除 interrupt。
        t2 = await driver.run_once()
        assert t2 is None
        assert driver.pending_threads == frozenset({"goal-grab"})
        assert effectors["arm"].log == []

        # 仅显式 resume 事件才放行。
        await driver.submit(resume_event("goal-grab", {"approved": True}))
        await driver.run_once()
        assert effectors["arm"].log == [{"action": "grasp", "target": "cup"}]


async def test_idle_yields_to_event_arriving_during_selection():
    """on_idle 选目标期间到达的紧急事件应优先处理，不被空闲回合挤后（codex review P1）。"""
    inbox = PriorityInbox()

    class _InjectingPolicy:
        """模拟 on_idle 的 await 窗口内有外部事件入队。"""

        async def on_idle(self):
            await inbox.put(
                Event(kind="sensor", payload={"text": "紧急"}, priority=100)
            )
            return Event(kind="timer", payload={"text": "空闲巡视"})

    class _FakeGraph:
        async def ainvoke(self, inp, config):
            return {"messages": [AIMessage("ok")]}

    driver = Driver(_FakeGraph(), inbox, idle_policy=_InjectingPolicy(), idle_tick=0.01)
    turn = await driver.run_once()

    # 复查收件箱后处理的是紧急事件（sensor），而非空闲事件（timer）。
    assert turn is not None and turn.from_idle is False
    assert turn.thread_id == "sensor"
