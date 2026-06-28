"""P10 验收：技能库 + 运维可观测（FR-13 / FR-15）。

对应 docs/IMPLEMENTATION_PLAN.md §P10 与 docs/ROBOT_AGENT_DESIGN.md §8.7/§8.9。覆盖：

- **技能库**：SkillStore 存/检索；build_skill_tools 动态装配；治理校验。
- **检索复用（验收）**：成功计划存为技能后，新场景检索 → 动态加载 → 复用执行。
- **决策日记（验收）**：make_journal_hook 记录每回合（含暂停）；replay 还原决策链。
- **健康度指标（验收）**：collect_health 聚合各域指标，可被外部读取。

全部离线（Mock LLM + Mock HAL + 内存 SQLite）。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from robot_agent import (
    DecisionJournal,
    Driver,
    GovernancePolicy,
    HealthReport,
    MetacogPolicy,
    PriorityInbox,
    Skill,
    SkillStore,
    build_effectors,
    build_robot_agent,
    build_skill_tools,
    collect_health,
    make_journal_hook,
    make_model,
    user_message,
)
from robot_agent.governance.policy import ToolPermission
from robot_agent.ops import introspect, journal_entry_from_turn


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


_FETCH_CUP = Skill(
    id="s1",
    name="fetch_cup",
    description="去取杯子：先移动到桌前再抓取",
    actions=[
        {"tool": "move_to", "args": {"x": 1.0, "y": 2.0}},
        {"tool": "grasp", "args": {"target": "cup"}},
    ],
)


# --------------------------------------------------------------------------- #
# 技能存储与检索
# --------------------------------------------------------------------------- #


async def test_skill_store_crud_and_search():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        ss = SkillStore(store, "r1")
        added = await ss.add(_FETCH_CUP)
        assert added.created_ts > 0.0  # add 补时间戳
        assert (await ss.get("s1")).name == "fetch_cup"
        assert [s.id for s in await ss.list()] == ["s1"]
        # 关键字检索命中 description。
        assert [s.id for s in await ss.search("杯子")] == ["s1"]
        assert await ss.search("不存在") == []


# --------------------------------------------------------------------------- #
# 动态工具加载 + 治理
# --------------------------------------------------------------------------- #


async def test_build_skill_tools_executes_actions():
    effectors = build_effectors("mock")
    tools = build_skill_tools([_FETCH_CUP], effectors)
    assert tools[0].name == "skill_fetch_cup"

    out = await tools[0].ainvoke({})
    assert "执行完成" in out
    assert effectors["base"].log == [{"action": "move_to", "x": 1.0, "y": 2.0}]
    assert effectors["arm"].log == [{"action": "grasp", "target": "cup"}]


def test_build_skill_tools_rejects_duplicate_names():
    # 同名技能会令 ToolNode 静默覆盖，须在装配时 fail-fast（codex review）。
    effectors = build_effectors("mock")
    dup = Skill(id="s2", name="fetch_cup", description="另一个同名技能")
    with pytest.raises(ValueError, match="技能工具名冲突"):
        build_skill_tools([_FETCH_CUP, dup], effectors)


async def test_skill_tool_respects_governance():
    effectors = build_effectors("mock")
    governance = GovernancePolicy(
        permission=ToolPermission(denied=frozenset({"grasp"}))
    )
    tools = build_skill_tools([_FETCH_CUP], effectors, governance=governance)

    out = await tools[0].ainvoke({})
    # grasp 被治理拒绝 → 技能中止；move_to 已执行，grasp 未下发。
    assert "中止" in out
    assert effectors["base"].log == [{"action": "move_to", "x": 1.0, "y": 2.0}]
    assert effectors["arm"].log == []


# --------------------------------------------------------------------------- #
# 验收：存技能 → 检索 → 动态加载 → 复用
# --------------------------------------------------------------------------- #


async def test_ac_store_retrieve_and_reuse_skill():
    """成功计划存为技能 → 新场景检索 → 动态装配进 agent → 模型调用复用。"""
    effectors = build_effectors("mock")
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        ss = SkillStore(store, "r1")
        await ss.add(_FETCH_CUP)  # 沉淀成功计划

        # 新场景：检索相关技能并动态加载。
        found = await ss.search("取杯子")
        skill_tools = build_skill_tools(found, effectors)

        model = make_model(
            responses=[
                _tool_call("skill_fetch_cup", {}, "k1"),  # 模型选用预存技能
                AIMessage("杯子拿好了"),
            ]
        )
        agent = build_robot_agent(
            model=model, effectors=effectors, store=store, extra_tools=skill_tools
        )
        out = await agent.ainvoke({"messages": [HumanMessage("帮我拿杯子")]})

    # 技能复用：底层两步动作都下发了。
    assert effectors["base"].log == [{"action": "move_to", "x": 1.0, "y": 2.0}]
    assert effectors["arm"].log == [{"action": "grasp", "target": "cup"}]
    assert out["messages"][-1].content == "杯子拿好了"


# --------------------------------------------------------------------------- #
# 决策日记 + 运行时自省
# --------------------------------------------------------------------------- #


def test_journal_entry_from_turn():
    class _T:
        index = 1
        thread_id = "user_msg"
        interrupted = False
        event = user_message("去A区")
        result = {
            "messages": [
                HumanMessage("去A区"),
                _tool_call("move_to", {"x": 1.0, "y": 0.0}, "c1"),
                AIMessage("到了"),
            ]
        }

    entry = journal_entry_from_turn(_T())
    assert entry.intent == "去A区"
    assert entry.decisions == ["move_to({'x': 1.0, 'y': 0.0})"]
    assert entry.outcome == "到了" and entry.interrupted is False


async def test_ac_journal_replays_decision_chain():
    """决策日记记录每回合，可离线还原一次回合的决策链（验收）。"""
    effectors = build_effectors("mock")
    journal = DecisionJournal()
    model = make_model(
        responses=[
            _tool_call("move_to", {"x": 3.0, "y": 4.0}, "c1"),
            AIMessage("已到达"),
        ]
    )
    inbox = PriorityInbox()
    await inbox.put(user_message("去 (3,4)"))
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(model=model, effectors=effectors, checkpointer=saver)
        driver = Driver(agent, inbox, idle_tick=0.01)
        driver.on_turn = make_journal_hook(journal)
        await driver.run_once()

    chain = journal.replay("user_msg")
    assert len(chain) == 1
    assert chain[0].intent == "去 (3,4)"
    assert chain[0].decisions == ["move_to({'x': 3.0, 'y': 4.0})"]
    assert chain[0].outcome == "已到达"
    # 运行时自省：当前回合数、最近决策。
    snap = introspect(journal, driver=driver)
    assert snap.turns == 1 and snap.latest.intent == "去 (3,4)"


# --------------------------------------------------------------------------- #
# 健康度指标
# --------------------------------------------------------------------------- #


def test_collect_health_aggregates_metrics():
    metacog = MetacogPolicy()
    metacog.metrics.update({"loops": 2, "escalations": 1, "budget_breaches": 0})
    governance = GovernancePolicy(
        permission=ToolPermission(denied=frozenset({"grasp"}))
    )
    governance.check("grasp", {"target": "x"})  # 制造一次拒绝
    journal = DecisionJournal()

    report = collect_health(metacog=metacog, governance=governance, journal=journal)
    assert isinstance(report, HealthReport)
    assert report.loops == 2 and report.escalations == 1 and report.denials == 1
    assert report.context_compactions == 0
    # 可导出为 dict 供远程巡检读取。
    assert report.to_dict()["loops"] == 2
