"""全面 review 后的修复验收（M1–M5 / L1–L5）。

每个用例对应一条 review 发现，先失败后修复（TDD）。命名 `test_<编号>_<行为>`。
全部离线（Mock LLM + Mock HAL + 内存 Store）。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.store.sqlite.aio import AsyncSqliteStore

from robot_agent import (
    GovernancePolicy,
    MetacogPolicy,
    SafetyPolicy,
    build_effectors,
    build_robot_agent,
    detect_loop,
    make_model,
)
from robot_agent.memory import KIND_FACTS, ns
from robot_agent.metacog import make_monitor_hook


def _mv(i: int, x: float = 1.0, y: float = 1.0) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "move_to",
                "args": {"x": x, "y": y},
                "id": f"m{i}",
                "type": "tool_call",
            }
        ],
    )


def _call(name: str, args: dict, cid: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}],
    )


# --------------------------------------------------------------------------- #
# M1：safety / metacog-escalate 依赖 checkpointer，缺失则装配期 fail-fast
# --------------------------------------------------------------------------- #


def test_m1_safety_without_checkpointer_raises():
    model = make_model(responses=[AIMessage("ok")])
    with pytest.raises(ValueError, match="checkpointer"):
        build_robot_agent(model=model, safety=SafetyPolicy())


def test_m1_metacog_escalate_without_checkpointer_raises():
    model = make_model(responses=[AIMessage("ok")])
    with pytest.raises(ValueError, match="checkpointer"):
        build_robot_agent(
            model=model, metacog=MetacogPolicy(max_repeats=3, on_breach="escalate")
        )


def test_m1_metacog_warn_without_checkpointer_ok():
    # warn 模式不走 interrupt，无需 checkpointer：不应抛错。
    model = make_model(responses=[AIMessage("ok")])
    build_robot_agent(
        model=model, metacog=MetacogPolicy(max_repeats=3, on_breach="warn")
    )


# --------------------------------------------------------------------------- #
# M2：记忆写回工具 remember_fact 也要过治理校验（违反直接拒绝、不写库）
# --------------------------------------------------------------------------- #


async def test_m2_remember_fact_blocked_by_governance():
    def no_secret(name, args):
        if name == "remember_fact" and "密码" in str(args.get("value", "")):
            return "禁止写入敏感信息"
        return None

    robot_id = "robot-1"
    governance = GovernancePolicy(constitution=[no_secret])
    model = make_model(
        responses=[
            _call("remember_fact", {"key": "k", "value": "密码123"}, "r1"),
            AIMessage("好的，不记了"),
        ]
    )
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        agent = build_robot_agent(model=model, store=store, governance=governance)
        out = await agent.ainvoke({"messages": [HumanMessage("记住密码")]})
        # 被治理拒绝：库里没有写入这条记忆。
        assert await store.aget(ns(robot_id, KIND_FACTS), "k") is None

    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert tool_msgs and "治理" in tool_msgs[0].content
    assert (
        governance.audit.denials
        and governance.audit.denials[0].action == "remember_fact"
    )


# --------------------------------------------------------------------------- #
# M4：同一持续循环在不同步数下只计一次（指纹不含步数）
# --------------------------------------------------------------------------- #


async def test_m4_loop_metric_not_double_counted_within_turn():
    async def inner(state):
        return {"llm_input_messages": list(state["messages"])}

    policy = MetacogPolicy(max_repeats=3, on_breach="warn")
    hook = make_monitor_hook(inner, policy)
    await hook({"messages": [HumanMessage("去"), _mv(1), _mv(2), _mv(3)]})  # repeats=3
    await hook(
        {"messages": [HumanMessage("去"), _mv(1), _mv(2), _mv(3), _mv(4)]}
    )  # repeats=4
    assert policy.metrics["loops"] == 1  # 同一回合的同一循环，不因步数变化翻倍


async def test_m4_distinct_turns_each_count(  # codex review：跨回合的不同循环须各计一次
):
    async def inner(state):
        return {"llm_input_messages": list(state["messages"])}

    policy = MetacogPolicy(max_repeats=3, on_breach="warn")
    hook = make_monitor_hook(inner, policy)
    # 回合一：循环 → 计 1。
    await hook({"messages": [HumanMessage("去A"), _mv(1), _mv(2), _mv(3)]})
    assert policy.metrics["loops"] == 1
    # 回合二（新增 human 锚点）：又一次循环 → 计 2，不被全局指纹坍缩。
    await hook(
        {
            "messages": [
                HumanMessage("去A"),
                _mv(1),
                _mv(2),
                _mv(3),
                HumanMessage("去B"),
                _mv(4),
                _mv(5),
                _mv(6),
            ]
        }
    )
    assert policy.metrics["loops"] == 2


# --------------------------------------------------------------------------- #
# M5：warn 告警不得排到身份锚点之前
# --------------------------------------------------------------------------- #


async def test_m5_warn_does_not_precede_identity_anchor():
    async def inner(state):
        return {
            "llm_input_messages": [
                SystemMessage("我是谁：小巡"),
                SystemMessage("已知的长期记忆：x"),
                HumanMessage("去"),
            ]
        }

    policy = MetacogPolicy(max_repeats=3, on_breach="warn")
    hook = make_monitor_hook(inner, policy)
    result = await hook({"messages": [HumanMessage("去"), _mv(1), _mv(2), _mv(3)]})
    msgs = result["llm_input_messages"]
    # 身份块仍在最前。
    assert "我是谁" in msgs[0].content
    contents = [getattr(m, "content", "") for m in msgs]
    warn_idx = next(i for i, c in enumerate(contents) if "元认知告警" in c)
    human_idx = next(i for i, m in enumerate(msgs) if m.type == "human")
    assert 0 < warn_idx < human_idx  # 告警在身份之后、历史之前


# --------------------------------------------------------------------------- #
# L1：同一收件箱内事件时间戳须同域——Observation.ts 契约是单调时钟，合成事件也用
#     单调时钟，否则跨事件时间比较失序（codex review 纠正：不可改 epoch）。
# --------------------------------------------------------------------------- #

_EPOCH_FLOOR = 1_000_000_000  # epoch 秒已超 1e9；单调时钟的进程运行秒数远小于此


def test_l1_driver_event_timestamps_are_monotonic_domain():
    from robot_agent.driver.policy import resume_event, user_message

    assert user_message("hi").ts < _EPOCH_FLOOR
    assert resume_event("t1", {"approved": True}).ts < _EPOCH_FLOOR


async def test_l1_prompt_idle_event_timestamp_is_monotonic_domain():
    from robot_agent.driver.policy import PromptIdlePolicy

    ev = await PromptIdlePolicy("巡逻").on_idle()
    assert ev is not None and ev.ts < _EPOCH_FLOOR


async def test_l1_goal_due_event_timestamp_is_monotonic_domain():
    from robot_agent.goals.models import Goal
    from robot_agent.goals.policy import GoalDrivenIdlePolicy
    from robot_agent.goals.store import GoalStore

    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        gs = GoalStore(store, "r1")
        await gs.add(Goal(id="g1", intent="巡逻"))
        ev = await GoalDrivenIdlePolicy(gs).on_idle()
    assert ev is not None and ev.ts < _EPOCH_FLOOR


# --------------------------------------------------------------------------- #
# L3：长期记忆检索失败时降级（不打断闭环）
# --------------------------------------------------------------------------- #


async def test_l3_memory_search_failure_degrades_gracefully():
    model = make_model(responses=[AIMessage("到了")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:

        async def boom(*a, **k):
            raise RuntimeError("store down")

        store.asearch = boom  # 模拟检索层故障
        agent = build_robot_agent(model=model, store=store)
        out = await agent.ainvoke({"messages": [HumanMessage("你好")]})
    assert out["messages"][-1].content == "到了"  # 闭环未被打断


# --------------------------------------------------------------------------- #
# L4：无 human 锚点时「本回合」为空（不误判跨回合动作为循环）
# --------------------------------------------------------------------------- #


def test_l4_detect_loop_no_human_returns_none():
    # 没有 human 输入界定本回合 → 不应把历史动作误判为本回合循环。
    msgs = [_mv(1), _mv(2), _mv(3), _mv(4)]
    assert detect_loop(msgs, max_repeats=3) is None


def test_l4_episode_current_turn_no_human_is_empty():
    from robot_agent.reflect.episode import _current_turn_messages

    assert _current_turn_messages([_mv(1), _mv(2)]) == []


# --------------------------------------------------------------------------- #
# L5：技能引用了不可用执行器时友好中止（不抛 KeyError）
# --------------------------------------------------------------------------- #


async def test_l5_skill_missing_effector_aborts_gracefully():
    from robot_agent.skills.models import Skill
    from robot_agent.skills.tools import build_skill_tools

    # 只装配 base，技能却要 grasp（路由到 arm）→ 应友好中止而非崩溃。
    effectors = {"base": build_effectors("mock")["base"]}
    skill = Skill(
        id="s1", name="pick", actions=[{"tool": "grasp", "args": {"target": "cup"}}]
    )
    (skill_tool,) = build_skill_tools([skill], effectors)
    res = await skill_tool.ainvoke({})
    assert "中止" in res and "arm" in res


async def test_l5_unavailable_effector_does_not_consume_governance_quota():
    """codex review：执行器不可用时不应先过 governance.check 而白白耗配额 / 留误导审计。"""
    from robot_agent.skills.models import Skill
    from robot_agent.skills.tools import build_skill_tools

    effectors = {"base": build_effectors("mock")["base"]}  # 无 arm
    governance = GovernancePolicy(rate_limit={"grasp": 1})
    skill = Skill(
        id="s1", name="pick", actions=[{"tool": "grasp", "args": {"target": "cup"}}]
    )
    (skill_tool,) = build_skill_tools([skill], effectors, governance=governance)
    res = await skill_tool.ainvoke({})
    assert "中止" in res and "arm" in res
    # 未消耗限频配额，也未记下「已放行」的误导审计。
    assert governance._counts.get("grasp", 0) == 0
    assert governance.audit.entries == []
