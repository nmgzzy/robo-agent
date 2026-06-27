"""P8 验收：元认知 / 自我监控（FR-11）。

对应 docs/IMPLEMENTATION_PLAN.md §P8 与 docs/ROBOT_AGENT_DESIGN.md §8.5。覆盖：

- **循环检测**：本回合连续相同工具调用达阈值被识别。
- **死循环中断/上报**：escalate 模式下循环触发 interrupt 上报（验收）。
- **预算耗尽收敛/上报**：步数预算越界触发上报（验收）。
- **warn 模式**：注入元认知告警促 LLM 收敛，不中断。
- **指标导出**：MetacogPolicy.metrics 累计循环/预算越界次数。

全部离线（Mock LLM + Mock HAL + 内存 SQLite）。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from robot_agent import (
    MetacogPolicy,
    build_effectors,
    build_robot_agent,
    detect_loop,
    make_model,
)
from robot_agent.metacog import steps_used


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


# --------------------------------------------------------------------------- #
# 纯函数检测
# --------------------------------------------------------------------------- #


def test_detect_loop_flags_repeated_calls():
    msgs = [HumanMessage("去那边"), _mv(1), _mv(2), _mv(3)]
    sig = detect_loop(msgs, max_repeats=3)
    assert sig is not None and sig["repeats"] == 3


def test_detect_loop_ignores_distinct_calls():
    msgs = [HumanMessage("做事"), _mv(1, 1, 1), _mv(2, 2, 2), _mv(3, 3, 3)]
    assert detect_loop(msgs, max_repeats=3) is None


def test_detect_loop_resets_per_turn():
    # 新一回合的 human 之后重新计数：跨回合相同动作不算循环。
    msgs = [HumanMessage("一"), _mv(1), _mv(2), HumanMessage("二"), _mv(3)]
    assert detect_loop(msgs, max_repeats=2) is None


def test_steps_used_counts_current_turn_tool_calls():
    msgs = [HumanMessage("一"), _mv(1), HumanMessage("二"), _mv(2), _mv(3)]
    assert steps_used(msgs) == 2  # 只数最后一个 human 之后的两步


# --------------------------------------------------------------------------- #
# 死循环 → escalate（interrupt 上报）
# --------------------------------------------------------------------------- #


async def test_loop_escalates_via_interrupt():
    effectors = build_effectors("mock")
    policy = MetacogPolicy(max_repeats=3)
    model = make_model(responses=[_mv(1), _mv(2), _mv(3)])  # 连续相同 move_to
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(
            model=model, effectors=effectors, checkpointer=saver, metacog=policy
        )
        out = await agent.ainvoke(
            {"messages": [HumanMessage("去那边")]},
            {"configurable": {"thread_id": "t-loop"}},
        )

    assert "__interrupt__" in out
    assert out["__interrupt__"][0].value["type"] == "metacog_escalation"
    assert out["__interrupt__"][0].value["loop"]["repeats"] == 3
    assert policy.metrics["loops"] == 1 and policy.metrics["escalations"] == 1
    # 第 4 次调用前被拦下：只下发了 3 次移动。
    assert len(effectors["base"].log) == 3


async def test_escalation_metrics_not_double_counted_on_resume():
    """resume 会令 pre_model_hook 重放；同一越界事件不得被重复计入 metrics（codex review）。"""
    effectors = build_effectors("mock")
    policy = MetacogPolicy(max_repeats=3)
    # 第 4 次 pre-hook escalate；resume 放行后模型给终态收敛。
    model = make_model(responses=[_mv(1), _mv(2), _mv(3), AIMessage("好，我停下")])
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(
            model=model, effectors=effectors, checkpointer=saver, metacog=policy
        )
        cfg = {"configurable": {"thread_id": "t-resume"}}
        out = await agent.ainvoke({"messages": [HumanMessage("去那边")]}, cfg)
        assert "__interrupt__" in out and policy.metrics["loops"] == 1

        # 人工放行续跑：hook 重放但不重复计数。
        out = await agent.ainvoke(Command(resume={"action": "continue"}), cfg)

    assert out["messages"][-1].content == "好，我停下"
    assert policy.metrics["loops"] == 1  # 仍为 1，未因重放翻倍
    assert policy.metrics["escalations"] == 1


# --------------------------------------------------------------------------- #
# 预算耗尽 → escalate
# --------------------------------------------------------------------------- #


async def test_step_budget_breach_escalates():
    effectors = build_effectors("mock")
    policy = MetacogPolicy(max_repeats=None, max_steps=2)  # 仅预算，关循环检测
    model = make_model(responses=[_mv(1, 1, 1), _mv(2, 2, 2), _mv(3, 3, 3)])
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(
            model=model, effectors=effectors, checkpointer=saver, metacog=policy
        )
        out = await agent.ainvoke(
            {"messages": [HumanMessage("做很多步")]},
            {"configurable": {"thread_id": "t-budget"}},
        )

    assert "__interrupt__" in out
    assert "步数预算" in out["__interrupt__"][0].value["reason"]
    assert policy.metrics["budget_breaches"] == 1
    assert len(effectors["base"].log) == 2  # 第 3 步前被拦下


# --------------------------------------------------------------------------- #
# warn 模式：注入告警，不中断
# --------------------------------------------------------------------------- #


async def test_warn_mode_injects_warning_and_continues():
    effectors = build_effectors("mock")
    policy = MetacogPolicy(max_repeats=3, on_breach="warn")
    # 循环 3 次后，第 4 次 pre-hook 注入告警 → 模型收敛给出终态。
    model = make_model(responses=[_mv(1), _mv(2), _mv(3), AIMessage("好，我停下")])
    agent = build_robot_agent(model=model, effectors=effectors, metacog=policy)
    out = await agent.ainvoke({"messages": [HumanMessage("去那边")]})

    assert out["messages"][-1].content == "好，我停下"  # 未中断，正常收敛
    # 第 4 次模型输入里含元认知告警。
    last_input = model.received[-1]
    assert any("元认知告警" in getattr(m, "content", "") for m in last_input)
    assert policy.metrics["loops"] == 1 and policy.metrics["escalations"] == 0


# --------------------------------------------------------------------------- #
# 正常无循环：不触发
# --------------------------------------------------------------------------- #


async def test_no_breach_runs_normally():
    effectors = build_effectors("mock")
    # on_breach="warn"：无 checkpointer 也可装配（escalate 需 checkpointer，见 graph 装配校验）。
    policy = MetacogPolicy(max_repeats=3, max_steps=10, on_breach="warn")
    model = make_model(responses=[_mv(1, 1, 1), AIMessage("到了")])
    out = await agent_run(effectors, policy, model)
    assert out["messages"][-1].content == "到了"
    assert policy.metrics == {"loops": 0, "budget_breaches": 0, "escalations": 0}


async def agent_run(effectors, policy, model):
    agent = build_robot_agent(model=model, effectors=effectors, metacog=policy)
    return await agent.ainvoke({"messages": [HumanMessage("去一个地方")]})
