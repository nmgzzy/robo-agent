"""P9 验收：安全 / 对齐策略层（FR-12）。

对应 docs/IMPLEMENTATION_PLAN.md §P9 与 docs/ROBOT_AGENT_DESIGN.md §8.6。覆盖：

- **宪章硬约束**：违反规则的动作被拦截并记审计（验收）。
- **工具权限范围**：越权工具调用被拒（验收）。
- **危险动作限幅 / 限频**：超硬上限 / 超频被拒（验收）。
- **审计**：允许/拒绝结构化记录，可离线审查。
- **闭环**：被拒动作不下发执行器，拒绝回执回灌模型，回合继续。

全部离线（Mock LLM + Mock HAL）。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from robot_agent import (
    AuditLog,
    GovernancePolicy,
    SafetyPolicy,
    ToolPermission,
    build_effectors,
    build_robot_agent,
    make_model,
)
from robot_agent.governance.policy import AmplitudeLimit


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# --------------------------------------------------------------------------- #
# check 纯逻辑：权限 / 宪章 / 限幅 / 限频
# --------------------------------------------------------------------------- #


def test_permission_whitelist_and_blacklist():
    p = GovernancePolicy(
        permission=ToolPermission(allowed=frozenset({"move_to", "speak"}))
    )
    assert p.check("move_to", {"x": 1, "y": 2})[0] is True
    ok, reason = p.check("grasp", {"target": "cup"})  # 不在白名单
    assert ok is False and "允许范围" in reason

    p2 = GovernancePolicy(permission=ToolPermission(denied=frozenset({"grasp"})))
    assert p2.check("grasp", {"target": "cup"})[0] is False


def test_constitution_rule_blocks_and_audits():
    def no_fragile(name, args):
        if name == "grasp" and args.get("target") == "鸡蛋":
            return "禁止抓取易碎品"
        return None

    p = GovernancePolicy(constitution=[no_fragile])
    ok, reason = p.check("grasp", {"target": "鸡蛋"})
    assert ok is False and "违反宪章" in reason and "易碎" in reason
    assert p.check("grasp", {"target": "杯子"})[0] is True
    # 审计记录了一次拒绝。
    assert len(p.audit.denials) == 1
    assert p.audit.denials[0].action == "grasp"


def test_amplitude_limit_rejects_over_speed():
    p = GovernancePolicy(amplitude=AmplitudeLimit(max_vx=1.5, max_wz=3.0))
    assert p.check("set_velocity", {"vx": 1.0, "wz": 1.0})[0] is True
    ok, reason = p.check("set_velocity", {"vx": 5.0, "wz": 0.0})
    assert ok is False and "硬上限" in reason


def test_rate_limit_rejects_after_quota():
    p = GovernancePolicy(rate_limit={"speak": 2})
    assert p.check("speak", {"text": "1"})[0] is True
    assert p.check("speak", {"text": "2"})[0] is True
    ok, reason = p.check("speak", {"text": "3"})  # 第三次超限
    assert ok is False and "限频" in reason


def test_audit_records_allow_and_deny():
    p = GovernancePolicy(permission=ToolPermission(denied=frozenset({"grasp"})))
    p.check("move_to", {"x": 0, "y": 0})
    p.check("grasp", {"target": "x"})
    assert len(p.audit.entries) == 2
    assert [e.allowed for e in p.audit.entries] == [True, False]
    assert len(p.audit.denials) == 1


# --------------------------------------------------------------------------- #
# 闭环：被治理拒绝的动作不下发执行器
# --------------------------------------------------------------------------- #


async def test_governed_action_blocked_in_full_loop():
    """模型调超速 set_velocity → 治理拒绝 → 执行器无指令 + 审计 + 回合继续。"""
    effectors = build_effectors("mock")
    audit = AuditLog()
    governance = GovernancePolicy(
        amplitude=AmplitudeLimit(max_vx=1.0, max_wz=2.0), audit=audit
    )
    model = make_model(
        responses=[
            _tool_call("set_velocity", {"vx": 9.0, "wz": 0.0}, "v1"),
            AIMessage("好的，我减速"),
        ]
    )
    agent = build_robot_agent(model=model, effectors=effectors, governance=governance)
    out = await agent.ainvoke({"messages": [HumanMessage("全速冲")]})

    # 超速被拒：执行器没有下发任何指令。
    assert effectors["base"].log == []
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert "被治理策略拒绝" in tool_msgs[0].content
    assert out["messages"][-1].content == "好的，我减速"
    # 审计留痕。
    assert len(audit.denials) == 1 and audit.denials[0].action == "set_velocity"


async def test_permitted_action_executes_and_audited():
    effectors = build_effectors("mock")
    audit = AuditLog()
    governance = GovernancePolicy(
        permission=ToolPermission(allowed=frozenset({"move_to"})), audit=audit
    )
    model = make_model(
        responses=[_tool_call("move_to", {"x": 1.0, "y": 2.0}, "m1"), AIMessage("到了")]
    )
    agent = build_robot_agent(model=model, effectors=effectors, governance=governance)
    out = await agent.ainvoke({"messages": [HumanMessage("过去")]})

    assert effectors["base"].log == [{"action": "move_to", "x": 1.0, "y": 2.0}]
    assert out["messages"][-1].content == "到了"
    # 审计记录一次允许。
    allowed = [e for e in audit.entries if e.allowed]
    assert len(allowed) == 1 and allowed[0].action == "move_to"


async def test_safety_override_is_audited():
    """governance 放行后 safety 门控的人工拒绝必须记入审计（codex review）。"""
    effectors = build_effectors("mock")
    audit = AuditLog()
    governance = GovernancePolicy(audit=audit)  # 无硬约束，仅放行 + 审计
    model = make_model(
        responses=[_tool_call("grasp", {"obj": "cup"}, "g1"), AIMessage("好的")]
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(
            model=model,
            effectors=effectors,
            checkpointer=saver,
            governance=governance,
            safety=SafetyPolicy(),
        )
        cfg = {"configurable": {"thread_id": "t-override"}}
        await agent.ainvoke({"messages": [HumanMessage("抓杯子")]}, cfg)
        # 人工拒绝。
        await agent.ainvoke(Command(resume={"approved": False, "reason": "易碎"}), cfg)

    assert effectors["arm"].log == []  # 被人工拒绝，未执行
    # 审计含：治理放行（allowed）+ 安全门控人工拒绝（denied）。
    actions = [(e.action, e.allowed) for e in audit.entries]
    assert ("grasp", True) in actions  # 治理放行留痕
    override = [e for e in audit.denials if e.action == "grasp"]
    assert override and "易碎" in override[0].reason  # 人工覆盖结果被审计
