"""P2 验收：可靠性与安全门控。

对应 docs/IMPLEMENTATION_PLAN.md §P2 与 docs/ROBOT_AGENT_DESIGN.md §7。覆盖：

- **AC-2**：危险动作 interrupt 暂停后「关库 → 重开 → 同 thread_id 续跑」，状态完整恢复。
- **AC-5**：危险动作（高速 / 抓取）被 interrupt 拦截，确认后放行、拒绝则跳过。
- **降级**：模拟 LLM 故障，重试耗尽后 ResilientChatModel 返回保守回复而非崩溃。
- **重试**：瞬态错误下重试后成功，闭环正常完成。
- **清理**：cleanup_threads 删除过期线程 checkpoint。

全部离线（Mock/Flaky LLM + Mock HAL + 内存/临时文件 SQLite）。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from robot_agent import (
    SafetyPolicy,
    build_effectors,
    build_robot_agent,
    cleanup_threads,
    make_model,
    make_resilient,
)
from robot_agent.reliability import DEFAULT_FALLBACK_TEXT
from robot_agent.safety import confirm_or_block, danger_reason, reject_reason


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class FlakyChatModel(BaseChatModel):
    """先抛 `fail_times` 次瞬态错误，再按脚本回放 AIMessage 的假模型（测试重试/降级）。"""

    responses: list[AIMessage]
    fail_times: int = 0
    error: type[BaseException] = ConnectionError
    idx: int = 0
    failed: int = 0

    @property
    def _llm_type(self) -> str:
        return "flaky-chat"

    def bind_tools(self, tools, **kwargs: Any) -> "FlakyChatModel":  # noqa: ANN001
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        if self.failed < self.fail_times:
            object.__setattr__(self, "failed", self.failed + 1)
            raise self.error(f"模拟瞬态故障 #{self.failed}")
        msg = self.responses[self.idx]
        object.__setattr__(self, "idx", self.idx + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> ChatResult:  # noqa: ANN001
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# --------------------------------------------------------------------------- #
# 危险判定（纯函数，无需 graph）
# --------------------------------------------------------------------------- #


def test_danger_reason_flags_high_speed_and_grasp():
    p = SafetyPolicy()
    assert danger_reason("set_velocity", {"vx": 2.0, "wz": 0.0}, p) is not None
    assert danger_reason("set_velocity", {"vx": 0.0, "wz": 3.0}, p) is not None
    assert danger_reason("grasp", {"target": "cup"}, p) is not None
    # 低速在阈值内、move_to 不门控。
    assert danger_reason("set_velocity", {"vx": 0.1, "wz": 0.1}, p) is None
    assert danger_reason("move_to", {"x": 1.0, "y": 2.0}, p) is None


def test_non_finite_velocity_is_hard_rejected():
    # NaN/Inf 是硬拒绝（不可批准），不走 danger_reason 的可确认路径（codex review）。
    p = SafetyPolicy()
    assert reject_reason("set_velocity", {"vx": float("nan"), "wz": 0.0}) is not None
    assert reject_reason("set_velocity", {"vx": 0.0, "wz": float("inf")}) is not None
    assert danger_reason("set_velocity", {"vx": float("nan"), "wz": 0.0}, p) is None
    # confirm_or_block 直接拒绝且不触发 interrupt（无 checkpointer 也安全）。
    approved, note = confirm_or_block(
        "set_velocity", {"vx": float("nan"), "wz": 0.0}, p
    )
    assert approved is False and "非有限速度" in note


def test_confirm_or_block_passes_safe_action_without_interrupt():
    # 安全动作不应触发 interrupt（否则无 checkpointer 会抛错）。
    approved, note = confirm_or_block("move_to", {"x": 1.0, "y": 2.0}, SafetyPolicy())
    assert approved is True and note == ""


# --------------------------------------------------------------------------- #
# AC-5：危险动作 interrupt 门控（确认放行 / 拒绝跳过）
# --------------------------------------------------------------------------- #


async def test_ac5_grasp_gated_then_approved():
    effectors = build_effectors("mock")
    model = make_model(
        responses=[
            _tool_call("grasp", {"obj": "cup"}, "g1"),
            AIMessage(content="已抓取"),
        ]
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(
            model=model, effectors=effectors, checkpointer=saver, safety=SafetyPolicy()
        )
        cfg = {"configurable": {"thread_id": "t-approve"}}
        out = await agent.ainvoke({"messages": [HumanMessage("抓杯子")]}, cfg)

        # 抓取前被门控暂停，执行器尚未动作。
        assert "__interrupt__" in out
        assert effectors["arm"].log == []
        interrupt = out["__interrupt__"][0]
        assert interrupt.value["action"] == "grasp"

        # 人工确认放行 → 续跑，执行器下发一次（无重复）。
        out = await agent.ainvoke(Command(resume={"approved": True}), cfg)

    assert effectors["arm"].log == [{"action": "grasp", "target": "cup"}]
    assert out["messages"][-1].content == "已抓取"


async def test_ac5_high_speed_gated_then_rejected():
    effectors = build_effectors("mock")
    model = make_model(
        responses=[
            _tool_call("set_velocity", {"vx": 2.0, "wz": 0.0}, "v1"),
            AIMessage(content="好的，我停下"),
        ]
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(
            model=model, effectors=effectors, checkpointer=saver, safety=SafetyPolicy()
        )
        cfg = {"configurable": {"thread_id": "t-reject"}}
        out = await agent.ainvoke({"messages": [HumanMessage("全速前进")]}, cfg)
        assert "__interrupt__" in out

        # 规则/人工拒绝 → 动作被跳过，执行器无任何指令。
        out = await agent.ainvoke(
            Command(resume={"approved": False, "reason": "走廊有人"}), cfg
        )

    assert effectors["base"].log == []
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert "被安全门控拒绝" in tool_msgs[-1].content
    assert "走廊有人" in tool_msgs[-1].content


async def test_ac5_malformed_resume_fails_closed():
    """畸形/非布尔恢复值（如字符串 "deny"）必须 fail-closed，不得放行危险动作。"""
    effectors = build_effectors("mock")
    model = make_model(
        responses=[_tool_call("grasp", {"obj": "cup"}, "g1"), AIMessage(content="好")]
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(
            model=model, effectors=effectors, checkpointer=saver, safety=SafetyPolicy()
        )
        cfg = {"configurable": {"thread_id": "t-malformed"}}
        await agent.ainvoke({"messages": [HumanMessage("抓杯子")]}, cfg)
        out = await agent.ainvoke(Command(resume="deny"), cfg)

    # bool("deny") 为 True，但安全门控只认显式 True → 动作被拒绝。
    assert effectors["arm"].log == []
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert "被安全门控拒绝" in tool_msgs[-1].content


async def test_safe_action_not_gated_even_with_policy():
    effectors = build_effectors("mock")
    model = make_model(
        responses=[
            _tool_call("set_velocity", {"vx": 0.1, "wz": 0.1}, "s1"),
            AIMessage(content="缓行中"),
        ]
    )
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        agent = build_robot_agent(
            model=model, effectors=effectors, checkpointer=saver, safety=SafetyPolicy()
        )
        out = await agent.ainvoke(
            {"messages": [HumanMessage("慢慢走")]},
            {"configurable": {"thread_id": "t-safe"}},
        )

    # 低速不触发门控，直接执行，无 interrupt。
    assert "__interrupt__" not in out
    assert effectors["base"].log == [{"action": "set_velocity", "vx": 0.1, "wz": 0.1}]


# --------------------------------------------------------------------------- #
# AC-2：危险动作暂停后「关库 → 重开 → 同 thread_id 续跑」
# --------------------------------------------------------------------------- #


async def test_ac2_crash_recovery_resume_after_reopen(tmp_path):
    """interrupt 暂停 → 关闭 saver（模拟杀进程）→ 新 saver 重开同库 → 续跑成功。"""
    db = str(tmp_path / "agent.db")
    cfg = {"configurable": {"thread_id": "t-crash"}}
    # 执行器代表物理硬件：跨「进程重启」仍是同一台机器，故复用同一对象。
    effectors = build_effectors("mock")

    # 进程 A：产出 grasp → 被门控暂停，状态落盘。
    model_a = make_model(
        responses=[_tool_call("grasp", {"obj": "cup"}, "c1"), AIMessage(content="完成")]
    )
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        agent_a = build_robot_agent(
            model=model_a,
            effectors=effectors,
            checkpointer=saver,
            safety=SafetyPolicy(),
        )
        out = await agent_a.ainvoke({"messages": [HumanMessage("抓杯子")]}, cfg)
        assert "__interrupt__" in out
        assert effectors["arm"].log == []

    # 进程 B：全新 saver/agent/model 重开同一个 db，恢复并放行。
    model_b = make_model(responses=[AIMessage(content="完成")])
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        agent_b = build_robot_agent(
            model=model_b,
            effectors=effectors,
            checkpointer=saver,
            safety=SafetyPolicy(),
        )
        # 状态确实从磁盘恢复：能取到被中断的快照。
        snap = await agent_b.aget_state(cfg)
        assert snap.next, "重开后应有待续跑的中断任务"

        out = await agent_b.ainvoke(Command(resume={"approved": True}), cfg)

    assert effectors["arm"].log == [{"action": "grasp", "target": "cup"}]
    assert out["messages"][-1].content == "完成"


# --------------------------------------------------------------------------- #
# 重试 / 降级（ResilientChatModel）
# --------------------------------------------------------------------------- #


async def test_resilient_retries_then_succeeds():
    flaky = FlakyChatModel(responses=[AIMessage(content="ok")], fail_times=2)
    model = make_resilient(flaky, max_attempts=3, initial_interval=0.0)
    msg = await model.ainvoke([HumanMessage("hi")])
    assert msg.content == "ok"
    assert model.calls[-1] == {"attempts": 3, "degraded": False}


async def test_resilient_degrades_when_exhausted():
    flaky = FlakyChatModel(responses=[AIMessage(content="ok")], fail_times=99)
    model = make_resilient(flaky, max_attempts=2, initial_interval=0.0)
    msg = await model.ainvoke([HumanMessage("hi")])
    assert msg.content == DEFAULT_FALLBACK_TEXT
    assert model.calls[-1]["degraded"] is True


async def test_resilient_does_not_retry_unlisted_error():
    # ValueError 不在 retry_on 中：应原样抛出，不被吞成降级。
    flaky = FlakyChatModel(
        responses=[AIMessage(content="ok")], fail_times=1, error=ValueError
    )
    model = make_resilient(flaky, max_attempts=3, initial_interval=0.0)
    with pytest.raises(ValueError):
        await model.ainvoke([HumanMessage("hi")])


async def test_resilient_retries_provider_transient_by_name():
    """默认谓词按类名识别 provider 瞬态异常（不 import anthropic）→ 重试而非崩溃。"""

    class RateLimitError(
        Exception
    ):  # 拟 anthropic.RateLimitError（不是 TimeoutError 子类）
        pass

    flaky = FlakyChatModel(
        responses=[AIMessage(content="ok")], fail_times=2, error=RateLimitError
    )
    model = make_resilient(flaky, max_attempts=3, initial_interval=0.0)
    msg = await model.ainvoke([HumanMessage("hi")])
    assert msg.content == "ok"
    assert model.calls[-1] == {"attempts": 3, "degraded": False}


def test_resilient_sync_generate_is_rejected():
    # 闭环 async-only：同步 _generate 显式拒绝，而非留事件循环隐患（review 收尾）。
    flaky = FlakyChatModel(responses=[AIMessage(content="ok")])
    model = make_resilient(flaky)
    with pytest.raises(NotImplementedError, match="仅支持异步"):
        model.invoke([HumanMessage("hi")])


async def test_ac_degrade_in_full_loop():
    """决策大脑彻底不可用 → 闭环返回保守降级回复（无工具调用），不崩溃。"""
    flaky = FlakyChatModel(responses=[AIMessage(content="never")], fail_times=99)
    model = make_resilient(flaky, max_attempts=2, initial_interval=0.0)
    effectors = build_effectors("mock")
    agent = build_robot_agent(model=model, effectors=effectors)
    out = await agent.ainvoke({"messages": [HumanMessage("过来")]})

    assert out["messages"][-1].content == DEFAULT_FALLBACK_TEXT
    # 降级回复不带工具调用 → 没有任何执行器被触发（停在原地）。
    assert effectors["base"].log == [] and effectors["arm"].log == []


# --------------------------------------------------------------------------- #
# 过期 checkpoint 清理
# --------------------------------------------------------------------------- #


async def test_cleanup_threads_removes_checkpoints(tmp_path):
    db = str(tmp_path / "cp.db")
    cfg = {"configurable": {"thread_id": "t-old"}}
    model = make_model(responses=[AIMessage(content="ok")])
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        agent = build_robot_agent(model=model, checkpointer=saver)
        await agent.ainvoke({"messages": [HumanMessage("hi")]}, cfg)
        assert await saver.aget_tuple(cfg) is not None

        deleted = await cleanup_threads(saver, ["t-old"])
        assert deleted == 1
        assert await saver.aget_tuple(cfg) is None
