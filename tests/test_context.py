"""中短期记忆：高水位滚动摘要、工具消息完整性与 checkpoint 持久化。"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from robot_agent import (
    ContextPolicy,
    RobotState,
    load_context_policy_from_env,
    make_model,
)
from robot_agent.context import SUMMARY_MARKER, prepare_context, trim_llm_input_messages
from robot_agent.identity import ensure_default_identity
from robot_agent.memory import KIND_FACTS, make_inject_memory, ns
from robot_agent.ops.health import collect_health


def _policy(**overrides) -> ContextPolicy:
    values = {
        "high_watermark_tokens": 150,
        "recent_window_tokens": 50,
        "max_summary_tokens": 80,
        "hard_limit_tokens": 240,
        "summary_batch_tokens": 500,
    }
    values.update(overrides)
    return ContextPolicy(**values)


def _summary(text: str = "用户早先要求巡视；尚未完成。") -> AIMessage:
    return AIMessage(content=f"{SUMMARY_MARKER}\n{text}")


def test_context_policy_rejects_invalid_watermarks():
    with pytest.raises(ValueError, match="recent_window_tokens"):
        ContextPolicy(
            high_watermark_tokens=100,
            recent_window_tokens=100,
            hard_limit_tokens=200,
        )
    with pytest.raises(ValueError, match="high_watermark_tokens"):
        ContextPolicy(
            high_watermark_tokens=300,
            recent_window_tokens=100,
            max_summary_tokens=50,
            hard_limit_tokens=200,
        )
    with pytest.raises(ValueError, match="立刻再次触发"):
        ContextPolicy(
            high_watermark_tokens=300,
            recent_window_tokens=200,
            max_summary_tokens=100,
            hard_limit_tokens=400,
        )
    with pytest.raises(ValueError, match="正整数"):
        ContextPolicy(max_summary_tokens=0)


def test_context_policy_loads_all_limits_from_environment(monkeypatch):
    values = {
        "CONTEXT_HIGH_WATERMARK_TOKENS": "6000",
        "CONTEXT_RECENT_WINDOW_TOKENS": "3000",
        "CONTEXT_MAX_SUMMARY_TOKENS": "1000",
        "CONTEXT_HARD_LIMIT_TOKENS": "8000",
        "CONTEXT_SUMMARY_BATCH_TOKENS": "2500",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    assert load_context_policy_from_env() == ContextPolicy(
        high_watermark_tokens=6000,
        recent_window_tokens=3000,
        max_summary_tokens=1000,
        hard_limit_tokens=8000,
        summary_batch_tokens=2500,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CONTEXT_HIGH_WATERMARK_TOKENS", "not-a-number"),
        ("CONTEXT_HARD_LIMIT_TOKENS", "0"),
    ],
)
def test_context_policy_rejects_invalid_environment_value(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_context_policy_from_env()


def test_context_policy_rejects_invalid_environment_combination(monkeypatch):
    monkeypatch.setenv("CONTEXT_HIGH_WATERMARK_TOKENS", "100")
    monkeypatch.setenv("CONTEXT_RECENT_WINDOW_TOKENS", "60")
    monkeypatch.setenv("CONTEXT_MAX_SUMMARY_TOKENS", "40")
    monkeypatch.setenv("CONTEXT_HARD_LIMIT_TOKENS", "200")

    with pytest.raises(ValueError, match="环境变量组合无效"):
        load_context_policy_from_env()


async def test_below_watermark_reuses_persisted_summary_without_model_call():
    summary_model = make_model(responses=[])
    result = await prepare_context(
        {
            "messages": [HumanMessage("继续")],
            "context_summary": "此前已经找到充电桩。",
        },
        summary_model=summary_model,
        policy=_policy(),
    )

    assert result.state_update == {}
    assert summary_model.received == []
    assert "此前已经找到充电桩" in result.messages[0].content
    assert result.messages[-1].content == "继续"


async def test_summary_is_injected_after_stable_leading_system_messages():
    result = await prepare_context(
        {
            "messages": [SystemMessage("稳定规则"), HumanMessage("继续")],
            "context_summary": "此前摘要",
        },
        summary_model=make_model(responses=[]),
        policy=_policy(),
    )

    assert result.messages[0].content == "稳定规则"
    assert "此前摘要" in result.messages[1].content
    assert result.messages[2].content == "继续"


async def test_high_watermark_archives_old_turn_and_keeps_current_tool_pair():
    summary_model = make_model(responses=[_summary()])
    messages = [
        HumanMessage("很早的巡视要求：" + "甲" * 550),
        AIMessage(content="早期答复"),
        HumanMessage("现在检查门口"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_world_state", "args": {}, "id": "call-current"}],
        ),
        ToolMessage(content='{"door":"closed"}', tool_call_id="call-current"),
    ]

    result = await prepare_context(
        {"messages": messages}, summary_model=summary_model, policy=_policy()
    )

    update = result.state_update
    assert isinstance(update["messages"][0], RemoveMessage)
    persisted = update["messages"][1:]
    assert [message.type for message in persisted[-3:]] == ["human", "ai", "tool"]
    assert persisted[-2].tool_calls[0]["id"] == persisted[-1].tool_call_id
    assert update["context_summary"] == "用户早先要求巡视；尚未完成。"
    assert update["context_compaction_count"] == 1
    assert update["context_archived_messages"] == 2
    assert "会话摘要" in result.messages[0].content
    assert (
        summary_model.received
        and "很早的巡视要求" in summary_model.received[0][0].content
    )


async def test_rolling_summary_uses_previous_summary_and_updates_counters():
    summary_model = make_model(responses=[_summary("已合并的新摘要。")])
    state = {
        "messages": [
            HumanMessage("旧回合：" + "乙" * 550),
            AIMessage("完成了一半"),
            HumanMessage("新回合继续"),
        ],
        "context_summary": "第一版摘要",
        "context_compaction_count": 2,
        "context_archived_messages": 7,
    }

    result = await prepare_context(state, summary_model=summary_model, policy=_policy())

    prompt = summary_model.received[0][0].content
    assert "第一版摘要" in prompt
    assert result.state_update["context_summary"] == "已合并的新摘要。"
    assert result.state_update["context_compaction_count"] == 3
    assert result.state_update["context_archived_messages"] == 8


async def test_bad_summary_falls_back_without_overwriting_checkpoint_state():
    summary_model = make_model(responses=[AIMessage("没有协议标记")])
    original = [
        HumanMessage("旧信息" + "丙" * 550),
        AIMessage("旧答复"),
        HumanMessage("当前问题"),
    ]
    state = {"messages": original}

    result = await prepare_context(state, summary_model=summary_model, policy=_policy())

    assert result.state_update == {"context_compaction_failures": 1}
    assert "messages" not in result.state_update
    assert "context_summary" not in result.state_update
    assert state["messages"] == original
    assert result.messages[-1].content == "当前问题"


async def test_memory_hook_propagates_compaction_update_and_omits_image_payload():
    secret_payload = "data:image/jpeg;base64," + "A" * 8000
    summary_model = make_model(responses=[_summary("旧图像回合已归档。")])
    hook = make_inject_memory(
        "robot-1", summary_model=summary_model, context_policy=_policy()
    )
    state = {
        "messages": [
            HumanMessage(
                content=[
                    {"type": "text", "text": "旧图像说明" + "戊" * 550},
                    {"type": "image_url", "image_url": {"url": secret_payload}},
                ]
            ),
            AIMessage("已看过"),
            HumanMessage("继续当前回合"),
        ]
    }

    result = await hook(state)

    assert result["context_summary"] == "旧图像回合已归档。"
    assert isinstance(result["messages"][0], RemoveMessage)
    assert "旧图像回合已归档" in result["llm_input_messages"][0].content
    assert secret_payload not in summary_model.received[0][0].content
    assert "非文本载荷未送入" in summary_model.received[0][0].content


async def test_context_summary_and_compacted_messages_persist_in_checkpoint():
    summary_model = make_model(responses=[_summary("旧回合要求记录温度。")])
    config = {"configurable": {"thread_id": "context-persist"}}
    old_text = "旧回合记录：" + "丁" * 550

    async def compact(state: dict):
        result = await prepare_context(
            state, summary_model=summary_model, policy=_policy()
        )
        return result.state_update

    workflow = StateGraph(RobotState)
    workflow.add_node("compact", compact)
    workflow.add_edge(START, "compact")
    workflow.add_edge("compact", END)

    graph = workflow.compile(checkpointer=InMemorySaver())
    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(old_text),
                AIMessage("温度是 25 度"),
                HumanMessage("继续"),
            ]
        },
        config,
    )
    snapshot = await graph.aget_state(config)

    assert snapshot.values["context_summary"] == "旧回合要求记录温度。"
    assert snapshot.values["context_compaction_count"] == 1
    assert all(
        old_text not in str(message.content) for message in snapshot.values["messages"]
    )
    assert snapshot.values["messages"][-1].content == "继续"


def test_trim_llm_input_preserves_identity_and_memory_prefix():
    prefix = [SystemMessage("IDENTITY-" + "锚" * 120)]
    body = [
        HumanMessage("应被裁掉" + "旧" * 600),
        HumanMessage("保留当前"),
    ]
    messages = [*prefix, *body]
    trimmed = trim_llm_input_messages(
        messages, preserve_prefix=1, hard_limit_tokens=120
    )

    assert trimmed[0] is prefix[0]
    assert count_tokens_approximately(trimmed) <= 120
    assert trimmed[-1].content == "保留当前"


def test_trim_llm_input_preserves_all_leading_system_blocks():
    identity = SystemMessage("IDENTITY")
    summary = SystemMessage("CONTEXT SUMMARY")
    trimmed = trim_llm_input_messages(
        [
            identity,
            summary,
            HumanMessage("旧消息" + "旧" * 500),
            HumanMessage("当前消息"),
        ],
        preserve_prefix=1,
        hard_limit_tokens=100,
    )

    assert trimmed[:2] == [identity, summary]
    assert trimmed[-1].content == "当前消息"
    assert count_tokens_approximately(trimmed) <= 100


def test_trim_llm_input_omits_oversized_optional_system_block(caplog):
    caplog.set_level(logging.WARNING, logger="robot_agent.context")
    identity = SystemMessage("IDENTITY")
    oversized_memory = SystemMessage("MEMORY-" + "大" * 1000)
    current = HumanMessage("继续")

    trimmed = trim_llm_input_messages(
        [identity, oversized_memory, current],
        preserve_prefix=1,
        hard_limit_tokens=80,
    )

    assert trimmed == [identity, current]
    assert count_tokens_approximately(trimmed) <= 80
    assert any("前导 system 块" in record.message for record in caplog.records)


def test_trim_llm_input_rejects_required_prefix_over_hard_limit():
    with pytest.raises(ValueError, match="必须保留的 system 前缀"):
        trim_llm_input_messages(
            [SystemMessage("IDENTITY-" + "大" * 1000)],
            preserve_prefix=1,
            hard_limit_tokens=20,
        )


async def test_prepare_context_reserved_tokens_triggers_compaction():
    summary_model = make_model(responses=[_summary("因预留预算触发压缩。")])
    filler = "甲" * 280
    messages = [
        HumanMessage("旧回合：" + filler),
        AIMessage("收到"),
        HumanMessage("当前"),
    ]
    policy = _policy(high_watermark_tokens=150, recent_window_tokens=50)

    below = await prepare_context(
        {"messages": messages},
        summary_model=summary_model,
        policy=policy,
        reserved_tokens=0,
    )
    assert below.state_update == {}

    above = await prepare_context(
        {"messages": messages},
        summary_model=make_model(responses=[_summary("因预留预算触发压缩。")]),
        policy=policy,
        reserved_tokens=80,
    )
    assert above.state_update.get("context_compaction_count") == 1

    # 压缩后的摘要 + 最近窗口 + 预留前缀应回落到水位以下，不能下一次 hook 立即再压缩。
    compacted_state = {
        "messages": above.state_update["messages"][1:],
        "context_summary": above.state_update["context_summary"],
    }
    second_model = make_model(responses=[])
    second = await prepare_context(
        compacted_state,
        summary_model=second_model,
        policy=policy,
        reserved_tokens=80,
    )
    assert second.state_update == {}
    assert second_model.received == []


async def test_prepare_context_rejects_negative_reserved_tokens():
    with pytest.raises(ValueError, match="reserved_tokens"):
        await prepare_context(
            {"messages": [HumanMessage("继续")]},
            summary_model=make_model(responses=[]),
            policy=_policy(),
            reserved_tokens=-1,
        )


async def test_oversized_wrapped_summary_rejected():
    body = "超长摘要正文" * 60
    summary_model = make_model(responses=[AIMessage(f"{SUMMARY_MARKER}\n{body}")])
    state = {
        "messages": [
            HumanMessage("旧信息" + "丙" * 550),
            AIMessage("旧答复"),
            HumanMessage("当前问题"),
        ]
    }

    result = await prepare_context(
        state, summary_model=summary_model, policy=_policy(max_summary_tokens=40)
    )

    assert result.state_update == {"context_compaction_failures": 1}


async def test_summary_failure_logs_warning(caplog):
    caplog.set_level(logging.WARNING, logger="robot_agent.context")
    summary_model = make_model(responses=[AIMessage("没有协议标记")])
    state = {
        "messages": [
            HumanMessage("旧信息" + "丙" * 550),
            AIMessage("旧答复"),
            HumanMessage("当前问题"),
        ]
    }

    await prepare_context(state, summary_model=summary_model, policy=_policy())

    assert any("上下文滚动摘要失败" in record.message for record in caplog.records)


async def test_memory_hook_trims_after_memory_injection(monkeypatch):
    store = InMemoryStore()
    await store.aput(
        ns("robot-1", KIND_FACTS),
        "big",
        {"value": "长期记忆" * 200},
    )
    monkeypatch.setattr("robot_agent.memory.get_store", lambda: store)

    hook = make_inject_memory(
        "robot-1",
        summary_model=make_model(responses=[]),
        context_policy=_policy(
            hard_limit_tokens=600,
            high_watermark_tokens=500,
            recent_window_tokens=50,
            max_summary_tokens=80,
        ),
        inject_identity=False,
    )
    result = await hook({"messages": [HumanMessage("继续" + "乙" * 500)]})

    assert count_tokens_approximately(result["llm_input_messages"]) <= 600
    assert any("长期记忆" in str(m.content) for m in result["llm_input_messages"][:1])


async def test_oversized_memory_cannot_displace_state_system_or_summary(monkeypatch):
    store = InMemoryStore()
    await store.aput(
        ns("robot-1", KIND_FACTS),
        "oversized",
        {"value": "不应挤掉稳定规则" * 1000},
    )
    monkeypatch.setattr("robot_agent.memory.get_store", lambda: store)
    hook = make_inject_memory(
        "robot-1",
        summary_model=make_model(responses=[]),
        context_policy=_policy(),
        inject_identity=False,
    )

    result = await hook(
        {
            "messages": [SystemMessage("稳定系统规则"), HumanMessage("继续")],
            "context_summary": "此前任务摘要",
        }
    )

    llm_input = result["llm_input_messages"]
    assert llm_input[0].content == "稳定系统规则"
    assert "此前任务摘要" in llm_input[1].content
    assert all("不应挤掉稳定规则" not in str(message.content) for message in llm_input)
    assert llm_input[-1].content == "继续"
    assert count_tokens_approximately(llm_input) <= _policy().hard_limit_tokens


async def test_memory_hook_system_priority_order(monkeypatch):
    store = InMemoryStore()
    await ensure_default_identity(store, "robot-1")
    await store.aput(ns("robot-1", KIND_FACTS), "dock", {"value": "东侧"})
    monkeypatch.setattr("robot_agent.memory.get_store", lambda: store)
    hook = make_inject_memory(
        "robot-1",
        summary_model=make_model(responses=[]),
        context_policy=_policy(hard_limit_tokens=500),
    )

    result = await hook(
        {
            "messages": [SystemMessage("应用稳定规则"), HumanMessage("继续")],
            "context_summary": "此前任务摘要",
        }
    )

    system_contents = [
        str(message.content)
        for message in result["llm_input_messages"]
        if isinstance(message, SystemMessage)
    ]
    assert "我是谁" in system_contents[0]
    assert system_contents[1] == "应用稳定规则"
    assert "此前任务摘要" in system_contents[2]
    assert "长期记忆" in system_contents[3] and "东侧" in system_contents[3]


def test_collect_health_includes_context_compaction_metrics():
    report = collect_health(
        agent_state={
            "context_compaction_count": 3,
            "context_compaction_failures": 1,
            "context_archived_messages": 42,
        }
    )
    assert report.context_compactions == 3
    assert report.context_compaction_failures == 1
    assert report.context_archived_messages == 42
    assert report.to_dict()["context_compactions"] == 3


def test_collect_health_accepts_graph_state_snapshot_shape():
    snapshot = SimpleNamespace(
        values={
            "context_compaction_count": 2,
            "context_compaction_failures": 1,
            "context_archived_messages": 9,
        }
    )

    report = collect_health(agent_state=snapshot)

    assert report.context_compactions == 2
    assert report.context_compaction_failures == 1
    assert report.context_archived_messages == 9
