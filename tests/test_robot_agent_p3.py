"""P3 验收：身份 / 自我模型（FR-14）。

对应 docs/IMPLEMENTATION_PLAN.md §P3 与 docs/ROBOT_AGENT_DESIGN.md §8.8。覆盖：

- 身份读写接口：set_identity / get_identity / ensure_default_identity（幂等）。
- format_identity 渲染稳定身份文本块。
- **稳定注入**：配置身份后，每个新 thread 的 LLM 输入都含身份 system 块（决策锚点）。
- **可观察变化**：移除身份则不注入；改写身份则注入内容随之改变（边界/语气锚点变化）。
- **注入顺序**：身份块置于长期记忆块之前（稳定锚点在最前）。

全部离线（Mock LLM + 内存/临时文件 SQLite）。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.store.sqlite.aio import AsyncSqliteStore
from robot_agent import (
    DEFAULT_IDENTITY,
    build_robot_agent,
    ensure_default_identity,
    get_identity,
    make_model,
    set_identity,
)
from robot_agent.identity import format_identity, identity_ns, load_identity_text
from robot_agent.memory import KIND_PREFS, ns


def _sys_texts(received_input) -> list[str]:
    return [m.content for m in received_input if isinstance(m, SystemMessage)]


# --------------------------------------------------------------------------- #
# 读写接口
# --------------------------------------------------------------------------- #


async def test_set_get_identity_roundtrip():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        assert await get_identity(store, "r1") is None
        await set_identity(store, "r1", {"name": "阿福", "persona": "管家机器人"})
        got = await get_identity(store, "r1")
    assert got == {"name": "阿福", "persona": "管家机器人"}


async def test_ensure_default_identity_is_idempotent():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        first = await ensure_default_identity(store, "r1")
        assert first == DEFAULT_IDENTITY
        # 已存在则不覆盖：先改名，再 ensure，应保留改后的值。
        await set_identity(store, "r1", {"name": "改过的名字"})
        second = await ensure_default_identity(store, "r1")
    assert second == {"name": "改过的名字"}


def test_format_identity_contains_key_fields():
    text = format_identity(DEFAULT_IDENTITY)
    assert "小巡" in text
    assert "价值观" in text and "安全第一" in text
    assert "擅长" in text and "不擅长" in text


async def test_load_identity_text_none_when_absent():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        assert await load_identity_text(store, "r1") is None


async def test_identity_read_is_isolated_from_store():
    """改动 get_identity 的返回值（含嵌套 list）不得回灌污染已存身份（codex review）。"""
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await set_identity(store, "r1", {"name": "x", "values": ["a"]})
        got = await get_identity(store, "r1")
        got["values"].append("篡改")  # 改动嵌套结构
        again = await get_identity(store, "r1")
    assert again["values"] == ["a"]


async def test_ensure_default_identity_does_not_mutate_global():
    """ensure_default_identity 返回值被改动不得污染进程级 DEFAULT_IDENTITY。"""
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        seeded = await ensure_default_identity(store, "r1")
        seeded["values"].append("篡改")
    assert "篡改" not in DEFAULT_IDENTITY["values"]


def test_identity_namespace():
    assert identity_ns("robot-7") == ("robot-7", "identity")


# --------------------------------------------------------------------------- #
# FR-14：稳定注入 + 可观察变化
# --------------------------------------------------------------------------- #


async def test_identity_injected_into_llm_input():
    """配置身份后，新 thread 的 LLM 输入含身份 system 块（稳定决策锚点）。"""
    robot_id = "robot-1"
    model = make_model(responses=[AIMessage(content="收到")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await ensure_default_identity(store, robot_id)
        agent = build_robot_agent(model=model, store=store, robot_id=robot_id)
        await agent.ainvoke(
            {"messages": [HumanMessage("你是谁")]},
            {"configurable": {"thread_id": "t-id"}},
        )

    sys_texts = _sys_texts(model.received[0])
    assert any("我是谁" in t and "小巡" in t for t in sys_texts), sys_texts


async def test_no_identity_no_injection():
    """未配置身份则不注入身份块（移除身份 → 行为锚点消失，可观察）。"""
    model = make_model(responses=[AIMessage(content="嗯")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        agent = build_robot_agent(model=model, store=store)
        await agent.ainvoke({"messages": [HumanMessage("在吗")]})

    sys_texts = _sys_texts(model.received[0])
    assert not any("我是谁" in t for t in sys_texts)


async def test_identity_change_changes_injection():
    """改写身份 → 注入内容随之变化（决策语气/边界锚点可观察地改变）。"""
    robot_id = "robot-1"
    model = make_model(responses=[AIMessage(content="好")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await set_identity(
            store, robot_id, {"name": "守卫者", "persona": "强调安全的巡逻机器人"}
        )
        agent = build_robot_agent(model=model, store=store, robot_id=robot_id)
        await agent.ainvoke(
            {"messages": [HumanMessage("巡逻")]},
            {"configurable": {"thread_id": "t-id2"}},
        )

    sys_texts = _sys_texts(model.received[0])
    assert any("守卫者" in t for t in sys_texts), sys_texts
    assert not any("小巡" in t for t in sys_texts)


async def test_identity_block_precedes_memory_block():
    """注入顺序：身份块（稳定锚点）在长期记忆块之前。"""
    robot_id = "robot-1"
    model = make_model(responses=[AIMessage(content="好的")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await ensure_default_identity(store, robot_id)
        await store.aput(ns(robot_id, KIND_PREFS), "lang", {"value": "讲中文"})
        agent = build_robot_agent(model=model, store=store, robot_id=robot_id)
        await agent.ainvoke(
            {"messages": [HumanMessage("你好")]},
            {"configurable": {"thread_id": "t-order"}},
        )

    received = model.received[0]
    sys_msgs = [m for m in received if isinstance(m, SystemMessage)]
    # 两个 system 块：身份在前，记忆在后。
    assert len(sys_msgs) >= 2
    assert "我是谁" in sys_msgs[0].content
    assert "长期记忆" in sys_msgs[1].content and "讲中文" in sys_msgs[1].content
