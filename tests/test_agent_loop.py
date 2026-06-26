"""Agent 主循环测试：思考 → 决策 → 工具调用（控制机器人）→ 记忆。

对应 docs/ROBOT_AGENT_DESIGN.md §3-§4。两条路径都覆盖：
- 手写 StateGraph（think + ToolNode + tools_condition），确定性、不依赖 LLM；
- create_react_agent + 脚本化假模型，覆盖推荐的高层入口（瘦身版仍可用）。
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import (
    InjectedStore,
    ToolNode,
    create_react_agent,
    tools_condition,
)
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt


class _ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def test_tools_condition_routing():
    with_call = {"messages": [_tool_call("grasp", {"obj": "cup"}, "1")]}
    without_call = {"messages": [AIMessage(content="done")]}
    assert tools_condition(with_call) == "tools"
    assert tools_condition(without_call) == END


def test_manual_think_tools_loop():
    """手写 think→tools 循环：第一次发起工具调用，拿到结果后给最终答案。"""
    executed: list[str] = []

    @tool
    def grasp(obj: str) -> str:
        """抓取一个物体（占位的机器人控制动作）。"""
        executed.append(obj)
        return f"grasped {obj}"

    def think(state: _ChatState):
        has_result = any(getattr(m, "type", None) == "tool" for m in state["messages"])
        if has_result:
            return {"messages": [AIMessage(content="任务完成")]}
        return {"messages": [_tool_call("grasp", {"obj": "cup"}, "t1")]}

    g = StateGraph(_ChatState)
    g.add_node("think", think)
    g.add_node("tools", ToolNode([grasp]))
    g.add_edge(START, "think")
    g.add_conditional_edges("think", tools_condition)
    g.add_edge("tools", "think")
    graph = g.compile()

    out = graph.invoke({"messages": [HumanMessage("把杯子拿起来")]})
    assert executed == ["cup"]
    assert out["messages"][-1].content == "任务完成"
    assert [m.type for m in out["messages"]] == ["human", "ai", "tool", "ai"]


async def test_manual_loop_resume_after_restart(tmp_path):
    """带 checkpointer 的主循环跑完后，新连接（重启）仍能取回完整对话历史。"""
    @tool
    def ping() -> str:
        """占位动作。"""
        return "pong"

    def think(state: _ChatState):
        has_result = any(getattr(m, "type", None) == "tool" for m in state["messages"])
        if has_result:
            return {"messages": [AIMessage(content="ok")]}
        return {"messages": [_tool_call("ping", {}, "p1")]}

    def build():
        g = StateGraph(_ChatState)
        g.add_node("think", think)
        g.add_node("tools", ToolNode([ping]))
        g.add_edge(START, "think")
        g.add_conditional_edges("think", tools_condition)
        g.add_edge("tools", "think")
        return g

    db = str(tmp_path / "loop.db")
    cfg = {"configurable": {"thread_id": "loop-1"}}

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = build().compile(checkpointer=saver)
        await graph.ainvoke({"messages": [HumanMessage("go")]}, cfg)

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = build().compile(checkpointer=saver)
        snap = await graph.aget_state(cfg)
        assert [m.type for m in snap.values["messages"]] == [
            "human",
            "ai",
            "tool",
            "ai",
        ]


def test_create_react_agent_tool_roundtrip(make_model):
    """create_react_agent 用脚本化假模型完成一次工具调用往返。"""
    calls: list[tuple[float, float]] = []

    @tool
    def move_to(x: float, y: float) -> str:
        """移动机器人到坐标 (x, y)。"""
        calls.append((x, y))
        return f"arrived ({x},{y})"

    model = make_model(
        [
            _tool_call("move_to", {"x": 1.0, "y": 2.0}, "a1"),
            AIMessage(content="已到达目标点"),
        ]
    )
    agent = create_react_agent(model, tools=[move_to])
    out = agent.invoke({"messages": [HumanMessage("去 (1,2)")]})

    assert calls == [(1.0, 2.0)]
    assert out["messages"][-1].content == "已到达目标点"
    assert [m.type for m in out["messages"]] == ["human", "ai", "tool", "ai"]


async def test_create_react_agent_with_memory(tmp_path, make_model):
    """create_react_agent 挂 checkpointer + store 可正常运行并落盘。"""
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    @tool
    def noop() -> str:
        """占位动作。"""
        return "ok"

    model = make_model([AIMessage(content="直接回答")])
    cfg = {"configurable": {"thread_id": "react-1"}}
    db = str(tmp_path / "react.db")

    async with AsyncSqliteSaver.from_conn_string(db) as saver, \
            AsyncSqliteStore.from_conn_string(":memory:") as store:
        agent = create_react_agent(
            model, tools=[noop], checkpointer=saver, store=store
        )
        out = await agent.ainvoke({"messages": [HumanMessage("hi")]}, cfg)
        assert out["messages"][-1].content == "直接回答"
        snap = await agent.aget_state(cfg)
        assert snap.values["messages"][-1].content == "直接回答"


class _GateState(TypedDict):
    val: str


def test_interrupt_and_resume():
    """安全门控：危险动作前 interrupt 暂停，确认后 Command(resume=...) 继续。

    对应 docs/ROBOT_AGENT_DESIGN.md §6 可靠性（人工/规则确认 + 可恢复执行）。
    """
    def gate(state: _GateState):
        decision = interrupt({"ask": "确认执行危险动作?"})
        return {"val": f"resumed:{decision}"}

    g = StateGraph(_GateState)
    g.add_node("gate", gate)
    g.add_edge(START, "gate")
    g.add_edge("gate", END)

    with SqliteSaver.from_conn_string(":memory:") as saver:
        graph = g.compile(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "gate-1"}}

        first = graph.invoke({"val": "start"}, cfg)
        assert "__interrupt__" in first  # 已在门控处暂停

        resumed = graph.invoke(Command(resume="yes"), cfg)
        assert resumed["val"] == "resumed:yes"


def test_tool_reads_injected_store(make_model):
    """工具通过 InjectedStore 读取长期记忆（设计 §4.2 / §5）。"""

    @tool
    def recall(key: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
        """从长期记忆读取一个事实。"""
        item = store.get(("facts",), key)
        return str(item.value) if item else "none"

    store = InMemoryStore()
    store.put(("facts",), "home", {"x": 0, "y": 0})

    model = make_model(
        [
            _tool_call("recall", {"key": "home"}, "r1"),
            AIMessage(content="已读取记忆"),
        ]
    )
    agent = create_react_agent(model, tools=[recall], store=store)
    out = agent.invoke({"messages": [HumanMessage("家在哪")]})

    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert tool_msgs and "0" in tool_msgs[0].content
    assert out["messages"][-1].content == "已读取记忆"
