"""记忆持久化测试：短期（checkpoint）崩溃恢复 + 长期（store）跨会话读写。

对应底座的核心可靠性承诺（docs/ROBOT_AGENT_DESIGN.md §5）。全部用本地 SQLite，
临时文件经 tmp_path 隔离，模拟「进程重启 = 新建数据库连接」。
"""

from __future__ import annotations

from typing import Annotated

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore
from typing_extensions import TypedDict


class _State(TypedDict):
    steps: Annotated[list[str], lambda a, b: (a or []) + (b or [])]


def _build_graph():
    g = StateGraph(_State)
    g.add_node("think", lambda s: {"steps": ["think"]})
    g.add_node("act", lambda s: {"steps": ["act"]})
    g.add_edge(START, "think")
    g.add_edge("think", "act")
    g.add_edge("act", END)
    return g


def test_sync_checkpoint_roundtrip(tmp_path):
    db = str(tmp_path / "cp_sync.db")
    cfg = {"configurable": {"thread_id": "t-sync"}}
    with SqliteSaver.from_conn_string(db) as saver:
        graph = _build_graph().compile(checkpointer=saver)
        out = graph.invoke({"steps": ["start"]}, cfg)
        assert out["steps"] == ["start", "think", "act"]
        assert graph.get_state(cfg).values["steps"] == ["start", "think", "act"]


async def test_async_checkpoint_persist_and_resume(tmp_path):
    """跑完一轮后用新连接（模拟重启）按 thread_id 恢复到中断前状态。"""
    db = str(tmp_path / "cp_async.db")
    cfg = {"configurable": {"thread_id": "episode-1"}}

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = _build_graph().compile(checkpointer=saver)
        out = await graph.ainvoke({"steps": ["start"]}, cfg)
        assert out["steps"] == ["start", "think", "act"]

    # 新连接 = 重启后
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = _build_graph().compile(checkpointer=saver)
        snap = await graph.aget_state(cfg)
        assert snap.values["steps"] == ["start", "think", "act"]


async def test_async_checkpoint_thread_isolation(tmp_path):
    db = str(tmp_path / "cp_iso.db")
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = _build_graph().compile(checkpointer=saver)
        await graph.ainvoke({"steps": ["A"]}, {"configurable": {"thread_id": "a"}})
        await graph.ainvoke({"steps": ["B"]}, {"configurable": {"thread_id": "b"}})
        sa = await graph.aget_state({"configurable": {"thread_id": "a"}})
        sb = await graph.aget_state({"configurable": {"thread_id": "b"}})
        assert sa.values["steps"][0] == "A"
        assert sb.values["steps"][0] == "B"


async def test_async_store_cross_session(tmp_path):
    """长期记忆：一个连接写入，新连接（跨会话）仍可读取。"""
    db = str(tmp_path / "store.db")
    ns = ("robot-1", "facts")

    async with AsyncSqliteStore.from_conn_string(db) as store:
        await store.aput(ns, "home", {"x": 0, "y": 0})
        await store.aput(ns, "dock", {"x": 3, "y": 2})

    async with AsyncSqliteStore.from_conn_string(db) as store:
        item = await store.aget(ns, "home")
        assert item is not None and item.value == {"x": 0, "y": 0}


async def test_async_store_search_namespace_and_delete():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(("robot", "facts"), "home", {"x": 0})
        await store.aput(("robot", "facts"), "dock", {"x": 3})
        await store.aput(("robot", "prefs"), "speed", {"v": 1})

        facts = await store.asearch(("robot", "facts"))
        assert sorted(i.key for i in facts) == ["dock", "home"]

        all_robot = await store.asearch(("robot",))
        assert len(all_robot) == 3

        namespaces = await store.alist_namespaces()
        assert ("robot", "facts") in namespaces
        assert ("robot", "prefs") in namespaces

        await store.adelete(("robot", "facts"), "home")
        assert await store.aget(("robot", "facts"), "home") is None


def test_sync_store_roundtrip(tmp_path):
    """同步长期记忆：put / get / search / delete 在本地文件上落盘可读。"""
    db = str(tmp_path / "store_sync.db")
    ns = ("robot-1", "episodic")

    with SqliteStore.from_conn_string(db) as store:
        store.put(ns, "ep1", {"task": "fetch cup", "ok": True})
        store.put(ns, "ep2", {"task": "charge", "ok": False})

        item = store.get(ns, "ep1")
        assert item is not None and item.value["task"] == "fetch cup"

        found = store.search(ns)
        assert sorted(i.key for i in found) == ["ep1", "ep2"]

        store.delete(ns, "ep2")
        assert store.get(ns, "ep2") is None
