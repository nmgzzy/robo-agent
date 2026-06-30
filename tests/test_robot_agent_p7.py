"""P7 验收：记忆治理 / compaction（FR-10 / NFR-6 / AC-6）。

对应 docs/IMPLEMENTATION_PLAN.md §P7 与 docs/ROBOT_AGENT_DESIGN.md §8.4。覆盖：

- **去重**（规则）：相同值多条只留最新一条。
- **冲突消解（AC-6）**：注入矛盾事实 → LLM 检出 → 删其余 → 治理后检索不再自相矛盾。
- **衰减**：超龄未更新条目被归档删除。
- **compact_all / 钩子**：多 namespace 批量；driver 周期触发。

全部离线（Mock LLM + 内存 SQLite）。
"""

from __future__ import annotations

from datetime import timedelta

from langchain_core.messages import AIMessage
from langgraph.store.sqlite.aio import AsyncSqliteStore

from robot_agent import (
    CompactionReport,
    compact_all,
    compact_namespace,
    make_compaction_hook,
    make_model,
)
from robot_agent.memory import KIND_EPISODIC, KIND_FACTS, KIND_PREFS, ns


class _Turn:
    def __init__(self, index, interrupted=False):
        self.index = index
        self.interrupted = interrupted


# --------------------------------------------------------------------------- #
# 去重（规则，无 LLM）
# --------------------------------------------------------------------------- #


async def test_dedupe_exact_duplicates():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        for k in ("k1", "k2", "k3"):
            await store.aput(ns("r1", KIND_FACTS), k, {"value": "门是关的"})
        report = await compact_namespace(store, "r1", KIND_FACTS)  # 无 model

        assert isinstance(report, CompactionReport)
        assert report.deduped == 2 and report.kept == 1
        assert len(await store.asearch(ns("r1", KIND_FACTS))) == 1


# --------------------------------------------------------------------------- #
# AC-6：冲突消解
# --------------------------------------------------------------------------- #


async def test_ac6_conflict_resolution_and_retrieval_not_degraded():
    """注入矛盾事实 → LLM 检出删其余 → 治理后检索只剩一条（不再自相矛盾）。"""
    model = make_model(responses=[AIMessage("DROP 1")])  # 删一组矛盾里的一条
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(ns("r1", KIND_FACTS), "k1", {"value": "充电桩在A区"})
        await store.aput(ns("r1", KIND_FACTS), "k2", {"value": "充电桩在B区"})

        report = await compact_namespace(store, "r1", KIND_FACTS, model=model)

        assert report.conflicts_resolved == 1
        assert len(report.removed_keys) == 1
        remaining = await store.asearch(ns("r1", KIND_FACTS))
        # 检索不退化：仍能取到充电桩事实，但不再返回自相矛盾的两条。
        assert len(remaining) == 1


async def test_no_conflict_keeps_all():
    model = make_model(responses=[AIMessage("（无矛盾，无需删除）")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(ns("r1", KIND_FACTS), "k1", {"value": "充电桩在A区"})
        await store.aput(ns("r1", KIND_FACTS), "k2", {"value": "喜欢安静"})
        report = await compact_namespace(store, "r1", KIND_FACTS, model=model)
    assert report.conflicts_resolved == 0 and report.removed_keys == []


async def test_conflict_resolution_skipped_with_single_item():
    # 只有一条时不调用 LLM（无可矛盾对象）。
    model = make_model(responses=[])  # 若误调用会因响应耗尽抛错
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(ns("r1", KIND_FACTS), "k1", {"value": "唯一事实"})
        report = await compact_namespace(store, "r1", KIND_FACTS, model=model)
    assert report.conflicts_resolved == 0


async def test_episodic_excluded_from_conflict_resolution():
    """episodic 是历史证据，不走 LLM 冲突消解（codex review P2）。"""
    model = make_model(responses=[])  # 误调用 LLM 会因响应耗尽抛错
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(
            ns("r1", KIND_EPISODIC), "e1", {"intent": "去充电", "outcome": "成功"}
        )
        await store.aput(
            ns("r1", KIND_EPISODIC), "e2", {"intent": "去充电", "outcome": "失败"}
        )
        report = await compact_namespace(store, "r1", KIND_EPISODIC, model=model)
        remaining = await store.asearch(ns("r1", KIND_EPISODIC))
    # 未调用 LLM；两条历史都保留（不同结果不算矛盾）。
    assert report.conflicts_resolved == 0
    assert len(remaining) == 2


async def test_conflict_resolver_prompt_includes_keys_and_timestamps():
    """冲突消解的 prompt 须带 key 与 updated_at，模型才能判同主题/保留最新（codex review P2）。"""
    model = make_model(responses=[AIMessage("（不删除）")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(ns("r1", KIND_FACTS), "充电桩位置", {"value": "A区"})
        await store.aput(ns("r1", KIND_FACTS), "另一事实", {"value": "B区"})
        await compact_namespace(store, "r1", KIND_FACTS, model=model)
    prompt = model.received[0][0].content
    assert "充电桩位置" in prompt  # key 暴露给模型
    assert "更新于=" in prompt  # 时间戳暴露给模型


# --------------------------------------------------------------------------- #
# 衰减
# --------------------------------------------------------------------------- #


async def test_decay_removes_stale_entries():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(ns("r1", KIND_FACTS), "old", {"value": "过时信息"})
        item = (await store.asearch(ns("r1", KIND_FACTS)))[0]
        future = item.updated_at + timedelta(seconds=100)  # 模拟 100s 后

        report = await compact_namespace(
            store, "r1", KIND_FACTS, max_age_seconds=50, now=future
        )
        assert report.decayed == 1
        assert await store.asearch(ns("r1", KIND_FACTS)) == []


async def test_decay_keeps_fresh_entries():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(ns("r1", KIND_FACTS), "fresh", {"value": "新信息"})
        item = (await store.asearch(ns("r1", KIND_FACTS)))[0]
        soon = item.updated_at + timedelta(seconds=10)  # 仍在 TTL 内
        report = await compact_namespace(
            store, "r1", KIND_FACTS, max_age_seconds=50, now=soon
        )
    assert report.decayed == 0


# --------------------------------------------------------------------------- #
# compact_all（多 namespace）
# --------------------------------------------------------------------------- #


async def test_compact_all_covers_multiple_namespaces():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.aput(ns("r1", KIND_FACTS), "f1", {"value": "重复"})
        await store.aput(ns("r1", KIND_FACTS), "f2", {"value": "重复"})
        await store.aput(ns("r1", KIND_PREFS), "p1", {"value": "同好"})
        await store.aput(ns("r1", KIND_PREFS), "p2", {"value": "同好"})

        reports = await compact_all(store, "r1", kinds=(KIND_FACTS, KIND_PREFS))
        by_kind = {r.kind: r for r in reports}
    assert by_kind[KIND_FACTS].deduped == 1
    assert by_kind[KIND_PREFS].deduped == 1


# --------------------------------------------------------------------------- #
# driver 衔接：周期 compaction 钩子
# --------------------------------------------------------------------------- #


async def test_compaction_hook_triggers_periodically():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        model = make_model(responses=[AIMessage("DROP 1")])
        hook = make_compaction_hook(store, "r1", model=model, every=2)
        await store.aput(ns("r1", KIND_FACTS), "k1", {"value": "充电桩在A区"})
        await store.aput(ns("r1", KIND_FACTS), "k2", {"value": "充电桩在B区"})

        await hook(_Turn(1))  # completed=1，未触发
        assert len(await store.asearch(ns("r1", KIND_FACTS))) == 2

        await hook(_Turn(2))  # completed=2，触发 compaction
        assert len(await store.asearch(ns("r1", KIND_FACTS))) == 1


async def test_compaction_hook_skips_interrupted():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        model = make_model(responses=[AIMessage("DROP 1")])
        hook = make_compaction_hook(store, "r1", model=model, every=1)
        await store.aput(ns("r1", KIND_FACTS), "k1", {"value": "A区"})
        await store.aput(ns("r1", KIND_FACTS), "k2", {"value": "B区"})

        await hook(_Turn(1, interrupted=True))  # 暂停回合不计数、不触发
        assert len(await store.asearch(ns("r1", KIND_FACTS))) == 2
