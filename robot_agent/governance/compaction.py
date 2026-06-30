"""记忆治理 / compaction（设计 §8.4，实现计划 §P7）：长跑不被脏记忆拖垮。

语义记忆（Store）跑久了会变得又大又脏又自相矛盾，检索质量崩塌——这是这类系统最常见
的死法。本模块对一个 namespace 做后台 compaction：

1. **去重**（规则）：值完全相同的多条只留 `updated_at` 最新的一条。
2. **冲突消解**（LLM，可选）：检出语义矛盾/冗余，按「保留最可信/最新」删其余。
3. **衰减**（可选）：超过 `max_age_seconds` 未更新的条目归档删除。

由 driver 周期调度（`make_compaction_hook` 或外部 cron）。治理只删冗余/旧/矛盾项，
保留的条目仍按原 namespace 检索——治理后检索 top-k 质量不退化（AC-6）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.store.base import BaseStore

from robot_agent import prompts
from robot_agent.memory import KIND_EPISODIC, KIND_FACTS, KIND_PREFS, _unwrap_value, ns

# 默认治理的 namespace 种类（身份/目标有各自语义，不在此处粗暴去重）。
DEFAULT_KINDS: tuple[str, ...] = (KIND_FACTS, KIND_PREFS, KIND_EPISODIC)
# 仅对**语义记忆**做 LLM 冲突消解；episodic 是历史证据（同动作/不同结果都属正常记录，
# 且要留给 reflect_and_distill 消费），不可当矛盾事实删除。
CONFLICT_KINDS: frozenset[str] = frozenset({KIND_FACTS, KIND_PREFS})

_DROP_RE = re.compile(r"DROP\s+(\d+)", re.IGNORECASE)
_PAGE = 100


@dataclass
class CompactionReport:
    """一次 compaction 的结果（按 namespace）。"""

    kind: str
    seen: int = 0
    deduped: int = 0
    conflicts_resolved: int = 0
    decayed: int = 0
    removed_keys: list[str] = field(default_factory=list)

    @property
    def kept(self) -> int:
        return self.seen - len(self.removed_keys)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b["text"]
            for b in content
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return str(content)


def _value_signature(value: Any) -> str:
    """把（还原后的）记忆值规整为可比较的签名，用于精确去重。"""
    return json.dumps(
        _unwrap_value(value), sort_keys=True, ensure_ascii=False, default=str
    )


async def _read_all(store: BaseStore, namespace: tuple[str, ...]) -> list[Any]:
    items: list[Any] = []
    offset = 0
    while True:
        batch = await store.asearch(namespace, limit=_PAGE, offset=offset)
        items.extend(batch)
        if len(batch) < _PAGE:
            break
        offset += _PAGE
    return items


def _dedupe(items: Sequence[Any], removed: set[str]) -> int:
    """精确去重：相同值的多条只保留 updated_at 最新者，旧者计入 removed。"""
    latest_by_sig: dict[str, Any] = {}
    count = 0
    for it in sorted(items, key=lambda i: i.updated_at):  # 旧 → 新
        sig = _value_signature(it.value)
        prev = latest_by_sig.get(sig)
        if prev is not None:
            removed.add(prev.key)  # 之前那条更旧，删它
            count += 1
        latest_by_sig[sig] = it
    return count


async def _resolve_conflicts(
    model: BaseChatModel, items: Sequence[Any], removed: set[str]
) -> int:
    """LLM 检出矛盾/冗余并删其余。返回消解（删除）的条目数。"""
    survivors = [it for it in items if it.key not in removed]
    if len(survivors) < 2:
        return 0
    numbered = dict(enumerate(survivors, start=1))
    # 带上 key 与 updated_at：让模型能判断「同主题」并执行「保留最新」策略。
    listing = "\n".join(
        f"{n}: [键={it.key} 更新于={it.updated_at.isoformat()}] {_unwrap_value(it.value)}"
        for n, it in numbered.items()
    )
    msg = await model.ainvoke(
        [HumanMessage(prompts.render("conflict", memories=listing))]
    )
    count = 0
    for m in _DROP_RE.finditer(_content_to_text(msg.content)):
        it = numbered.get(int(m.group(1)))
        if it is not None and it.key not in removed:
            removed.add(it.key)
            count += 1
    return count


async def compact_namespace(
    store: BaseStore,
    robot_id: str,
    kind: str,
    *,
    model: BaseChatModel | None = None,
    max_age_seconds: float | None = None,
    now: datetime | None = None,
) -> CompactionReport:
    """对一个 namespace 做 compaction（去重 → 冲突消解 → 衰减）。

    - `model`：给定则做 LLM 冲突消解；否则只做规则去重/衰减（无算力开销）。
    - `max_age_seconds` + `now`：给定则衰减——`updated_at` 早于 `now - max_age` 的条目删除。
    """
    items = await _read_all(store, ns(robot_id, kind))
    removed: set[str] = set()

    deduped = _dedupe(items, removed)

    conflicts = 0
    # 只对语义记忆（facts/prefs）做 LLM 冲突消解；episodic 等历史记录不参与。
    if model is not None and kind in CONFLICT_KINDS:
        conflicts = await _resolve_conflicts(model, items, removed)

    decayed = 0
    if max_age_seconds is not None and now is not None:
        for it in items:
            if it.key in removed:
                continue
            if (now - it.updated_at).total_seconds() > max_age_seconds:
                removed.add(it.key)
                decayed += 1

    for key in removed:
        await store.adelete(ns(robot_id, kind), key)

    return CompactionReport(
        kind=kind,
        seen=len(items),
        deduped=deduped,
        conflicts_resolved=conflicts,
        decayed=decayed,
        removed_keys=sorted(removed),
    )


async def compact_all(
    store: BaseStore,
    robot_id: str,
    *,
    model: BaseChatModel | None = None,
    kinds: Sequence[str] = DEFAULT_KINDS,
    max_age_seconds: float | None = None,
    now: datetime | None = None,
) -> list[CompactionReport]:
    """对多个 namespace 依次 compaction（默认 facts/prefs/episodic）。"""
    reports = []
    for kind in kinds:
        reports.append(
            await compact_namespace(
                store,
                robot_id,
                kind,
                model=model,
                max_age_seconds=max_age_seconds,
                now=now,
            )
        )
    return reports


def make_compaction_hook(
    store: BaseStore,
    robot_id: str,
    *,
    model: BaseChatModel | None = None,
    every: int = 10,
    kinds: Sequence[str] = DEFAULT_KINDS,
    max_age_seconds: float | None = None,
) -> Callable[[Any], Awaitable[None]]:
    """构造 driver 的 `on_turn` 钩子：每 `every` 个**已完成**回合做一次后台 compaction。

    与 `make_reflect_hook` 同构：跳过安全暂停回合、按完成回合计数。compaction 较重，
    `every` 宜大（默认 10）。需与 reflect 钩子同挂时，把两个 `on_turn` 组合调用即可。
    """
    completed = 0

    async def on_turn(turn: Any) -> None:
        nonlocal completed
        if getattr(turn, "interrupted", False):
            return
        completed += 1
        if every > 0 and completed % every == 0:
            now = datetime.now(timezone.utc) if max_age_seconds is not None else None
            await compact_all(
                store,
                robot_id,
                model=model,
                kinds=kinds,
                max_age_seconds=max_age_seconds,
                now=now,
            )

    return on_turn
