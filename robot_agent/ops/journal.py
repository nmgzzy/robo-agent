"""决策日记 / 审计（设计 §8.9，实现计划 §P10）：可离线还原一次回合的决策链。

`stream_mode=debug` 面向开发期调试，不等于对常驻个体的生产期监护。本模块在 driver 的
`on_turn` 上结构化打点：记录每个回合的「意图 → 决策（工具调用）→ 结果」，可离线回放。

与 P6 episodic 的区别：episodic 是给蒸馏的「经历」（跳过暂停回合）；journal 是给运维的
「审计日记」——**所有**回合都记，包括被安全门控暂停的（标 `interrupted`），便于追责复盘。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot_agent.reflect.episode import episode_from_turn


@dataclass
class JournalEntry:
    """一条决策日记：一个回合的意图、决策链、结果与是否暂停。"""

    ts: float
    thread_id: str
    intent: str
    decisions: list[str]  # 本回合的工具调用（决策链）
    outcome: str
    interrupted: bool = False


class DecisionJournal:
    """决策日记（内存）：append 打点、按线程回放，供运行时自省与离线审查。"""

    def __init__(self) -> None:
        self.entries: list[JournalEntry] = []

    def append(self, entry: JournalEntry) -> None:
        self.entries.append(entry)

    def replay(self, thread_id: str) -> list[JournalEntry]:
        """还原某线程的决策链（按记录顺序）。"""
        return [e for e in self.entries if e.thread_id == thread_id]

    @property
    def latest(self) -> JournalEntry | None:
        return self.entries[-1] if self.entries else None


def journal_entry_from_turn(turn: Any) -> JournalEntry:
    """从 driver 的 `TurnResult` 提取一条决策日记。"""
    ep = episode_from_turn(turn)
    return JournalEntry(
        ts=ep.ts,
        thread_id=ep.thread_id,
        intent=ep.intent,
        decisions=ep.actions,
        outcome=ep.outcome,
        interrupted=getattr(turn, "interrupted", False),
    )


def make_journal_hook(journal: DecisionJournal):
    """构造 driver 的 `on_turn` 钩子：记录**每个**回合（含暂停回合）到决策日记。"""

    async def on_turn(turn: Any) -> None:
        journal.append(journal_entry_from_turn(turn))

    return on_turn


@dataclass
class _Introspection:
    """运行时自省快照（「现在在干嘛、为什么」）。"""

    turns: int
    pending_threads: list[str] = field(default_factory=list)
    latest: JournalEntry | None = None


def introspect(journal: DecisionJournal, *, driver: Any = None) -> _Introspection:
    """运行时自省：当前回合数、暂停中的线程、最近一次决策（设计 §8.9）。"""
    pending = sorted(driver.pending_threads) if driver is not None else []
    turns = driver.turns if driver is not None else len(journal.entries)
    return _Introspection(turns=turns, pending_threads=pending, latest=journal.latest)
