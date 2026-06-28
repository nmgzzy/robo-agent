"""运维 / 可观测（设计 §8.9，实现计划 §P10）：常驻个体的生产期监护。

- `DecisionJournal` / `make_journal_hook` / `introspect`（journal）：决策日记 + 运行时自省。
- `HealthReport` / `collect_health`（health）：健康度指标聚合导出（含会话压缩指标），
  供远程巡检读取。
"""

from __future__ import annotations

from robot_agent.ops.health import HealthReport, collect_health
from robot_agent.ops.journal import (
    DecisionJournal,
    JournalEntry,
    introspect,
    journal_entry_from_turn,
    make_journal_hook,
)

__all__ = [
    "DecisionJournal",
    "HealthReport",
    "JournalEntry",
    "collect_health",
    "introspect",
    "journal_entry_from_turn",
    "make_journal_hook",
]
