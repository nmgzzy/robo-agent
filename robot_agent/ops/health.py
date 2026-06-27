"""健康度指标（设计 §8.9，实现计划 §P10）：可被外部读取的常驻个体监护指标。

把分散在各能力域的运行时信号聚合成一份可导出的 `HealthReport`：回合数、元认知上报/循环/
预算越界（P8）、治理拒绝数（P9）、暂停中的线程数（P2/P4）。供远程巡检与健康检查读取。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class HealthReport:
    """常驻个体健康度快照。"""

    turns: int = 0
    pending_threads: int = 0
    escalations: int = 0
    loops: int = 0
    budget_breaches: int = 0
    denials: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def collect_health(
    *,
    driver: Any = None,
    metacog: Any = None,
    governance: Any = None,
    journal: Any = None,
) -> HealthReport:
    """聚合各能力域的运行时信号为一份健康度报告（设计 §8.9）。

    各来源可选传入：driver（回合/暂停线程）、metacog（循环/预算/上报）、governance（拒绝）、
    journal（无 driver 时以日记条数兜底回合数）。
    """
    m = getattr(metacog, "metrics", {}) if metacog is not None else {}
    if driver is not None:
        turns = driver.turns
        pending = len(driver.pending_threads)
    else:
        turns = len(journal.entries) if journal is not None else 0
        pending = 0
    denials = len(governance.audit.denials) if governance is not None else 0
    return HealthReport(
        turns=turns,
        pending_threads=pending,
        escalations=m.get("escalations", 0),
        loops=m.get("loops", 0),
        budget_breaches=m.get("budget_breaches", 0),
        denials=denials,
    )
