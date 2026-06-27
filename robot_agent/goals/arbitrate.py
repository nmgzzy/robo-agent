"""多目标仲裁（设计 §8.2，实现计划 §P5 任务③）：竞争目标里选「当前该做哪个」。

driver 空闲时调用：从目标栈里挑出最该推进的一个。排序优先级（依次）：

1. **优先级高**者先（`priority` 越大越优先）；
2. 同优先级，**deadline 早**者先（`None` 视为无限远，排最后）；
3. 再同，**登记早**者先（`created_ts`，稳定 tiebreak，避免抖动）。

只在可推进目标（pending/active）中选；blocked/done/abandoned 不参与。
被紧急事件打断后，因目标仍在 Store，下次仲裁会再次选回它——这就是「恢复到被打断目标」。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from robot_agent.goals.models import Goal, is_actionable


def _sort_key(goal: Goal) -> tuple[int, float, float]:
    deadline = goal.deadline if goal.deadline is not None else math.inf
    # priority 取负 → 高优先排前；deadline/created_ts 升序 → 早者排前。
    return (-goal.priority, deadline, goal.created_ts)


def order_goals(goals: Iterable[Goal]) -> list[Goal]:
    """把可推进目标按仲裁优先级排序（最该做的在前），过滤掉终态/受阻目标。"""
    return sorted((g for g in goals if is_actionable(g)), key=_sort_key)


def arbitrate(goals: Iterable[Goal]) -> Goal | None:
    """返回当前最该推进的目标；无可推进目标则 None。"""
    ordered = order_goals(goals)
    return ordered[0] if ordered else None
