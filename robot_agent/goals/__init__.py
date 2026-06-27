"""目标系统（设计 §8.2，实现计划 §P5）：跨回合目标的一等表示、持久化、仲裁、规划。

- `Goal` / 状态常量（models）：目标的结构化表示与生命周期。
- `GoalStore`（store）：Store `(robot_id,"goals")` 上的持久化 CRUD。
- `arbitrate` / `order_goals`（arbitrate）：多目标按优先级/deadline 仲裁。
- `plan_goal`（planning）：把意图分解为子步骤（支持重规划）。
- `GoalDrivenIdlePolicy`（policy）：driver 空闲时推进目标栈（衔接 P4）。
"""

from __future__ import annotations

from robot_agent.goals.arbitrate import arbitrate, order_goals
from robot_agent.goals.models import (
    ACTIONABLE_STATUSES,
    KIND_GOALS,
    STATUS_ABANDONED,
    STATUS_ACTIVE,
    STATUS_BLOCKED,
    STATUS_DONE,
    STATUS_PENDING,
    Goal,
    is_actionable,
)
from robot_agent.goals.planning import plan_goal
from robot_agent.goals.policy import GoalDrivenIdlePolicy
from robot_agent.goals.store import GoalStore

__all__ = [
    "ACTIONABLE_STATUSES",
    "KIND_GOALS",
    "STATUS_ABANDONED",
    "STATUS_ACTIVE",
    "STATUS_BLOCKED",
    "STATUS_DONE",
    "STATUS_PENDING",
    "Goal",
    "GoalDrivenIdlePolicy",
    "GoalStore",
    "arbitrate",
    "is_actionable",
    "order_goals",
    "plan_goal",
]
