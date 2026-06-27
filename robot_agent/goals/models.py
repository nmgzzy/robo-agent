"""目标模型（设计 §8.2，实现计划 §P5）：跨回合的意图/目标一等表示。

`messages` 是回合内的工作记忆，扛不住跨回合的目标。Goal 把「想做什么」结构化、
持久化，使个体能在多个回合、被打断后仍记得并推进长期目标。

`status` 生命周期：
- `pending`：待办（已登记，未开始）。
- `active`：进行中（当前正在推进）。
- `blocked`：受阻（缺前置/资源，暂不可推进）。
- `done` / `abandoned`：终态（完成 / 放弃），不再参与仲裁。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# 长期目标的 Store namespace 种类。
KIND_GOALS = "goals"

STATUS_PENDING = "pending"
STATUS_ACTIVE = "active"
STATUS_BLOCKED = "blocked"
STATUS_DONE = "done"
STATUS_ABANDONED = "abandoned"

ALL_STATUSES: tuple[str, ...] = (
    STATUS_PENDING,
    STATUS_ACTIVE,
    STATUS_BLOCKED,
    STATUS_DONE,
    STATUS_ABANDONED,
)
# 可被仲裁选为「当前目标」的状态（待办或进行中）。
ACTIONABLE_STATUSES: frozenset[str] = frozenset({STATUS_PENDING, STATUS_ACTIVE})
# 终态：不再推进，也不参与仲裁。
TERMINAL_STATUSES: frozenset[str] = frozenset({STATUS_DONE, STATUS_ABANDONED})


class Goal(BaseModel):
    """一个目标/意图（设计 §8.2）。`plan` 是分解出的子任务/步骤序列。

    时间戳用 **UTC epoch 秒**（`time.time()`），不可用单调时钟——目标跨重启持久化，
    而单调时钟重启即归零，会让上一次启动留存的 deadline 与新目标错误比较。
    """

    id: str
    intent: str  # 自然语言意图
    parent: str | None = None  # 目标树：父目标 id
    priority: int = 0  # 越大越优先
    deadline: float | None = None  # 截止时间（UTC epoch 秒，越早越急）；None 表示无期限
    status: str = STATUS_PENDING
    plan: list[str] = Field(default_factory=list)  # 分解出的步骤
    created_ts: float = (
        0.0  # 登记时间（UTC epoch 秒；GoalStore.add 补默认值，稳定 tiebreak）
    )


def is_actionable(goal: Goal) -> bool:
    """目标是否可被推进/仲裁（pending 或 active）。"""
    return goal.status in ACTIONABLE_STATUSES
