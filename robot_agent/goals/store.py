"""目标持久化（设计 §8.2，实现计划 §P5 任务①）：Store `(robot_id,"goals")` 上的 CRUD。

把目标栈存进长期记忆 Store，使其跨回合、跨重启存活（衔接 P2 恢复）。每个目标一条记录，
`key=goal.id`、`value=goal.model_dump()`。仲裁（`arbitrate`）只读列表，不在此处耦合。
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence

from langgraph.store.base import BaseStore
from robot_agent.goals.models import KIND_GOALS, Goal

# asearch 默认 limit=10；分页拉满，避免目标数 >10 时漏选最高优先目标。
_PAGE = 100


class GoalStore:
    """目标栈的持久化句柄，绑定一个 `robot_id` 的 `(robot_id,"goals")` namespace。"""

    def __init__(self, store: BaseStore, robot_id: str) -> None:
        self.store = store
        self.robot_id = robot_id

    @property
    def ns(self) -> tuple[str, str]:
        return (self.robot_id, KIND_GOALS)

    async def add(self, goal: Goal) -> Goal:
        """登记一个目标（已存在同 id 则覆盖）。返回登记的目标。

        未显式给 `created_ts` 时，用 UTC epoch 时间戳补上——否则全为 0.0，
        同优先级/期限时的「登记早者先」tiebreak 形同虚设（见 arbitrate）。
        """
        if not goal.created_ts:
            goal = goal.model_copy(update={"created_ts": time.time()})
        await self.store.aput(self.ns, goal.id, goal.model_dump())
        return goal

    async def get(self, goal_id: str) -> Goal | None:
        """按 id 读取目标；不存在返回 None。"""
        item = await self.store.aget(self.ns, goal_id)
        return Goal(**copy.deepcopy(dict(item.value))) if item is not None else None

    async def list(self, *, status: str | None = None) -> list[Goal]:
        """列出全部目标（分页拉满）；`status` 给定时按状态过滤。"""
        items = []
        offset = 0
        while True:
            batch = await self.store.asearch(self.ns, limit=_PAGE, offset=offset)
            items.extend(batch)
            if len(batch) < _PAGE:
                break
            offset += _PAGE
        goals = [Goal(**copy.deepcopy(dict(it.value))) for it in items]
        if status is not None:
            goals = [g for g in goals if g.status == status]
        return goals

    async def update(self, goal_id: str, **fields) -> Goal:
        """局部更新目标字段（status/plan/priority/deadline…）。目标不存在则 KeyError。"""
        goal = await self.get(goal_id)
        if goal is None:
            raise KeyError(f"未知目标 {goal_id!r}（robot_id={self.robot_id!r}）。")
        updated = goal.model_copy(update=fields)
        await self.store.aput(self.ns, goal_id, updated.model_dump())
        return updated

    async def mark(self, goal_id: str, status: str) -> Goal:
        """更新目标状态（active/blocked/done/abandoned…）的便捷封装。"""
        return await self.update(goal_id, status=status)

    async def set_plan(self, goal_id: str, plan: Sequence[str]) -> Goal:
        """写入/重写目标的分解步骤（重规划，见 planning.plan_goal）。"""
        return await self.update(goal_id, plan=list(plan))

    async def remove(self, goal_id: str) -> None:
        """删除一个目标记录。"""
        await self.store.adelete(self.ns, goal_id)
