"""目标驱动的空闲策略（设计 §8.2，实现计划 §P5 任务④）：driver 空闲时推进目标栈。

把 P4 的 driver 与 P5 的目标系统接起来：`idle_tick` 到点无外部事件时，从目标栈仲裁出
当前最该做的目标，包成一个 `goal_due` 事件开回合（每个目标用独立连续线程 `goal-<id>`）。

「看收件箱 + 看目标栈」的分工（设计 §8.1/§8.2）：紧急外部事件由收件箱**优先**取出
（P4 已实现）；空闲时才轮到目标栈。被紧急事件打断后，目标仍在 Store，下个空闲 tick
再次仲裁会选回它——天然实现「处理完恢复到被打断目标」。
"""

from __future__ import annotations

import time

from langchain_core.language_models import BaseChatModel

from robot_agent import prompts
from robot_agent.driver.events import KIND_GOAL_DUE, Event
from robot_agent.goals.arbitrate import arbitrate
from robot_agent.goals.models import STATUS_ACTIVE, Goal
from robot_agent.goals.planning import plan_goal
from robot_agent.goals.store import GoalStore


def _format_goal_prompt(goal: Goal) -> str:
    """把目标（含已分解的 plan）渲染成开回合的指令文本。"""
    if goal.plan:
        steps = "；".join(f"{i + 1}) {s}" for i, s in enumerate(goal.plan))
        return prompts.render("goal_turn", intent=goal.intent, steps=steps)
    return goal.intent


class GoalDrivenIdlePolicy:
    """空闲时推进目标栈的 `IdlePolicy`：仲裁选当前目标 → （首次）分解 → 开 `goal_due` 回合。

    - `mark_active`：被选中的目标顺手标记为 `active`（让外部可观测「在追哪个目标」）。
    - `planner_model`：非 None 时，目标**首次**被推进且 `plan` 为空则调 `plan_goal` 分解，
      持久化进 `goal.plan` 并把步骤注入回合指令——闭合「分解 → 逐步执行」（建议用独立/更省
      的模型，避免与回合决策共用脚本/预算）。已有 plan 的目标不重复分解。
    - `priority`：空闲目标事件的优先级（默认 0，确保真正紧急的外部事件能抢先）。
    - 无可推进目标时返回 None（交还给 driver → 待机）。
    """

    def __init__(
        self,
        goal_store: GoalStore,
        *,
        planner_model: BaseChatModel | None = None,
        max_plan_steps: int = 10,
        priority: int = 0,
        mark_active: bool = True,
        thread_prefix: str = "goal-",
    ) -> None:
        self.goal_store = goal_store
        self.planner_model = planner_model
        self.max_plan_steps = max_plan_steps
        self.priority = priority
        self.mark_active = mark_active
        self.thread_prefix = thread_prefix

    async def on_idle(self) -> Event | None:
        goal = arbitrate(await self.goal_store.list())
        if goal is None:
            return None  # 无目标可推进 → 待机
        if self.mark_active and goal.status != STATUS_ACTIVE:
            goal = await self.goal_store.mark(goal.id, STATUS_ACTIVE)
        # 规划接线：首次推进、尚未分解时，分解并持久化 plan，供本回合及后续按计划执行。
        if self.planner_model is not None and not goal.plan:
            steps = await plan_goal(
                self.planner_model, goal, max_steps=self.max_plan_steps
            )
            if steps:
                goal = await self.goal_store.set_plan(goal.id, steps)
        return Event(
            kind=KIND_GOAL_DUE,
            ts=time.monotonic(),
            payload={
                "text": _format_goal_prompt(goal),
                "thread_id": f"{self.thread_prefix}{goal.id}",
                "goal_id": goal.id,
            },
            priority=self.priority,
        )
