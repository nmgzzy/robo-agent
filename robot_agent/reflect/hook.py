"""复盘触发（设计 §8.3，实现计划 §P6）：把记录+蒸馏挂到 driver 的回合后钩子。

设计的挂载点是「driver 定时触发或 post_model_hook」。这里用前者：返回一个 `on_turn`
async 钩子，挂到 `Driver.on_turn` 上——每回合自动记录 `Episode`，每满 `reflect_every`
个回合触发一次蒸馏（`reflect_and_distill`）。蒸馏建议用独立（更省）模型，避免与回合
决策共用脚本/预算。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.store.base import BaseStore

from robot_agent.reflect.distill import reflect_and_distill
from robot_agent.reflect.episode import episode_from_turn, record_episode


def make_reflect_hook(
    store: BaseStore,
    robot_id: str,
    *,
    distill_model: BaseChatModel | None = None,
    reflect_every: int = 0,
    min_episodes: int = 1,
    prune: bool = False,
) -> Callable[[Any], Awaitable[None]]:
    """构造 driver 的 `on_turn` 钩子：记录已完成回合 + 周期蒸馏（设计 §8.3）。

    - 被安全门控**暂停**（`turn.interrupted`）的回合**跳过**：其 result 含尚未执行的提议
      动作，记录会谎报动作发生；待后续 resume 完成回合再记真正执行的结果。
    - 每记录满 `reflect_every` 个**已完成**回合（>0 且配了 `distill_model`）触发一次蒸馏。
      用独立的完成计数器（不是 `turn.index`，后者会因暂停回合虚增）。
    """
    completed = 0

    async def on_turn(turn: Any) -> None:
        nonlocal completed
        if getattr(turn, "interrupted", False):
            return  # 暂停回合不记录（动作尚未真正发生）
        await record_episode(store, robot_id, episode_from_turn(turn))
        completed += 1
        if (
            distill_model is not None
            and reflect_every > 0
            and completed % reflect_every == 0
        ):
            await reflect_and_distill(
                distill_model, store, robot_id, min_episodes=min_episodes, prune=prune
            )

    return on_turn
