"""复盘闭环（设计 §8.3，实现计划 §P6）：回合自评 + episodic→semantic 蒸馏。

让「越用越懂」从愿望变机制：

- `Episode` / `record_episode` / `episode_from_turn`（episode）：把回合经历写入 episodic。
- `reflect_and_distill`（distill）：读 episodic，LLM 蒸馏为 facts/prefs 写回。
- `make_reflect_hook`（hook）：挂到 driver `on_turn`，自动记录 + 周期蒸馏（衔接 P4）。

蒸馏出的偏好沿用 `{"value": ...}` 包装写入，会被 `pre_model_hook` 在后续回合自动注入。
"""

from __future__ import annotations

from robot_agent.reflect.distill import (
    DistillResult,
    parse_distilled,
    reflect_and_distill,
)
from robot_agent.reflect.episode import (
    Episode,
    episode_from_turn,
    prune_episodes,
    read_episodes,
    record_episode,
)
from robot_agent.reflect.hook import make_reflect_hook

__all__ = [
    "DistillResult",
    "Episode",
    "episode_from_turn",
    "make_reflect_hook",
    "parse_distilled",
    "prune_episodes",
    "read_episodes",
    "record_episode",
    "reflect_and_distill",
]
