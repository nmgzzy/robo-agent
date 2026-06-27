"""技能库（设计 §8.7，实现计划 §P10）：技能作为数据，检索复用 + 动态工具加载。

- `Skill`（models）：命名的动作序列。
- `SkillStore`（store）：Store `(robot_id,"skills")` 持久化 + 关键字检索。
- `build_skill_tools`（tools）：把技能动态装配成运行时 `@tool`，可选过 P9 治理校验。
"""

from __future__ import annotations

from robot_agent.skills.models import KIND_SKILLS, Skill
from robot_agent.skills.store import SkillStore
from robot_agent.skills.tools import build_skill_tools

__all__ = [
    "KIND_SKILLS",
    "Skill",
    "SkillStore",
    "build_skill_tools",
]
