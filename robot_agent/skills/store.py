"""技能持久化与检索（设计 §8.7，实现计划 §P10）：Store `(robot_id,"skills")`。

技能存成数据，按需检索复用（与 P6 复盘衔接——成功回合可沉淀为技能）。检索用关键字
匹配 name/description（嵌入式首选，无额外算力；语义检索属可选增强）。
"""

from __future__ import annotations

import copy
import time

from langgraph.store.base import BaseStore

from robot_agent.skills.models import KIND_SKILLS, Skill

_PAGE = 100


class SkillStore:
    """技能库的持久化句柄，绑定 `(robot_id,"skills")` namespace。"""

    def __init__(self, store: BaseStore, robot_id: str) -> None:
        self.store = store
        self.robot_id = robot_id

    @property
    def ns(self) -> tuple[str, str]:
        return (self.robot_id, KIND_SKILLS)

    async def add(self, skill: Skill) -> Skill:
        """登记/覆盖一个技能（未给 created_ts 则补 UTC epoch）。"""
        if not skill.created_ts:
            skill = skill.model_copy(update={"created_ts": time.time()})
        await self.store.aput(self.ns, skill.id, skill.model_dump())
        return skill

    async def get(self, skill_id: str) -> Skill | None:
        item = await self.store.aget(self.ns, skill_id)
        return Skill(**copy.deepcopy(dict(item.value))) if item is not None else None

    async def list(self) -> list[Skill]:
        """列出全部技能（分页拉满）。"""
        items = []
        offset = 0
        while True:
            batch = await self.store.asearch(self.ns, limit=_PAGE, offset=offset)
            items.extend(batch)
            if len(batch) < _PAGE:
                break
            offset += _PAGE
        return [Skill(**copy.deepcopy(dict(it.value))) for it in items]

    async def search(self, query: str) -> list[Skill]:
        """关键字检索：query 命中 name 或 description（大小写不敏感）。"""
        q = query.lower()
        return [
            s
            for s in await self.list()
            if q in s.name.lower() or q in s.description.lower()
        ]

    async def remove(self, skill_id: str) -> None:
        await self.store.adelete(self.ns, skill_id)
