"""技能模型（设计 §8.7，实现计划 §P10）：把成功的计划存成**数据**而非代码。

工具是启动时硬编码的 Python，运行时不能自我扩展。技能库把「成功的动作序列」存成
数据、检索复用、动态装配进 `tools`（Voyager 思路）——这是「成长」的载体。

一个 `Skill` 是一段命名的动作序列（`actions`），每步 `{"tool": 名, "args": 参数}`。
（参数化技能——按入参改写 args——是后续扩展；当前为固定计划的复用。）
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# 技能的 Store namespace 种类。
KIND_SKILLS = "skills"


class Skill(BaseModel):
    """一段可复用的技能（设计 §8.7）。`actions` 是有序动作步骤。"""

    id: str
    name: str  # 用作动态工具名（应为合法标识符）
    description: str = ""  # 何时用、做什么——供检索与 LLM 选用
    actions: list[dict] = Field(default_factory=list)  # [{"tool": ..., "args": {...}}]
    created_ts: float = 0.0
