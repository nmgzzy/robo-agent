"""规划与重规划（设计 §8.2，实现计划 §P5 任务②）：把目标意图分解为子步骤。

`plan_goal` 用决策大脑把一句话意图拆成有序、可执行的步骤序列（每行一步）。
重规划 = 重新调用并用 `GoalStore.set_plan` 覆盖旧 `plan`。解析按行 + 去前缀编号/符号，
对模型输出格式宽容（嵌入式下越鲁棒越好）；Mock 模型可确定性回放，离线可测。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from robot_agent import prompts
from robot_agent.goals.models import Goal
from robot_agent.reliability import DEFAULT_FALLBACK_TEXT

# 去掉行首的编号/项目符号：如 "1. " "2) " "- " "* " "• "。
_BULLET_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def _content_to_text(content: Any) -> str:
    """把模型回复的 content 规整为纯文本。

    Anthropic 扩展思考等会返回**内容块列表**（如 `[{"type":"text","text":...}, ...]`），
    直接 `str(list)` 会得到带转义换行/元数据的畸形串，导致按行解析失败。此处只抽取并拼接
    文本块。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _parse_steps(text: str, max_steps: int) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        stripped = _BULLET_RE.sub("", line.strip()).strip()
        if stripped:
            steps.append(stripped)
        if len(steps) >= max_steps:
            break
    return steps


async def plan_goal(
    model: BaseChatModel, goal: Goal, *, max_steps: int = 10
) -> list[str]:
    """让决策大脑把 `goal.intent` 分解为步骤列表（≤ max_steps）。不落库，调用方决定持久化。"""
    msg = await model.ainvoke(
        [HumanMessage(prompts.render("plan", intent=goal.intent))]
    )
    text = _content_to_text(msg.content)
    # 决策大脑降级（ResilientChatModel 重试耗尽返回保守话术）不是计划——丢弃，留待重试，
    # 避免把「停在原地等待」当成步骤持久化、永久阻止重规划。
    if text.strip() == DEFAULT_FALLBACK_TEXT:
        return []
    return _parse_steps(text, max_steps)
