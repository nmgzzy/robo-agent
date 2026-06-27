"""复盘蒸馏（设计 §8.3，实现计划 §P6）：把 episodic 经历提炼为 semantic 记忆。

「越用越懂」要从愿望变机制：周期性读 `episodic`，让决策大脑对比意图与结果，蒸馏出可
复用的长期知识，写回 `facts`（环境事实）/`prefs`（偏好）。写入沿用 `{"value": ...}` 包装，
与 `remember_fact` 一致，使 `pre_model_hook` 的注入/还原逻辑透明复用——蒸馏出的偏好会在
后续回合被自动检索注入并影响决策（验收②）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from langgraph.store.base import BaseStore
from robot_agent import prompts
from robot_agent.memory import KIND_FACTS, KIND_PREFS, ns
from robot_agent.reflect.episode import Episode, _content_to_text, read_episodes

# 解析蒸馏行：兼容全角冒号/等号。
_LINE_RE = re.compile(
    r"^\s*(fact|pref)\s*[:：]\s*(.+?)\s*[=＝]\s*(.+?)\s*$", re.IGNORECASE
)
_KIND_MAP = {"fact": KIND_FACTS, "pref": KIND_PREFS}


@dataclass
class DistillResult:
    """一次蒸馏的结果：看了几条经历、写回了哪些 (kind, key, value)。"""

    episodes_seen: int
    written: list[tuple[str, str, str]] = field(default_factory=list)


def _format_episodes(episodes: Sequence[Episode]) -> str:
    lines = []
    for e in episodes:
        actions = "、".join(e.actions) if e.actions else "（无动作）"
        lines.append(
            f"- 想做：{e.intent}；做了：{actions}；结果：{e.outcome or '（无）'}"
        )
    return "\n".join(lines)


def parse_distilled(text: str) -> list[tuple[str, str, str]]:
    """把蒸馏文本解析为 `(kind, key, value)` 列表（kind 已规整为 facts/prefs namespace 名）。"""
    out: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if m is None:
            continue
        kind = _KIND_MAP[m.group(1).lower()]
        out.append((kind, m.group(2).strip(), m.group(3).strip()))
    return out


# 单次蒸馏处理的最近 episode 上限：防止长跑下 prompt 无限增长、阻塞同步 on_turn 钩子。
DEFAULT_MAX_EPISODES = 50


async def reflect_and_distill(
    model: BaseChatModel,
    store: BaseStore,
    robot_id: str,
    *,
    min_episodes: int = 1,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    prune: bool = False,
) -> DistillResult:
    """读 episodic → LLM 蒸馏 → 写回 facts/prefs（设计 §8.3）。

    - `min_episodes`：少于此数不触发蒸馏（攒够再复盘，省算力）。
    - `max_episodes`：单次只取**最近** N 条送蒸馏，避免长跑下 prompt 无限增长。
    - `prune=True`：**仅当蒸馏成功产出非空结果**时，删除本次处理的那批 episodic 记录
      （模型畸形输出/降级 fallback 解析为空时不删，避免经验无谓丢失）。
    """
    episodes = await read_episodes(store, robot_id)
    if len(episodes) < min_episodes:
        return DistillResult(episodes_seen=len(episodes))

    batch = episodes[-max_episodes:]  # read_episodes 已按 ts 升序，取最近一批
    prompt = prompts.render("distill", episodes=_format_episodes(batch))
    msg = await model.ainvoke([HumanMessage(prompt)])
    items = parse_distilled(_content_to_text(msg.content))

    for kind, key, value in items:
        await store.aput(ns(robot_id, kind), key, {"value": value})

    if prune and items:  # 仅成功蒸馏出内容才清理，且只清本次处理的批次
        from robot_agent.reflect.episode import prune_episodes

        await prune_episodes(store, robot_id, [e.id for e in batch])

    return DistillResult(episodes_seen=len(batch), written=items)
