"""回合记录（设计 §8.3，实现计划 §P6）：把一个回合的经历写入 episodic 记忆。

`episodic` namespace 此前只有约定、无写入路径。复盘闭环的第一步是**记录**：每个回合
结束后，把「想做什么（intent）→ 实际下发了什么（actions）→ 结果如何（outcome）」存成
一条 `Episode`，供后续蒸馏（distill）读取，把重复经验提炼为 `facts`/`prefs`。

时间戳用 UTC epoch（与 goals 一致，跨重启稳定）。读取分页拉满，避开 asearch 默认 limit=10。
"""

from __future__ import annotations

import copy
import time
from typing import Any

from pydantic import BaseModel, Field

from langgraph.store.base import BaseStore
from robot_agent.memory import KIND_EPISODIC, ns

_PAGE = 100


class Episode(BaseModel):
    """一个回合的结构化经历（设计 §8.3 的 intended vs actual 素材）。"""

    id: str
    ts: float = 0.0  # UTC epoch 秒
    intent: str = ""  # 这回合想做什么（触发事件/目标意图）
    actions: list[str] = Field(default_factory=list)  # 实际下发的动作（工具调用）
    outcome: str = ""  # 结果摘要（回合末的回答）
    thread_id: str = ""
    goal_id: str | None = None
    success: bool | None = None  # 自评（蒸馏时可由 LLM 回填）；None 表示未评


def _content_to_text(content: Any) -> str:
    """把消息 content 规整为纯文本（兼容 Anthropic 结构化内容块列表）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b["text"]
            for b in content
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        ]
        return "\n".join(parts)
    return str(content)


def _extract_actions(messages: list[Any]) -> list[str]:
    actions: list[str] = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            actions.append(f"{tc['name']}({tc.get('args', {})})")
    return actions


def _extract_outcome(messages: list[Any]) -> str:
    # 最后一条「无工具调用」的 AI 消息即本回合的最终回答。
    for m in reversed(messages):
        if getattr(m, "type", None) == "ai" and not (getattr(m, "tool_calls", None)):
            return _content_to_text(m.content)
    return ""


def _current_turn_messages(messages: list[Any]) -> list[Any]:
    """切出**本回合**新增的消息（默认 driver 复用 thread_id，checkpointer 累积全历史）。

    以「最后一条输入消息（human）起到末尾」为本回合边界——`make_input` 每回合注入一条
    `HumanMessage` 作起点；切掉之前回合，避免把历史工具调用重复算进本回合的 actions。
    """
    last_input = -1
    for i, m in enumerate(messages):
        if getattr(m, "type", None) == "human":
            last_input = i
    # 无 human 锚点则本回合无从界定：返回空，避免把上一回合的动作算进本回合。
    return messages[last_input:] if last_input >= 0 else []


def episode_from_turn(turn: Any) -> Episode:
    """从 driver 的 `TurnResult` 提取一条 `Episode`（intent/actions/outcome）。"""
    payload = getattr(turn.event, "payload", {}) or {}
    intent = payload.get("text") or payload.get("instruction") or f"[{turn.event.kind}]"
    messages = []
    if isinstance(turn.result, dict):
        messages = turn.result.get("messages") or []
    turn_messages = _current_turn_messages(messages)
    ts = time.time()
    # id 跨 driver 重启唯一：微秒时间戳 + 回合序号（turn.index 会随新 Driver 重置）。
    episode_id = f"ep-{int(ts * 1_000_000):020d}-{turn.index:06d}"
    return Episode(
        id=episode_id,
        ts=ts,
        intent=str(intent),
        actions=_extract_actions(turn_messages),
        outcome=_extract_outcome(turn_messages),
        thread_id=turn.thread_id,
        goal_id=payload.get("goal_id"),
    )


async def record_episode(store: BaseStore, robot_id: str, episode: Episode) -> Episode:
    """把一条 `Episode` 写入 `(robot_id,"episodic")`。"""
    await store.aput(ns(robot_id, KIND_EPISODIC), episode.id, episode.model_dump())
    return episode


async def read_episodes(store: BaseStore, robot_id: str) -> list[Episode]:
    """读出全部 episodic 记录（分页拉满）。"""
    items: list[Any] = []
    offset = 0
    while True:
        batch = await store.asearch(
            ns(robot_id, KIND_EPISODIC), limit=_PAGE, offset=offset
        )
        items.extend(batch)
        if len(batch) < _PAGE:
            break
        offset += _PAGE
    episodes = [Episode(**copy.deepcopy(dict(it.value))) for it in items]
    episodes.sort(key=lambda e: (e.ts, e.id))
    return episodes


async def prune_episodes(store: BaseStore, robot_id: str, ids: list[str]) -> int:
    """删除指定 episodic 记录（蒸馏后清理已消化的经历）。返回删除数。"""
    for eid in ids:
        await store.adelete(ns(robot_id, KIND_EPISODIC), eid)
    return len(ids)
