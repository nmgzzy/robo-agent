"""元认知检测（设计 §8.5，实现计划 §P8）：循环/卡死识别 + 步数预算。

retry/timeout 是节点级容错，不是对自身认知状态的感知。本模块提供纯函数检测：

- `detect_loop`：本回合内连续重复**相同工具调用**（name+args）达阈值 → 判为无进展循环。
- `steps_used`：本回合已下发的工具调用步数（用于步数预算）。

只看「本回合」（最后一条 human 输入之后）——跨回合做相同动作通常是正常的，单回合内
反复同一动作才是 react 死循环。纯函数无副作用，便于在 `pre_model_hook` 内调用与单测。
"""

from __future__ import annotations

import json
from typing import Any


def _current_turn(messages: list[Any]) -> list[Any]:
    last_input = -1
    for i, m in enumerate(messages):
        if getattr(m, "type", None) == "human":
            last_input = i
    # 无 human 锚点则「本回合」无从界定：返回空，避免把历史动作误判为本回合循环。
    return messages[last_input:] if last_input >= 0 else []


def _tool_signatures(messages: list[Any]) -> list[tuple]:
    """提取每条 AI 决策的工具调用签名（name + 规范化 args），无工具调用的不计。"""
    sigs: list[tuple] = []
    for m in messages:
        if getattr(m, "type", None) != "ai":
            continue
        tcs = getattr(m, "tool_calls", None) or []
        if tcs:
            sig = tuple(
                sorted(
                    (
                        tc["name"],
                        json.dumps(
                            tc.get("args", {}), sort_keys=True, ensure_ascii=False
                        ),
                    )
                    for tc in tcs
                )
            )
            sigs.append(sig)
    return sigs


def detect_loop(messages: list[Any], max_repeats: int = 3) -> dict | None:
    """本回合末尾连续相同工具调用达 `max_repeats` 次则判为循环，返回 {signature, repeats}。"""
    if max_repeats is None or max_repeats <= 0:
        return None
    sigs = _tool_signatures(_current_turn(messages))
    if len(sigs) < max_repeats:
        return None
    last = sigs[-1]
    count = 0
    for s in reversed(sigs):
        if s == last:
            count += 1
        else:
            break
    if count >= max_repeats:
        return {"signature": list(last), "repeats": count}
    return None


def steps_used(messages: list[Any]) -> int:
    """本回合已下发的工具调用步数（用于步数预算判定）。"""
    return len(_tool_signatures(_current_turn(messages)))
