"""空闲策略与回合编排（设计 §8.1，实现计划 §P4 任务④⑤）。

driver 消费收件箱时有两类决定：

- **空闲策略**（`IdlePolicy`）：`idle_tick` 内无事件时做什么——待机 / 自发巡检 / 整理记忆。
  返回一个要执行的 `Event`（开一个自发回合）或 `None`（纯待机，本 tick 不动）。
- **回合编排**：`decide_thread(event)` 决定开哪个 `thread_id`（与 P2 恢复语义衔接）；
  `make_input(event)` 把事件翻译成图输入（`messages` + 可选世界状态快照）。

目标系统（P5）落地后，`decide_thread` 会进一步「看收件箱 + 看目标栈」做多目标仲裁。
"""

from __future__ import annotations

import json
import time
from typing import Protocol, runtime_checkable

from langchain_core.messages import HumanMessage

from robot_agent.driver.events import KIND_TIMER, KIND_USER_MSG, Event

# make_input 会把这些键从事件 payload 透传进 State 世界状态字段（只读快照，§5.1）。
WORLD_STATE_KEYS = ("pose", "battery", "detections")


@runtime_checkable
class IdlePolicy(Protocol):
    """空闲策略协议：`idle_tick` 超时无事件时被调用。返回 Event 则开回合，None 则待机。"""

    async def on_idle(self) -> Event | None: ...


class StandbyPolicy:
    """待机：空闲时什么都不做（最省电，默认）。"""

    async def on_idle(self) -> Event | None:
        return None


class PromptIdlePolicy:
    """自发回合：空闲时用一段固定自我提示开一个 `timer` 回合（巡检/整理记忆等）。

    `prompt` 决定空闲时的行为取向，例如「请巡视环境并报告异常」「请整理近期记忆」。
    """

    def __init__(self, prompt: str, *, priority: int = 0) -> None:
        self.prompt = prompt
        self.priority = priority

    async def on_idle(self) -> Event | None:
        return Event(
            kind=KIND_TIMER,
            ts=time.monotonic(),
            payload={"text": self.prompt},
            priority=self.priority,
        )


def default_decide_thread(event: Event) -> str:
    """默认回合选择：payload 显式 `thread_id` 优先，否则按事件类别归并到同一回合。

    同类事件归并到同一 `thread_id`（如所有用户对话进 `user_msg`、所有自发心跳进 `timer`），
    使长期记忆与上下文在该线程内连续累积（超长由 P2 的 `trim_messages` 控制）。
    """
    return event.payload.get("thread_id") or event.kind


def default_make_input(event: Event) -> dict:
    """把事件翻译成 `create_react_agent` 的输入 dict（messages + 世界状态快照）。

    取 messages 的优先级：显式 `messages` > `text`/`instruction` 文本 > 兜底序列化 payload。
    识别到的世界状态键（pose/battery/detections）透传为 State 只读字段。
    """
    payload = event.payload
    inp: dict = {}

    if payload.get("messages"):
        inp["messages"] = list(payload["messages"])
    else:
        text = payload.get("text") or payload.get("instruction")
        if text:
            inp["messages"] = [HumanMessage(str(text))]
        else:
            # 无自然语言载荷（纯传感事件）：序列化 payload 让 LLM 自行解读。
            body = json.dumps(payload, ensure_ascii=False, default=str)
            inp["messages"] = [HumanMessage(f"[{event.kind}] {body}")]

    for key in WORLD_STATE_KEYS:
        if key in payload:
            inp[key] = payload[key]
    return inp


# 便捷别名：把用户文本包成一个 user_msg 事件（外部接入常用）。
def user_message(
    text: str, *, priority: int = 0, thread_id: str | None = None
) -> Event:
    """构造一个用户消息事件（外部把人类指令投递进收件箱时的便捷构造器）。"""
    payload: dict = {"text": text}
    if thread_id is not None:
        payload["thread_id"] = thread_id
    return Event(
        kind=KIND_USER_MSG, ts=time.monotonic(), payload=payload, priority=priority
    )


def resume_event(thread_id: str, decision, *, priority: int = 100) -> Event:
    """构造一个**安全确认 resume** 事件，路由到被门控暂停的 `thread_id`（设计 §7 衔接）。

    `decision` 即 `Command(resume=...)` 的值（如 `{"approved": True}`）。默认高优先级，
    使确认能尽快被 driver 取出、续跑暂停的危险动作回合。
    """
    return Event(
        kind=KIND_USER_MSG,
        ts=time.monotonic(),
        payload={"thread_id": thread_id, "resume": decision},
        priority=priority,
    )
