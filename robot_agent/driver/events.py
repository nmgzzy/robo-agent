"""事件与收件箱（设计 §8.1，实现计划 §P4）：自主引擎的输入汇聚层。

被动库只在有人 `ainvoke` 时动一下。要成为自主个体，需要一个**收件箱**把定时器、
外部消息、事件型传感观测（§5.3.2）统一汇入，由常驻 driver 消费。本模块定义：

- `Event`：统一事件信封（kind/ts/payload/priority）。
- `Inbox`：收件箱协议（put/get + 优先级 + 超时）——鸭子类型，便于替换实现。
- `PriorityInbox`：基于 `asyncio.PriorityQueue` 的实现，高优先级先出、同级 FIFO。
- `pump_sensor`：把**事件型**感知观测从 `SensorSource` 路由进收件箱。

依赖纪律：只依赖 HAL 接口与 asyncio，不碰硬件 SDK。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from robot_agent.hal.interfaces import Observation, SensorSource

# 常见事件类别（kind）。仅约定字符串，便于扩展，不做枚举强约束。
KIND_USER_MSG = "user_msg"  # 外部用户指令
KIND_SENSOR = "sensor"  # 事件型传感观测（检测到新物体、低电告警…）
KIND_TIMER = "timer"  # 定时器/空闲心跳触发的自发回合
KIND_GOAL_DUE = "goal_due"  # 目标到期（P5 衔接）


class Event(BaseModel):
    """统一事件信封（设计 §8.1）。`priority` 越大越紧急（可打断空闲）。"""

    kind: str
    ts: float = 0.0
    payload: dict = Field(default_factory=dict)
    priority: int = 0


@runtime_checkable
class Inbox(Protocol):
    """收件箱协议：put 投递、get 消费（带超时，超时返回 None 以驱动空闲策略）。"""

    async def put(self, event: Event) -> None: ...

    async def get(self, timeout: float | None) -> Event | None: ...


class PriorityInbox:
    """优先级收件箱：高 `priority` 先出；同级按到达顺序 FIFO。

    用 `(-priority, seq, event)` 入堆——`seq` 单调自增，既保证 FIFO，又避免
    `priority` 相同时去比较 `Event`（BaseModel 不可排序）。`get(timeout)` 超时返回 None。
    """

    def __init__(self) -> None:
        self._q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = 0

    async def put(self, event: Event) -> None:
        item = (-event.priority, self._seq, event)
        self._seq += 1
        await self._q.put(item)

    async def get(self, timeout: float | None) -> Event | None:
        if timeout is None:
            _, _, event = await self._q.get()
            return event
        if timeout <= 0:
            # 零/负超时 = 非阻塞读：wait_for(get(), 0) 会在协程启动前取消而漏读已入队项。
            try:
                _, _, event = self._q.get_nowait()
            except asyncio.QueueEmpty:
                return None
            return event
        try:
            _, _, event = await asyncio.wait_for(self._q.get(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return None
        return event

    def qsize(self) -> int:
        return self._q.qsize()


async def pump_sensor(
    source: SensorSource,
    inbox: Inbox,
    *,
    kind: str = KIND_SENSOR,
    priority: int = 0,
    predicate: Callable[[Observation], bool] | None = None,
) -> int:
    """把**事件型**感知观测从 `SensorSource` 路由进 `Inbox`（设计 §5.3.2 / §8.1 任务②）。

    连续只读量（位姿/电量）应走 State 世界状态快照，不经此处。`predicate(obs)->bool`
    可筛选只投递关心的观测（如「仅检测到新物体或低电时」）。返回投递的事件数。
    """
    count = 0
    async for obs in source.stream():
        if predicate is not None and not predicate(obs):
            continue
        # 保留 source/frame：合规实现把它们放在 Observation 字段而非 payload 里，
        # 丢了就无法区分来源或解读坐标系相对量。已在 payload 的同名键不覆盖。
        payload = dict(obs.payload)
        payload.setdefault("source", obs.source)
        if obs.frame is not None:
            payload.setdefault("frame", obs.frame)
        await inbox.put(Event(kind=kind, ts=obs.ts, payload=payload, priority=priority))
        count += 1
    return count
