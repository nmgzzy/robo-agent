"""自主引擎（设计 §8.1，实现计划 §P4）：driver + 收件箱 + 事件总线。

**被动库 → 自主个体的分界线**。底座（create_react_agent）只在被 `ainvoke` 时动一下；
本子包提供一个常驻 `Driver`，把定时器/外部消息/事件型传感观测统一汇入 `Inbox`，
由主循环消费——无人输入时也能按空闲策略自发开回合（满足 FR-7 / AC-4）。

组装示例：

    inbox = PriorityInbox()
    driver = Driver(graph, inbox, idle_policy=PromptIdlePolicy("巡视环境并报告异常"))
    await driver.submit(user_message("把杯子拿给我"))   # 外部事件
    await driver.run()                                  # 常驻，直到 stop()
"""

from __future__ import annotations

from robot_agent.driver.events import (
    KIND_GOAL_DUE,
    KIND_SENSOR,
    KIND_TIMER,
    KIND_USER_MSG,
    Event,
    Inbox,
    PriorityInbox,
    pump_sensor,
)
from robot_agent.driver.loop import DEFAULT_IDLE_TICK, Driver, TurnResult
from robot_agent.driver.policy import (
    IdlePolicy,
    PromptIdlePolicy,
    StandbyPolicy,
    default_decide_thread,
    default_make_input,
    resume_event,
    user_message,
)

__all__ = [
    "DEFAULT_IDLE_TICK",
    "Driver",
    "Event",
    "IdlePolicy",
    "Inbox",
    "KIND_GOAL_DUE",
    "KIND_SENSOR",
    "KIND_TIMER",
    "KIND_USER_MSG",
    "PriorityInbox",
    "PromptIdlePolicy",
    "StandbyPolicy",
    "TurnResult",
    "default_decide_thread",
    "default_make_input",
    "pump_sensor",
    "resume_event",
    "user_message",
]
