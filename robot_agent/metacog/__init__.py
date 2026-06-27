"""元认知 / 自我监控（设计 §8.5，实现计划 §P8）。

对自身认知状态的感知（而非节点级容错）：循环/卡死检测、步数预算、不确定即上报。

- `detect_loop` / `steps_used`（detect）：纯函数检测。
- `MetacogPolicy` / `make_monitor_hook`（monitor）：装饰 `pre_model_hook`，越界则
  `interrupt` 上报或注入告警；`MetacogPolicy.metrics` 导出循环/预算越界计数。

经 `build_robot_agent(..., metacog=MetacogPolicy(...))` 接入。
"""

from __future__ import annotations

from robot_agent.metacog.detect import detect_loop, steps_used
from robot_agent.metacog.monitor import (
    ESCALATE,
    WARN,
    MetacogPolicy,
    make_monitor_hook,
)

__all__ = [
    "ESCALATE",
    "WARN",
    "MetacogPolicy",
    "detect_loop",
    "make_monitor_hook",
    "steps_used",
]
