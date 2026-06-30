"""前端通道层（可拔插接口）：把常驻 agent 接到外部世界的稳定边界。

本仓库的天然分界线是 `Driver`（driver/loop.py）的 `submit(Event)`（输入）与
`on_turn(TurnResult)`（输出）。Web / IM / 麦克风扬声器等本质都是这条边界上不同的「通道」。
本子包把这条边界封装为：

- `AgentService`（service.py）：**通道无关门面**——对话投递 / 历史 / 记忆 / 工具 / 健康度，
  以及把回合输出广播给所有订阅者。核心闭环（create_react_agent）对通道一无所知。
- `Channel`（channel.py）：通道契约（鸭子类型）。新增一种前端 = 实现 `start/stop`，
  输入走 `service.submit_user_text`、输出 `service.subscribe`。
- `build_default_service`：一行装配离线可跑（mock + 内存存储 + 常驻 driver）的服务。

第一个通道实现是 Web 控制台（`robot_agent.frontends.web`）：纯 stdlib + SSE，零第三方依赖。
"""

from __future__ import annotations

from robot_agent.frontends.channel import Channel
from robot_agent.frontends.service import (
    DEFAULT_USER_THREAD,
    AgentService,
    build_default_service,
)

__all__ = [
    "DEFAULT_USER_THREAD",
    "AgentService",
    "Channel",
    "build_default_service",
]
