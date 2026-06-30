"""通道契约（可拔插前端层）：把外部世界接到 `AgentService` 的稳定边界。

设计动机见 `AgentService`（service.py）：Web / IM / 麦克风扬声器等本质都是同一条边界上
不同的「通道」——把外部输入翻译成投递给门面的用户文本/事件，把门面广播出的回合输出翻译回
各自的外部表现（网页 SSE、IM 卡片、TTS 语音…）。核心闭环对此一无所知。

新增一种通道 = 实现本 `Channel` Protocol（鸭子类型，无需继承）：
- `start(service)`：绑定门面、开始收发（输入侧调 `service.submit_user_text`，
  输出侧 `service.subscribe()` 订阅回合广播）。
- `stop()`：优雅停止本通道（不负责停 service 本身）。

web 通道见 `robot_agent.frontends.web`。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from robot_agent.frontends.service import AgentService


@runtime_checkable
class Channel(Protocol):
    """前端通道协议：把一种外部 I/O（web/im/语音）桥接到通道无关的 `AgentService`。"""

    async def start(self, service: AgentService) -> None: ...

    async def stop(self) -> None: ...
