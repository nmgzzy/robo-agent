"""HAL 接口契约（设计 §5.3.1–§5.3.3）：核心唯一允许依赖的硬件抽象。

- `Observation`：感知源产出的统一观测信封（source/ts/frame/payload）。
- `SensorSource`：输入侧协议，流式产出 `Observation`。
- `Actuator`：输出侧协议，接收意图级 `command` 并异步执行。

三者都是**结构化协议（Protocol）**——实现侧只要鸭子类型匹配即可接入，
无需继承，核心与实现彻底解耦。控制算法（路径规划/避障/运动学/SLAM/TTS）
全部在接口背后的实现里，不进底座。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class Observation(BaseModel):
    """感知源产出的统一观测信封（设计 §5.3.2）。

    两条喂入路径按性质二选一（见设计 §5.3.2，落地在 P4）：
    - 事件型（语音指令、检测到新物体、低电告警）→ 汇入 driver 收件箱；
    - 连续只读量（位姿、电量、最近距离）→ 以快照写入 State 世界状态字段。
    """

    source: str  # "camera" / "lidar" / "asr" / "battery" / "pose" ...
    ts: float  # 单调时钟时间戳
    frame: str | None = None  # 坐标系（如适用）
    payload: dict  # 该来源的结构化数据：检测框 / ASR 文本 / 距离 / 位姿…


@runtime_checkable
class SensorSource(Protocol):
    """输入侧：感知源插件（设计 §5.3.2）。

    `stream()` 流式产出 `Observation`；拉取式实现可在其上自行封装 poll 语义。
    """

    name: str

    def stream(self) -> AsyncIterator[Observation]: ...


@runtime_checkable
class Actuator(Protocol):
    """输出侧：执行器插件（设计 §5.3.3）。

    所有动作的唯一落点。`execute` 接收**意图/指令级** `command`（去某坐标、设某速度、
    播报某文本），返回执行结果/句柄。是 async：工具层须 `await`，否则只是返回未执行的协程。
    """

    name: str

    async def execute(self, command: dict) -> dict: ...
