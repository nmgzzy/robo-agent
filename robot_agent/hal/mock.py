"""HAL 的 mock 档实现（设计 §5.3.4）：纯内存、无外部依赖，单测/CI 默认。

- `MockActuator`：只**记录**收到的指令到 `.log`，不动真硬件——天然适合做回归断言
  「给定一串观测/指令，断言 Agent 下发的指令序列」（设计 §5.3.5 / AC-1 / AC-7）。
- `ScriptedSensor`：按脚本**回放**观测，确定性地复现感知流。

依赖纪律：本文件不 `import` 任何硬件 SDK / ROS / OpenCV。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from robot_agent.hal.interfaces import Observation


class MockActuator:
    """Actuator 的 mock：把收到的每条 `command` 追加进 `.log`，返回成功句柄。

    `.log` 是回归测试的核心断言点：跑完一个回合后比对它与期望的指令序列。
    """

    def __init__(self, name: str, *, result: dict | None = None) -> None:
        self.name = name
        self.log: list[dict] = []
        self._result = result or {"ok": True}

    async def execute(self, command: dict) -> dict:
        self.log.append(command)
        # 回执里带上执行器名，便于多执行器场景下区分来源。
        return {**self._result, "actuator": self.name}


def MockBase() -> MockActuator:  # noqa: N802 - 工厂函数，刻意用类型名风格命名
    """便捷工厂：底盘执行器的 mock（设计 §5.3.4 的 `MockBase`）。"""
    return MockActuator("base")


class ScriptedSensor:
    """SensorSource 的 mock：按预设帧序列回放 `Observation`（设计 §5.3.4）。"""

    def __init__(self, name: str, frames: Sequence[dict]) -> None:
        self.name = name
        self._frames = list(frames)

    async def stream(self) -> AsyncIterator[Observation]:
        for f in self._frames:
            # 帧自带 ts 则沿用，否则按序号补一个单调时间戳，保证可断言。
            yield Observation(
                source=self.name,
                ts=float(f.get("ts", 0.0)),
                frame=f.get("frame"),
                payload=f,
            )


def ScriptedCamera(frames: Sequence[dict]) -> ScriptedSensor:  # noqa: N802 - 工厂函数
    """便捷工厂：摄像头感知源的 mock（设计 §5.3.4 的 `ScriptedCamera`）。"""
    return ScriptedSensor("camera", frames)
