"""硬件抽象层（HAL）：感知输入 / 执行输出 / Mock（对齐设计 §5.3）。

依赖倒置：核心只依赖本包的**接口**（`SensorSource` / `Actuator` / `Observation`），
具体实现分三档（real / sim / mock），通过注册表选择。核心代码与图结构永不变，
也永不 `import` 任何硬件 SDK / ROS / OpenCV / 控制算法——那些只许出现在实现侧。

P1 仅内置 `mock` 档（纯内存、无外部依赖、CI 默认）；`real` / `sim` 留给后续实现包。
"""

from __future__ import annotations

from robot_agent.hal.interfaces import Actuator, Observation, SensorSource
from robot_agent.hal.mock import MockActuator, MockBase, ScriptedCamera, ScriptedSensor
from robot_agent.hal.registry import EffectorRegistry, build_effectors

__all__ = [
    "Actuator",
    "EffectorRegistry",
    "MockActuator",
    "MockBase",
    "Observation",
    "ScriptedCamera",
    "ScriptedSensor",
    "SensorSource",
    "build_effectors",
]
