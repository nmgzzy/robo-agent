"""HAL 注册表（设计 §5.3.4）：按档位选择执行器实现，核心代码与图结构不变。

`EffectorRegistry` 是「执行器名 → Actuator」的映射，工具层只按名字取用、不关心档位与实现。
`build_effectors(tier)` 按档位装配一组执行器：

- `mock`（P1 内置）：纯内存 `MockActuator`，无外部依赖。
- `real` / `sim`：留给后续实现包（`hal/plugins/<impl>`），届时在此分发，
  核心依赖树保持干净——硬件 SDK 只在那些实现包里 `import`。
"""

from __future__ import annotations

from robot_agent.hal.interfaces import Actuator
from robot_agent.hal.mock import MockActuator

# P1 闭环涉及的执行器名（与工具层 §5.2 一一对应）。
DEFAULT_EFFECTOR_NAMES = ("base", "arm", "speaker")


class EffectorRegistry(dict[str, Actuator]):
    """执行器注册表：`name -> Actuator`。

    继承 `dict` 以便工具层直接 `effectors["base"]` 取用（对齐设计 §5.3.3 示例），
    取不到时给出清晰错误而非裸 `KeyError`。
    """

    def __missing__(self, name: str) -> Actuator:
        raise KeyError(
            f"未注册执行器 {name!r}；已注册：{sorted(self)}。"
            "请检查 build_effectors 档位或工具与执行器名是否一致。"
        )


def build_effectors(
    tier: str = "mock",
    *,
    names: tuple[str, ...] = DEFAULT_EFFECTOR_NAMES,
) -> EffectorRegistry:
    """按档位装配执行器注册表。

    P1 仅支持 `mock`；`real` / `sim` 由后续实现包接入（此处显式拒绝以免误用真硬件）。
    """
    if tier == "mock":
        return EffectorRegistry({n: MockActuator(n) for n in names})
    raise ValueError(
        f"未知或暂未实现的 HAL 档位 {tier!r}；P1 仅内置 'mock'，"
        "real / sim 待后续实现包（hal/plugins/<impl>）接入。"
    )
