"""安全 / 对齐策略层（设计 §8.6，实现计划 §P9）：物理机器人不能只靠 prompt「小心点」。

`interrupt`（P2）只是「能暂停」的机制；这里是**策略层**，在动作下发前做硬性校验：

- **宪章硬约束**：一组规则谓词，命中即拦截（违反不可批准，区别于 P2 的可确认门控）。
- **工具权限范围**：白名单/黑名单，越权直接拒绝。
- **危险动作限幅**：速度等超硬上限拒绝（物理极限，非可商量的危险阈值）。
- **危险动作限频**：单工具累计调用上限，超频拒绝。
- **审计**：每次允许/拒绝都记入 `AuditLog`，可离线审查（衔接 P10 决策日记）。

在工具封装层执行（设计挂载点）：`build_robot_tools(effectors, governance=...)` 会让每个
副作用工具在 `await execute` 前过 `GovernancePolicy.check`，拒绝则不下发、回执原因并记审计。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# 宪章规则：给 (工具名, 参数) → 违反原因（None 表示通过）。
ConstitutionRule = Callable[[str, dict], "str | None"]


@dataclass
class ToolPermission:
    """工具权限范围。`allowed=None` 表示不限白名单；`denied` 始终优先拒绝。"""

    allowed: frozenset[str] | None = None
    denied: frozenset[str] = frozenset()


@dataclass
class AmplitudeLimit:
    """危险动作幅度硬上限（物理极限，超过直接拒绝，不可批准）。"""

    max_vx: float = 1.5  # m/s
    max_wz: float = 3.0  # rad/s


@dataclass
class AuditEntry:
    action: str
    args: dict
    allowed: bool
    reason: str = ""


class AuditLog:
    """结构化审计日志：记录每次治理决策（允许/拒绝 + 原因）。"""

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def record(self, action: str, args: dict, allowed: bool, reason: str = "") -> None:
        self.entries.append(AuditEntry(action, dict(args), allowed, reason))

    @property
    def denials(self) -> list[AuditEntry]:
        return [e for e in self.entries if not e.allowed]


@dataclass
class GovernancePolicy:
    """安全/对齐策略集合（设计 §8.6）。各项为 None/空则不启用该项检查。"""

    permission: ToolPermission | None = None
    constitution: list[ConstitutionRule] = field(default_factory=list)
    amplitude: AmplitudeLimit | None = None
    rate_limit: dict[str, int] | None = None  # 工具名 → 累计调用上限
    audit: AuditLog = field(default_factory=AuditLog)
    _counts: dict[str, int] = field(default_factory=dict, repr=False)

    def _deny(self, name: str, args: dict, reason: str) -> tuple[bool, str]:
        self.audit.record(name, args, False, reason)
        return False, reason

    def check(self, name: str, args: dict) -> tuple[bool, str]:
        """校验一次工具调用。返回 `(allowed, reason)` 并记审计；允许时累计限频计数。"""
        # 1) 权限范围
        if self.permission is not None:
            if name in self.permission.denied:
                return self._deny(name, args, f"工具 {name!r} 被列入黑名单")
            if (
                self.permission.allowed is not None
                and name not in self.permission.allowed
            ):
                return self._deny(name, args, f"工具 {name!r} 不在允许范围内")

        # 2) 宪章硬约束
        for rule in self.constitution:
            reason = rule(name, args)
            if reason:
                return self._deny(name, args, f"违反宪章：{reason}")

        # 3) 危险动作限幅
        if self.amplitude is not None and name == "set_velocity":
            vx = abs(float(args.get("vx", 0.0)))
            wz = abs(float(args.get("wz", 0.0)))
            if vx > self.amplitude.max_vx or wz > self.amplitude.max_wz:
                return self._deny(
                    name,
                    args,
                    f"超过速度硬上限（vx≤{self.amplitude.max_vx} wz≤{self.amplitude.max_wz}）",
                )

        # 4) 危险动作限频（累计上限）
        if self.rate_limit is not None and name in self.rate_limit:
            used = self._counts.get(name, 0)
            if used >= self.rate_limit[name]:
                return self._deny(
                    name, args, f"{name!r} 调用超过限频上限 {self.rate_limit[name]}"
                )
            self._counts[name] = used + 1

        self.audit.record(name, args, True)
        return True, ""
