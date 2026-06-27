"""安全门控（设计 §7「安全门控」，实现计划 §P2.4）：危险动作执行前 `interrupt`。

物理机器人不能只靠 prompt「小心点」。本模块在**副作用发生前**用 `interrupt` 暂停闭环，
等规则或人工给出确认（`Command(resume=...)`）后才放行；拒绝则跳过动作并回执原因。

危险判定（`SafetyPolicy`，可配阈值）：
- `set_velocity`：线速度 `vx` 或角速度 `wz` 超过阈值（高速）。
- `grasp`：机械臂抓取（默认一律门控，夹爪有夹伤风险）。

关键时序约束：`interrupt` 在恢复时会**从节点头部重放**整段逻辑（见 `langgraph.types.interrupt`
文档）。因此门控必须先于执行器下发——`confirm_or_block` 返回放行后才 `await execute`，
重放时再次进入 `interrupt` 即拿到 resume 值、只下发一次，**不会重复执行**副作用。

恢复值约定（`Command(resume=...)`）：
- `True` / `{"approved": True}` → 放行；可带 `{"reason": "..."}` 记审计。
- `False` / `{"approved": False}` → 拒绝；动作被跳过，工具回执拒绝原因。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from langgraph.types import interrupt


@dataclass(frozen=True)
class SafetyPolicy:
    """危险动作阈值（设计 §7）。超阈或命中规则即触发 `interrupt` 人工/规则确认。"""

    max_vx: float = 0.5  # m/s，底盘线速度安全上限
    max_wz: float = 1.0  # rad/s，底盘角速度安全上限
    gate_grasp: bool = True  # 抓取是否一律门控


def reject_reason(name: str, args: dict[str, Any]) -> str | None:
    """非法/无意义命令的**硬拒绝**原因（与策略无关，永不可批准）。

    目前：`set_velocity` 的非有限速度（NaN/Inf）——`abs(x) > 阈值` 恒为 False 会漏判，
    且这类值送到执行器无意义且危险，必须直接拒绝，不进入可批准的 `interrupt`。
    """
    if name == "set_velocity":
        vx = float(args.get("vx", 0.0))
        wz = float(args.get("wz", 0.0))
        if not (math.isfinite(vx) and math.isfinite(wz)):
            return f"非有限速度 vx={args.get('vx')} wz={args.get('wz')}（拒绝下发）"
    return None


def danger_reason(name: str, args: dict[str, Any], policy: SafetyPolicy) -> str | None:
    """判断一次工具调用是否**需要确认**；是则返回人类可读原因，否则返回 None。

    仅覆盖「危险但合法、可经人工/规则批准」的动作（高速运动、抓取）。
    非法命令（如非有限速度）由 `reject_reason` 硬拒绝，不在此处。
    """
    if name == "set_velocity":
        vx = float(args.get("vx", 0.0))
        wz = float(args.get("wz", 0.0))
        # 非有限值先被 reject_reason 拦下；此处只在有限值上比较阈值。
        if (
            math.isfinite(vx)
            and math.isfinite(wz)
            and (abs(vx) > policy.max_vx or abs(wz) > policy.max_wz)
        ):
            return (
                f"高速运动 vx={args.get('vx')} wz={args.get('wz')}"
                f"（阈值 vx≤{policy.max_vx} wz≤{policy.max_wz}）"
            )
    elif name == "grasp" and policy.gate_grasp:
        return f"机械臂抓取 {args.get('target') or args.get('obj')!r}"
    return None


def confirm_or_block(
    name: str, args: dict[str, Any], policy: SafetyPolicy
) -> tuple[bool, str]:
    """放行/拦截决策。硬拒绝 > 需确认 > 放行。

    返回 `(approved, note)`：`approved=False` 时调用方应跳过执行并回执 `note`。
    - 非法命令（`reject_reason`）：直接拒绝，**不** `interrupt`（不可批准）。
    - 危险命令（`danger_reason`）：`interrupt` 等确认，需 checkpointer。
    - 其余：放行。
    """
    hard = reject_reason(name, args)
    if hard is not None:
        return False, hard  # 硬拒绝，不可批准

    reason = danger_reason(name, args, policy)
    if reason is None:
        return True, ""  # 非危险动作：放行

    decision = interrupt(
        {
            "type": "safety_confirmation",
            "action": name,
            "args": args,
            "reason": reason,
        }
    )
    # 物理安全边界：fail-closed——只有显式布尔 True（或 {"approved": True}）才放行；
    # 字符串 "deny" / 非布尔等畸形恢复值一律视为拒绝（`bool("deny")` 为 True 会误放行）。
    if isinstance(decision, dict):
        return decision.get("approved") is True, str(decision.get("reason", ""))
    return decision is True, ""
