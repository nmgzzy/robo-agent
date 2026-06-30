"""工具层（设计 §5.2）：把 HAL 执行器包成 `@tool`，是思考与物理世界之间的受控边界。

所有副作用都过工具（便于审计与安全拦截，见 §7/§9）。工具只持**接口引用**
（`Actuator`），不关心档位与实现；执行器经注册表按名取用（设计 §5.3.3 示例的 `effectors[...]`）。

`build_robot_tools(effectors)` 用工厂闭包绑定一组执行器，避免全局可变状态——
每个回合/测试可注入独立的 mock 执行器，回归时直接断言其 `.log`（AC-1 / AC-7）。

注意：执行器 `execute` 是 async，下发动作的工具必须 `async def` + `await`，
否则只是返回未执行的协程（设计 §5.3.3 明确提醒）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from robot_agent.governance.policy import GovernancePolicy
from robot_agent.hal.interfaces import Actuator
from robot_agent.safety import SafetyPolicy, confirm_or_block


def build_robot_tools(
    effectors: Mapping[str, Actuator],
    *,
    safety: SafetyPolicy | None = None,
    governance: GovernancePolicy | None = None,
) -> list[Any]:
    """构建机器人控制工具列表，闭包绑定给定的执行器注册表。

    返回：`[move_to, set_velocity, grasp, speak, get_world_state]`。
    前四个经 `Actuator` 下发动作；`get_world_state` 只读注入式世界状态，不触硬件。

    动作下发前的两道关卡（顺序：先硬后软）：
    - `governance`（§8.6/P9）：宪章硬约束 / 权限 / 限幅 / 限频，违反**直接拒绝**并记审计。
    - `safety`（§7/P2）：危险动作（高速 `set_velocity` / `grasp`）`interrupt` 门控，等确认。
    """

    def _govern(name: str, args: dict) -> str | None:
        """治理硬约束检查：拒绝则返回回执文本，放行返回 None。"""
        if governance is None:
            return None
        ok, reason = governance.check(name, args)
        return None if ok else f"{name} 被治理策略拒绝：{reason}"

    def _audit_safety(name: str, args: dict, approved: bool, note: str) -> None:
        """把安全门控的**人工覆盖结果**（批准/拒绝）记入审计（P9 覆盖审计）。"""
        if governance is not None:
            reason = note or ("人工批准" if approved else "人工拒绝")
            governance.audit.record(name, args, approved, reason)

    @tool
    async def move_to(x: float, y: float) -> str:
        """移动底盘到坐标 (x, y)。路径规划/避障在执行器实现侧，本工具只下发意图。"""
        blocked = _govern("move_to", {"x": x, "y": y})
        if blocked:
            return blocked
        res = await effectors["base"].execute({"action": "move_to", "x": x, "y": y})
        return f"move_to({x}, {y}) -> {res}"

    @tool
    async def set_velocity(vx: float, wz: float) -> str:
        """设置底盘线速度 vx(m/s)、角速度 wz(rad/s)。"""
        blocked = _govern("set_velocity", {"vx": vx, "wz": wz})
        if blocked:
            return blocked
        if safety is not None:
            approved, note = confirm_or_block(
                "set_velocity", {"vx": vx, "wz": wz}, safety
            )
            _audit_safety("set_velocity", {"vx": vx, "wz": wz}, approved, note)
            if not approved:
                return f"set_velocity({vx}, {wz}) 被安全门控拒绝：{note or '未确认'}"
        res = await effectors["base"].execute(
            {"action": "set_velocity", "vx": vx, "wz": wz}
        )
        return f"set_velocity({vx}, {wz}) -> {res}"

    @tool
    async def grasp(obj: str) -> str:
        """用机械臂抓取一个物体（按名称/标签）。"""
        blocked = _govern("grasp", {"target": obj})
        if blocked:
            return blocked
        if safety is not None:
            approved, note = confirm_or_block("grasp", {"target": obj}, safety)
            _audit_safety("grasp", {"target": obj}, approved, note)
            if not approved:
                return f"grasp({obj!r}) 被安全门控拒绝：{note or '未确认'}"
        res = await effectors["arm"].execute({"action": "grasp", "target": obj})
        return f"grasp({obj!r}) -> {res}"

    @tool
    async def speak(text: str) -> str:
        """通过扬声器播报文本。"""
        blocked = _govern("speak", {"text": text})
        if blocked:
            return blocked
        res = await effectors["speaker"].execute({"action": "speak", "text": text})
        return f"speak({text!r}) -> {res}"

    @tool
    def get_world_state(state: Annotated[dict, InjectedState]) -> str:
        """读取当前世界状态快照：位姿 pose、电量 battery、检测到的物体 detections。

        数据由外部感知源注入 State（设计 §5.1/§5.3.2），本工具只读，不触硬件。
        """
        snapshot = {
            "pose": state.get("pose"),
            "battery": state.get("battery"),
            "detections": state.get("detections") or [],
        }
        return json.dumps(snapshot, ensure_ascii=False)

    return [move_to, set_velocity, grasp, speak, get_world_state]
