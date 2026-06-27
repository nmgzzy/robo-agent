"""动态工具加载（设计 §8.7，实现计划 §P10）：把技能装配成运行时 `@tool`。

`build_skill_tools(skills, effectors)` 为每个技能生成一个工具（名为 `skill_<name>`），
调用时依次执行技能的动作步骤。可选 `governance`：每步过 P9 治理校验，越权/超限即中止——
技能是「宏」，但不能借此绕过安全策略。

动作路由：技能步骤的 `tool` 名 → 执行器名 + 指令构造，与 `tools.build_robot_tools` 的
意图级映射一致（控制算法仍在执行器实现侧）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.tools import tool

from robot_agent.governance.policy import GovernancePolicy
from robot_agent.hal.interfaces import Actuator
from robot_agent.skills.models import Skill

# 技能动作 → (执行器名, 指令构造函数)。
_ROUTING: dict[str, Any] = {
    "move_to": ("base", lambda a: {"action": "move_to", "x": a["x"], "y": a["y"]}),
    "set_velocity": (
        "base",
        lambda a: {"action": "set_velocity", "vx": a["vx"], "wz": a["wz"]},
    ),
    "grasp": (
        "arm",
        lambda a: {"action": "grasp", "target": a.get("target") or a.get("obj")},
    ),
    "speak": ("speaker", lambda a: {"action": "speak", "text": a["text"]}),
}


def _make_skill_tool(
    skill: Skill,
    effectors: Mapping[str, Actuator],
    governance: GovernancePolicy | None,
):
    async def run_skill() -> str:
        done: list[str] = []
        for step in skill.actions:
            name = step["tool"]
            args = step.get("args", {})
            route = _ROUTING.get(name)
            if route is None:
                return f"技能 {skill.name} 中止：未知动作 {name!r}"
            if governance is not None:
                ok, reason = governance.check(name, args)
                if not ok:
                    return f"技能 {skill.name} 中止：{reason}"
            eff_name, build = route
            await effectors[eff_name].execute(build(args))
            done.append(name)
        return f"技能 {skill.name} 执行完成：{' → '.join(done)}"

    run_skill.__name__ = f"skill_{skill.name}"
    run_skill.__doc__ = skill.description or f"执行预存技能 {skill.name}。"
    return tool(run_skill)


def build_skill_tools(
    skills: Sequence[Skill],
    effectors: Mapping[str, Actuator],
    *,
    governance: GovernancePolicy | None = None,
) -> list[Any]:
    """把一组技能动态装配成工具列表（每个技能一个 `skill_<name>` 工具）。

    技能 `name` 必须互不相同：`ToolNode` 以工具名为键，重名会被静默覆盖、调用到错误的
    计划，故此处 fail-fast 拒绝（请改名或去重后再装配）。
    """
    seen: set[str] = set()
    for s in skills:
        if s.name in seen:
            raise ValueError(
                f"技能工具名冲突：{s.name!r} 重复——不同技能不能同名（ToolNode 会静默覆盖）。"
                "请改名或去重后再装配。"
            )
        seen.add(s.name)
    return [_make_skill_tool(s, effectors, governance) for s in skills]
