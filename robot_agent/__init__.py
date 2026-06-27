"""机器人 Agent 应用层（数字个体策略层）。

本包是「嵌入式机器人 Agent 底座」的**应用层**，建立在裁剪后的四个核心库
（checkpoint / checkpoint-sqlite / langgraph / prebuilt）之上，只经接口/hook 依赖它们。

设计与分阶段计划见：
- `docs/ROBOT_AGENT_DESIGN.md`（需求 + 技术总纲）
- `docs/IMPLEMENTATION_PLAN.md`（P0–P10 分阶段实现计划）

目录布局（随阶段填充，对齐设计 §0.3 与 docs/IMPLEMENTATION_PLAN.md）：

    robot_agent/
    ├── llm.py          # P0：LLM 工厂 make_model(profile)（含 Mock）
    ├── state.py        # P1：State schema（messages + 只读世界状态）
    ├── hal/            # P1：SensorSource / Actuator 接口 + Mock 实现
    ├── tools.py        # P1：机器人控制工具（Actuator → @tool，可选 safety 门控）
    ├── memory.py       # P1/P3：长期记忆 + 身份注入 hook + namespace 约定
    ├── graph.py        # P1：装配 create_react_agent（build_robot_agent）
    ├── reliability.py  # P2：ResilientChatModel 重试/超时/降级 + checkpoint 清理
    ├── safety.py       # P2：危险动作 interrupt 门控（硬拒绝/需确认/放行）
    ├── identity.py     # P3：身份/自我模型（稳定 system 锚点）
    ├── driver/         # P4：自主引擎（收件箱 + 空闲策略 + 常驻循环）
    ├── goals/          # P5：目标系统（模型 + 持久化 + 仲裁 + 规划）
    └── reflect/        # P6：复盘闭环（回合记录 + episodic→semantic 蒸馏）

**依赖纪律**（对齐 `docs/SLIMMING_NOTES.md`）：硬件 SDK / ROS / OpenCV / 控制算法
只允许出现在 `hal/plugins/<impl>` 实现包内，不进核心四库依赖树；远程 LLM 客户端
（`langchain-anthropic`）也只在请求真实模型时惰性导入，离线/测试用 Mock。
"""

from __future__ import annotations

from robot_agent.driver import (
    Driver,
    Event,
    PriorityInbox,
    PromptIdlePolicy,
    StandbyPolicy,
    user_message,
)
from robot_agent.goals import (
    Goal,
    GoalDrivenIdlePolicy,
    GoalStore,
    arbitrate,
    plan_goal,
)
from robot_agent.graph import build_robot_agent
from robot_agent.hal import build_effectors
from robot_agent.identity import (
    DEFAULT_IDENTITY,
    ensure_default_identity,
    get_identity,
    set_identity,
)
from robot_agent.llm import DEFAULT_PROFILE, MockChatModel, make_model
from robot_agent.reflect import (
    Episode,
    make_reflect_hook,
    reflect_and_distill,
)
from robot_agent.reliability import (
    ResilientChatModel,
    cleanup_threads,
    make_resilient,
)
from robot_agent.safety import SafetyPolicy
from robot_agent.state import RobotState

__all__ = [
    "DEFAULT_IDENTITY",
    "DEFAULT_PROFILE",
    "Driver",
    "Episode",
    "Event",
    "Goal",
    "GoalDrivenIdlePolicy",
    "GoalStore",
    "MockChatModel",
    "PriorityInbox",
    "PromptIdlePolicy",
    "ResilientChatModel",
    "RobotState",
    "SafetyPolicy",
    "StandbyPolicy",
    "arbitrate",
    "build_effectors",
    "build_robot_agent",
    "cleanup_threads",
    "ensure_default_identity",
    "get_identity",
    "make_model",
    "make_reflect_hook",
    "make_resilient",
    "plan_goal",
    "reflect_and_distill",
    "set_identity",
    "user_message",
]
