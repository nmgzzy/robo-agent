"""装配（设计 §4.1）：用 `create_react_agent` 组合大脑 + 工具 + 记忆 + hook。

核心闭环「思考 → 决策 → 行动 → 记忆」的最小可运行体：

- `model`：决策大脑（经 `make_model` 产出，可 Mock，FR-1 分层）。
- `tools`：机器人控制工具（§5.2）+ 记忆回写工具（§6.3）。
- `checkpointer`：短期记忆（`AsyncSqliteSaver`，线程内可恢复，§6.1）。
- `store`：长期记忆（`AsyncSqliteStore`，跨会话，§6.2）。
- `pre_model_hook`：调 LLM 前注入长期记忆 + 裁剪历史（§6.3）。
- `state_schema=RobotState`：messages + 只读世界状态（§5.1）。

需要更强控制（自定义状态通道、显式安全门控节点）时可平滑下沉到 `StateGraph`（设计 §4.2），
能力等价——P1 先用高层入口起步。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from robot_agent.hal import build_effectors
from robot_agent.hal.interfaces import Actuator
from robot_agent.memory import (
    DEFAULT_RECALL_KINDS,
    build_memory_tools,
    make_inject_memory,
)
from robot_agent.governance.policy import GovernancePolicy
from robot_agent.metacog import MetacogPolicy, make_monitor_hook
from robot_agent.safety import SafetyPolicy
from robot_agent.state import RobotState
from robot_agent.tools import build_robot_tools
from robot_agent.vision import (
    VisionSource,
    build_vision_tools,
    make_vision_trust_hook,
)

DEFAULT_ROBOT_ID = "robot-1"


def build_robot_agent(
    *,
    model: BaseChatModel,
    effectors: Mapping[str, Actuator] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    robot_id: str = DEFAULT_ROBOT_ID,
    recall_kinds: Sequence[str] = DEFAULT_RECALL_KINDS,
    safety: SafetyPolicy | None = None,
    metacog: MetacogPolicy | None = None,
    governance: GovernancePolicy | None = None,
    vlm_model: BaseChatModel | None = None,
    vision_source: VisionSource | None = None,
    extra_tools: Sequence[Any] | None = None,
) -> Any:
    """装配并编译机器人 Agent（设计 §4.1）。

    - `effectors` 缺省用 `build_effectors("mock")`（纯内存执行器，离线可跑、可断言 `.log`）。
    - `checkpointer` / `store` 由调用方按生命周期打开后传入（见 `tests/` 用 `async with`）。
    - 装上 `pre_model_hook=inject_memory`：每次调 LLM 前注入长期记忆 + 裁剪历史。
    - `safety` 非 None 时开启危险动作 `interrupt` 门控（设计 §7，需同时配 `checkpointer`）。
      重试 / 超时 / 降级在「决策大脑」一侧用 `reliability.make_resilient(model)` 包装后传入。
    - `metacog` 非 None 时用元认知监控装饰 `pre_model_hook`（循环/预算检测，§8.5）；
      `on_breach="escalate"` 会 `interrupt` 上报，需同时配 `checkpointer`。
    - `vlm_model` 与 `vision_source` 必须同时配置才会挂载 `describe_image`；主模型仅传
      不透明帧引用，图片由 source 直接交给 VLM，不进入 Agent 消息与 checkpoint。

    返回值是已编译的 `create_react_agent`，支持 `ainvoke`（设计 §4.3 时序）。
    """
    if effectors is None:
        effectors = build_effectors("mock")

    if (vlm_model is None) != (vision_source is None):
        raise ValueError("vlm_model 与 vision_source 必须同时配置或同时省略。")

    # interrupt 门控（safety / metacog-escalate）依赖 checkpointer 暂存暂停状态；
    # 缺失时不在装配期放过、留到动作触发才崩，而是 fail-fast 给出可操作报错（设计 §7）。
    if checkpointer is None:
        if safety is not None:
            raise ValueError(
                "safety 门控依赖 interrupt，必须同时配置 checkpointer（否则危险动作触发时才报错）。"
            )
        if metacog is not None and metacog.on_breach == "escalate":
            raise ValueError(
                "metacog on_breach='escalate' 会 interrupt 上报，必须同时配置 checkpointer；"
                "无 checkpointer 时请改用 on_breach='warn'。"
            )

    tools = list(build_robot_tools(effectors, safety=safety, governance=governance))
    # 记忆回写/读取工具依赖 InjectedStore：仅在配置了 store 时才挂载，
    # 否则它们一旦被调用会在 ToolNode 注入阶段直接抛错（无 store 可注入）。
    if store is not None:
        tools += build_memory_tools(robot_id, governance=governance)
    if vlm_model is not None:
        if vision_source is None:
            raise ValueError("vlm_model 与 vision_source 必须同时配置或同时省略。")
        tools += build_vision_tools(vlm_model, vision_source, governance=governance)
    # 动态技能工具（P10）：由 build_skill_tools 生成后传入，运行时扩展能力。
    if extra_tools:
        tools += list(extra_tools)
    pre_model_hook = make_inject_memory(robot_id, kinds=recall_kinds)
    if vlm_model is not None:
        pre_model_hook = make_vision_trust_hook(pre_model_hook)
    # 元认知监控装饰在最外层：先做循环/预算检测，再委托记忆注入。
    if metacog is not None:
        pre_model_hook = make_monitor_hook(pre_model_hook, metacog)

    return create_react_agent(
        model,
        tools=tools,
        checkpointer=checkpointer,
        store=store,
        pre_model_hook=pre_model_hook,
        state_schema=RobotState,
    )
