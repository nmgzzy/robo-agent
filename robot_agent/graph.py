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

    返回值是已编译的 `create_react_agent`，支持 `ainvoke`（设计 §4.3 时序）。
    """
    if effectors is None:
        effectors = build_effectors("mock")

    tools = list(build_robot_tools(effectors, safety=safety, governance=governance))
    # 记忆回写/读取工具依赖 InjectedStore：仅在配置了 store 时才挂载，
    # 否则它们一旦被调用会在 ToolNode 注入阶段直接抛错（无 store 可注入）。
    if store is not None:
        tools += build_memory_tools(robot_id)
    # 动态技能工具（P10）：由 build_skill_tools 生成后传入，运行时扩展能力。
    if extra_tools:
        tools += list(extra_tools)
    pre_model_hook = make_inject_memory(robot_id, kinds=recall_kinds)
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
