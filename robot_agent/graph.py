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
) -> Any:
    """装配并编译机器人 Agent（设计 §4.1）。

    - `effectors` 缺省用 `build_effectors("mock")`（纯内存执行器，离线可跑、可断言 `.log`）。
    - `checkpointer` / `store` 由调用方按生命周期打开后传入（见 `tests/` 用 `async with`）。
    - 装上 `pre_model_hook=inject_memory`：每次调 LLM 前注入长期记忆 + 裁剪历史。

    返回值是已编译的 `create_react_agent`，支持 `ainvoke`（设计 §4.3 时序）。
    """
    if effectors is None:
        effectors = build_effectors("mock")

    tools = [*build_robot_tools(effectors), *build_memory_tools(robot_id)]
    pre_model_hook = make_inject_memory(robot_id, kinds=recall_kinds)

    return create_react_agent(
        model,
        tools=tools,
        checkpointer=checkpointer,
        store=store,
        pre_model_hook=pre_model_hook,
        state_schema=RobotState,
    )
