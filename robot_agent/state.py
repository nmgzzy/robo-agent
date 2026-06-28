"""State schema（设计 §5.1）：工作记忆 `messages` + 滚动摘要 + 注入式只读世界状态。

`messages`（`add_messages` 累积）= 短期工作记忆；`remaining_steps` 由 `create_react_agent`
用于限制单回合步数（二者继承自 `AgentState`，保持与高层装配完全兼容）。

只读世界状态字段（pose / battery / detections）由**外部**（感知源快照，设计 §5.3.2 的
「连续只读量」路径）写入；本 Agent 只读、不负责它们怎么来——经 `get_world_state` 工具
（§5.2）暴露给 LLM。字段都用 `NotRequired`，未注入时缺省即可，不强制每回合都带。
"""

from __future__ import annotations

from typing_extensions import NotRequired, TypedDict

from langgraph.prebuilt.chat_agent_executor import AgentState


class Pose(TypedDict):
    """底盘位姿（平面）。"""

    x: float
    y: float
    theta: float


class Detection(TypedDict):
    """一个被检测到的物体。"""

    label: str
    # 物体在世界系下的位置（如适用）。
    x: NotRequired[float]
    y: NotRequired[float]
    confidence: NotRequired[float]


class RobotState(AgentState):
    """机器人 Agent 状态：短期消息、会话摘要及只读世界状态。"""

    # 中短期记忆：较老的完整消息在高水位时滚动压缩到摘要，随 checkpoint 持久化。
    context_summary: NotRequired[str]
    context_compaction_count: NotRequired[int]
    context_archived_messages: NotRequired[int]
    context_compaction_failures: NotRequired[int]

    pose: NotRequired[Pose | None]
    battery: NotRequired[float | None]  # 电量百分比 0–100
    detections: NotRequired[list[Detection]]
