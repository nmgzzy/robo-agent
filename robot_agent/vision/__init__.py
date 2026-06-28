"""内置 VLM 视觉理解（辅助大脑 + 工具）。

外部感知/HAL 通过 `VisionSource` 按不透明引用提供图片；主 Agent 不接触原始 payload。
本包负责严格校验并调用多模态 LLM，远程客户端仍走 `make_model("vision")` 惰性导入。
"""

from __future__ import annotations

from robot_agent.vision.analyze import analyze_image, build_vision_message
from robot_agent.vision.images import normalize_image, to_data_url
from robot_agent.vision.source import (
    MemoryVisionSource,
    VisionFrame,
    VisionSource,
)
from robot_agent.vision.tools import build_vision_tools
from robot_agent.vision.trust import make_vision_trust_hook

__all__ = [
    "analyze_image",
    "build_vision_message",
    "build_vision_tools",
    "MemoryVisionSource",
    "make_vision_trust_hook",
    "normalize_image",
    "to_data_url",
    "VisionFrame",
    "VisionSource",
]
