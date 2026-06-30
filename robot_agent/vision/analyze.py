"""VLM 推理核心：把规范化图像 + 问题组装为多模态消息并调用 ChatModel。"""

from __future__ import annotations

import base64
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_image_block

from robot_agent import prompts
from robot_agent.reliability import DEFAULT_FALLBACK_TEXT
from robot_agent.vision.images import normalize_image
from robot_agent.vision.source import VisionFrame


def _content_to_text(content: Any) -> str:
    """把模型回复的 content 规整为纯文本（兼容 Anthropic 内容块列表）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _resolve_image_payload(
    image: str | bytes | VisionFrame,
    *,
    media_type: str | None = None,
) -> tuple[str, str]:
    """返回 `(base64_data, media_type)`；`VisionFrame` 视为已在 source 侧规范化。"""
    if isinstance(image, VisionFrame):
        return base64.b64encode(image.data).decode("ascii"), image.media_type
    return normalize_image(image, media_type=media_type)


def build_vision_message(
    *,
    question: str,
    image: str | bytes | VisionFrame,
    media_type: str | None = None,
) -> HumanMessage:
    """构造含 provider-agnostic 标准图片块的多模态 `HumanMessage`。"""
    b64, mt = _resolve_image_payload(image, media_type=media_type)
    prompt_text = prompts.render("vision_describe", question=question)
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            create_image_block(base64=b64, mime_type=mt),
        ]
    )


async def analyze_image(
    model: BaseChatModel,
    *,
    question: str,
    image: str | bytes | VisionFrame,
    media_type: str | None = None,
) -> str:
    """调用 VLM 理解单张图片并回答问题。

    Agent 路径应经 `describe_image` + `VisionSource` 传入 `VisionFrame`，避免重复校验。
    本函数也接受原始 `str | bytes` 供测试与内部脚本直接调用。
    """
    msg = build_vision_message(question=question, image=image, media_type=media_type)
    out = await model.ainvoke([msg])
    text = _content_to_text(out.content).strip()
    # 仅当 VLM 被 make_resilient 包装、重试耗尽时才会原样返回 DEFAULT_FALLBACK_TEXT；
    # 比较导入同一常量（其文案改动不破坏此链接），把保守兜底转成视觉语义的提示。
    # VLM 正常输出恰好等于该串的概率可忽略。
    if text == DEFAULT_FALLBACK_TEXT:
        return "视觉理解暂时不可用（模型重试耗尽），请稍后再试或改用语义更低的任务。"
    if not text:
        return "视觉理解暂时不可用（模型返回空结果），请稍后重试。"
    return text
