"""视觉帧来源：模型仅使用不透明引用，原始图片不进入工具参数或 checkpoint。"""

from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from robot_agent.vision.images import normalize_image


@dataclass(frozen=True)
class VisionFrame:
    """单帧图片；`MemoryVisionSource.put_frame` 写入前已完成校验与 720p 优化。"""

    data: bytes
    media_type: str


@runtime_checkable
class VisionSource(Protocol):
    """HAL/插件侧实现的按不透明引用取帧接口。"""

    async def get_frame(self, image_ref: str) -> VisionFrame: ...


class MemoryVisionSource:
    """有界内存帧源，供 Mock、测试和轻量部署使用。"""

    def __init__(
        self,
        *,
        max_frames: int = 8,
        max_total_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        if max_frames < 1:
            raise ValueError("max_frames 必须大于 0。")
        if max_total_bytes < 1:
            raise ValueError("max_total_bytes 必须大于 0。")
        self.max_frames = max_frames
        self.max_total_bytes = max_total_bytes
        self._total_bytes = 0
        self._frames: OrderedDict[str, VisionFrame] = OrderedDict()

    def put_frame(
        self,
        image_ref: str,
        image: str | bytes,
        *,
        media_type: str | None = None,
    ) -> None:
        ref = _validate_ref(image_ref)
        data, detected = normalize_image(image, media_type=media_type)
        raw = base64.b64decode(data, validate=True)
        if len(raw) > self.max_total_bytes:
            raise ValueError(
                f"单帧大小 {len(raw)} 超过帧源总内存上限 {self.max_total_bytes}。"
            )
        previous = self._frames.pop(ref, None)
        if previous is not None:
            self._total_bytes -= len(previous.data)
        self._frames[ref] = VisionFrame(data=raw, media_type=detected)
        self._total_bytes += len(raw)
        self._frames.move_to_end(ref)
        while (
            len(self._frames) > self.max_frames
            or self._total_bytes > self.max_total_bytes
        ):
            _, evicted = self._frames.popitem(last=False)
            self._total_bytes -= len(evicted.data)

    async def get_frame(self, image_ref: str) -> VisionFrame:
        ref = _validate_ref(image_ref)
        try:
            frame = self._frames[ref]
        except KeyError as e:
            raise ValueError(f"未知或已过期的 image_ref={ref!r}。") from e
        self._frames.move_to_end(ref)
        return frame


def _validate_ref(image_ref: str) -> str:
    ref = image_ref.strip()
    if not ref or len(ref) > 128:
        raise ValueError("image_ref 长度必须在 1..128 之间。")
    if any(ord(char) < 32 for char in ref):
        raise ValueError("image_ref 不能包含控制字符。")
    return ref


__all__ = [
    "MemoryVisionSource",
    "VisionFrame",
    "VisionSource",
]
