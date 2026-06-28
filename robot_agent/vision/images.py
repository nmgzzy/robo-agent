"""受限图像输入规范化：严格校验大小、编码、格式与可选文件边界。"""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path

from robot_agent.vision.resize import optimize_for_vlm

_DATA_URL_PREFIX = "data:"
_ASCII_WHITESPACE_RE = re.compile(r"[\t\n\r ]+")

SUPPORTED_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)
MAX_SOURCE_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_BASE64_CHARS = ((MAX_SOURCE_IMAGE_BYTES + 2) // 3) * 4
MAX_BASE64_INPUT_CHARS = MAX_BASE64_CHARS + 8192


def _sniff_media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _validate_media_type(media_type: str | None) -> str | None:
    if media_type is None:
        return None
    normalized = media_type.strip().lower()
    if normalized not in SUPPORTED_MEDIA_TYPES:
        raise ValueError(
            f"不支持的图片 media_type={media_type!r}；可选：{sorted(SUPPORTED_MEDIA_TYPES)}。"
        )
    return normalized


def _parse_data_url(text: str) -> tuple[str, str]:
    """解析 `data:[<mediatype>][;param...];base64,<payload>`。"""
    if not text.lower().startswith(_DATA_URL_PREFIX):
        raise ValueError("不是 data URL。")
    try:
        header, payload = text.split(",", 1)
    except ValueError as e:
        raise ValueError("data URL 格式非法。") from e
    if ";base64" not in header.lower():
        raise ValueError("仅支持 base64 编码的 data URL。")
    media_part = header[len(_DATA_URL_PREFIX) :].split(";base64", 1)[0].strip()
    media_type_raw = media_part.split(";", 1)[0].strip()
    if not media_type_raw:
        raise ValueError("data URL 缺少 media type。")
    declared = _validate_media_type(media_type_raw)
    return declared, payload


def _validate_bytes(data: bytes, declared_media_type: str | None) -> tuple[bytes, str]:
    if not data:
        raise ValueError("图片数据不能为空。")
    if len(data) > MAX_SOURCE_IMAGE_BYTES:
        raise ValueError(
            f"原始图片过大（{len(data)} 字节，上限 {MAX_SOURCE_IMAGE_BYTES}）。"
        )
    detected = _sniff_media_type(data)
    if detected is None:
        raise ValueError("图片内容不是受支持的 JPEG / PNG / GIF / WebP。")
    declared = _validate_media_type(declared_media_type)
    if declared is not None and declared != detected:
        raise ValueError(f"图片声明类型 {declared!r} 与实际内容 {detected!r} 不一致。")
    optimized, optimized_type = optimize_for_vlm(data, detected)
    if len(optimized) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"降采样后图片仍过大（{len(optimized)} 字节，上限 {MAX_IMAGE_BYTES}）。"
        )
    return optimized, optimized_type


def _decode_base64(data: str, media_type: str | None) -> tuple[bytes, str]:
    if len(data) > MAX_BASE64_INPUT_CHARS:
        raise ValueError(
            f"base64 图片输入过大（输入长度 {len(data)}，上限 {MAX_BASE64_INPUT_CHARS}）。"
        )
    compact = _ASCII_WHITESPACE_RE.sub("", data)
    if not compact:
        raise ValueError("base64 图片数据不能为空。")
    if len(compact) > MAX_BASE64_CHARS:
        raise ValueError(
            f"base64 图片数据过大（编码长度 {len(compact)}，上限 {MAX_BASE64_CHARS}）。"
        )
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("图片不是有效 base64。") from e
    return _validate_bytes(decoded, media_type)


def _read_restricted_path(path_text: str, allowed_root: Path) -> bytes:
    root = allowed_root.resolve(strict=True)
    path = Path(path_text).expanduser().resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as e:
        raise ValueError(f"图片路径必须位于允许目录 {root} 内。") from e
    if not path.is_file():
        raise ValueError("图片路径不是普通文件。")
    size = path.stat().st_size
    if size > MAX_SOURCE_IMAGE_BYTES:
        raise ValueError(
            f"图片文件过大（{size} 字节，上限 {MAX_SOURCE_IMAGE_BYTES}）。"
        )
    try:
        return path.read_bytes()
    except OSError as e:
        raise ValueError(f"图片文件不可读：{e}") from e


def normalize_image(
    image: str | bytes,
    *,
    media_type: str | None = None,
    allow_path: bool = False,
    allowed_root: Path | None = None,
) -> tuple[str, str]:
    """规范为严格校验后的 `(base64_data, media_type)`。

    字符串仅按 data URL 或裸 base64 处理。文件路径默认禁用；确需读取时必须同时传入
    `allow_path=True` 和 `allowed_root`，且真实路径必须位于该目录内。
    """
    if isinstance(image, bytes):
        decoded, detected = _validate_bytes(image, media_type)
    else:
        text = image.strip()
        if not text:
            raise ValueError("image 不能为空。")
        if text.lower().startswith(_DATA_URL_PREFIX):
            declared, payload = _parse_data_url(text)
            if media_type is not None and _validate_media_type(media_type) != declared:
                raise ValueError("参数 media_type 与 data URL 中声明的类型不一致。")
            decoded, detected = _decode_base64(payload, declared)
        elif allow_path:
            if allowed_root is None:
                raise ValueError("启用文件路径输入时必须显式配置 allowed_root。")
            try:
                raw = _read_restricted_path(text, allowed_root)
            except (OSError, RuntimeError) as e:
                raise ValueError(f"图片路径无效：{e}") from e
            decoded, detected = _validate_bytes(raw, media_type)
        else:
            decoded, detected = _decode_base64(text, media_type)
    return base64.b64encode(decoded).decode("ascii"), detected


def to_data_url(base64_data: str, media_type: str) -> str:
    """组装兼容 OpenAI 的 data URL；调用方应先经过 `normalize_image`。"""
    return f"data:{_validate_media_type(media_type)};base64,{base64_data}"
