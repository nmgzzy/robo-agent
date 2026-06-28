"""VLM 图片降采样：限制到 720p，并在高质量文件过大时适度重编码。"""

from __future__ import annotations

from io import BytesIO

MAX_LANDSCAPE_SIZE = (1280, 720)
MAX_PORTRAIT_SIZE = (720, 1280)
MAX_SOURCE_PIXELS = 40_000_000
REENCODE_THRESHOLD_BYTES = 1024 * 1024
JPEG_QUALITY = 85
WEBP_QUALITY = 85


def _target_size(width: int, height: int) -> tuple[int, int]:
    bounds = MAX_LANDSCAPE_SIZE if width >= height else MAX_PORTRAIT_SIZE
    scale = min(bounds[0] / width, bounds[1] / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def optimize_for_vlm(data: bytes, media_type: str) -> tuple[bytes, str]:
    """验证图片可解码，并按需缩至 720p 或降低有损编码质量。

    小于尺寸与文件阈值的图片保持原字节不变。超限动图会固定为第一帧，避免一次请求
    隐式携带大量帧。Pillow 惰性导入，未使用 VLM 时不增加应用启动依赖。
    """
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as e:
        raise ImportError(
            "VLM 图片校验/降采样需要 `pillow`；请执行 `uv pip install 'pillow>=11,<13'`。"
        ) from e

    try:
        with Image.open(BytesIO(data)) as opened:
            width, height = opened.size
            if width < 1 or height < 1 or width * height > MAX_SOURCE_PIXELS:
                raise ValueError(
                    f"图片像素数不合法或过大（{width}x{height}，"
                    f"上限 {MAX_SOURCE_PIXELS} 像素）。"
                )
            image = ImageOps.exif_transpose(opened)
            image.load()
            target = _target_size(*image.size)
            needs_resize = target != image.size
            needs_reencode = len(data) > REENCODE_THRESHOLD_BYTES
            if not needs_resize and not needs_reencode:
                return data, media_type

            if needs_resize:
                image.thumbnail(target, Image.Resampling.LANCZOS)

            output = BytesIO()
            if media_type == "image/jpeg":
                image.convert("RGB").save(
                    output,
                    format="JPEG",
                    quality=JPEG_QUALITY,
                    optimize=True,
                    progressive=True,
                )
            elif media_type == "image/webp":
                image.save(
                    output,
                    format="WEBP",
                    quality=WEBP_QUALITY,
                    method=4,
                )
            elif media_type == "image/gif":
                image.convert("RGBA").save(output, format="PNG", optimize=True)
                media_type = "image/png"
            else:
                image.save(output, format="PNG", optimize=True)
                media_type = "image/png"
            return output.getvalue(), media_type
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise ValueError(f"图片无法完整解码：{e}") from e


__all__ = [
    "JPEG_QUALITY",
    "MAX_LANDSCAPE_SIZE",
    "MAX_PORTRAIT_SIZE",
    "MAX_SOURCE_PIXELS",
    "REENCODE_THRESHOLD_BYTES",
    "optimize_for_vlm",
]
