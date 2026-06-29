#!/usr/bin/env python3
"""真实 VLM 兼容性探针：固定图片经 VisionSource 和 describe_image 完整链路识别。

本脚本会访问 `.env` 或命令行指定的远程视觉模型并产生实际费用，不属于
`make test` 的离线回归集合。默认只发送一张 320x200 PNG，并限制输出、重试和超时。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_agent import make_model, prompts  # noqa: E402
from robot_agent.llm import (  # noqa: E402
    load_llm_config_from_env,
    merge_llm_config,
    resolve_model_name,
)
from robot_agent.vision import MemoryVisionSource, build_vision_tools  # noqa: E402


@dataclass
class CheckResult:
    """一项 VLM 兼容性检查结果。"""

    name: str
    passed: bool
    details: dict[str, Any]


class RecordingModel:
    """透明记录底层模型响应，以便汇总 provider token usage。"""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.responses: list[AIMessage] = []

    async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        response = await self.delegate.ainvoke(*args, **kwargs)
        self.responses.append(response)
        return response


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用真实 `.env` vision 模型检查图片输入、VisionSource、结构化视觉观察和识别准确性。"
        )
    )
    parser.add_argument("--provider", help="覆盖 `.env` 的 LLM_PROVIDER")
    parser.add_argument("--model", help="覆盖 `.env` 的 LLM_MODEL_VISION")
    parser.add_argument("--base-url", help="覆盖 `.env` 的 LLM_BASE_URL")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="客户端请求重试次数；默认 0，避免放大费用和等待时间",
    )
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--case-timeout", type=float, default=60.0)
    parser.add_argument(
        "--save-image",
        type=Path,
        help="可选：保存探针实际提交的 PNG，便于人工核对",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="把结构化报告写入指定 JSON 文件",
    )
    return parser


def _make_probe_image() -> bytes:
    """生成固定测试图：白底、左侧红圆、右侧蓝圆。"""
    image = Image.new("RGB", (320, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 40, 130, 140), fill="red")
    draw.ellipse((190, 40, 290, 140), fill="blue")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _config(args: argparse.Namespace) -> Any:
    return merge_llm_config(
        load_llm_config_from_env(provider=args.provider),
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )


def _config_summary(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args)
    return {
        "provider": config.provider,
        "profile": "vision",
        "model": resolve_model_name("vision", config),
        "base_url": config.base_url or "<provider-default>",
        "api_key_configured": bool(config.api_key),
    }


def _usage(responses: list[AIMessage]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for response in responses:
        usage = response.usage_metadata or {}
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def _accuracy_checks(observation: str) -> dict[str, bool]:
    text = observation.lower()
    return {
        "white_background": "白" in text or "white" in text,
        "two_circles": any(marker in text for marker in ("2", "两个", "二个")),
        "red_circle": "红" in text or "red" in text,
        "blue_circle": "蓝" in text or "blue" in text,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    image = _make_probe_image()
    image_ref = "live-vlm-probe/two-circles"
    source = MemoryVisionSource()
    source.put_frame(image_ref, image, media_type="image/png")
    frame = await source.get_frame(image_ref)

    config = _config(args)
    delegate = make_model(
        "vision",
        config=config,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        temperature=0,
        timeout=args.request_timeout,
    )
    model = RecordingModel(delegate)
    describe_image = build_vision_tools(model, source)[0]
    tool_properties = describe_image.args_schema.model_json_schema()["properties"]

    started = time.monotonic()
    error: str | None = None
    result: dict[str, Any] | None = None
    try:
        raw = await asyncio.wait_for(
            describe_image.ainvoke(
                {
                    "question": prompts.render("live_probe_vision"),
                    "image_ref": image_ref,
                }
            ),
            timeout=args.case_timeout,
        )
        result = json.loads(raw)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    duration = time.monotonic() - started

    checks: list[CheckResult] = []
    if result is None:
        checks.append(CheckResult("request", False, {"error": error}))
        observation = ""
    else:
        observation = str(result.get("observation", ""))
        structure_details = {
            "type": result.get("type"),
            "trusted": result.get("trusted"),
            "image_ref": result.get("image_ref"),
        }
        checks.append(
            CheckResult(
                "structured-observation",
                structure_details
                == {
                    "type": "vision_observation",
                    "trusted": False,
                    "image_ref": image_ref,
                },
                structure_details,
            )
        )
        accuracy = _accuracy_checks(observation)
        checks.append(CheckResult("visual-accuracy", all(accuracy.values()), accuracy))

    schema_fields = sorted(tool_properties)
    checks.append(
        CheckResult(
            "opaque-image-ref",
            schema_fields == ["image_ref", "question"],
            {
                "tool_schema_fields": schema_fields,
                "image_payload_exposed": "image" in schema_fields,
            },
        )
    )

    return {
        "duration_seconds": round(duration, 3),
        "image": {
            "width": 320,
            "height": 200,
            "source_bytes": len(image),
            "submitted_bytes": len(frame.data),
            "media_type": frame.media_type,
            "sha256": hashlib.sha256(frame.data).hexdigest(),
        },
        "observation": observation,
        "observed_model_responses": len(model.responses),
        "token_usage": _usage(model.responses),
        "checks": checks,
    }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.request_timeout <= 0 or args.case_timeout <= 0:
        parser.error("max-tokens、request-timeout、case-timeout 必须大于 0")
    if args.max_retries < 0:
        parser.error("max-retries 不能小于 0")

    config = _config_summary(args)
    print("真实 VLM 兼容性探针（会访问远程 API 并产生费用）", flush=True)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    if args.save_image is not None:
        args.save_image.parent.mkdir(parents=True, exist_ok=True)
        args.save_image.write_bytes(_make_probe_image())
        print(f"Probe image: {args.save_image}", flush=True)

    run = asyncio.run(_run(args))
    checks: list[CheckResult] = run.pop("checks")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}", flush=True)
        for key, value in check.details.items():
            print(f"  {key}: {value}", flush=True)
    print(f"OBSERVATION: {run['observation']}", flush=True)

    passed = all(check.passed for check in checks)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "config": config,
        "limits": {
            "max_tokens_per_model_call": args.max_tokens,
            "max_retries": args.max_retries,
            "request_timeout_seconds": args.request_timeout,
            "case_timeout_seconds": args.case_timeout,
        },
        **run,
        "checks": [asdict(check) for check in checks],
    }
    print("\nSUMMARY")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json_report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
