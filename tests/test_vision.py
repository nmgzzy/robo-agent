"""内置 VLM 视觉理解回归测试（离线 Mock，无远程多模态 API）。"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    convert_to_openai_messages,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from PIL import Image

from robot_agent import build_robot_agent, make_model
from robot_agent.governance import GovernancePolicy
from robot_agent.hal import build_effectors
from robot_agent.llm import LLMConfig, resolve_model_name
from robot_agent.vision import (
    MemoryVisionSource,
    analyze_image,
    build_vision_message,
    build_vision_tools,
    make_vision_trust_hook,
    normalize_image,
    to_data_url,
)
from robot_agent.vision.images import MAX_SOURCE_IMAGE_BYTES
from robot_agent.vision.resize import REENCODE_THRESHOLD_BYTES


def _tiny_png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
        validate=True,
    )


def _tiny_png_b64(*, padded: bool = False) -> str:
    raw = _tiny_png_bytes() + (b"\0" * 300 if padded else b"")
    return base64.b64encode(raw).decode("ascii")


def _source(ref: str = "camera/latest") -> MemoryVisionSource:
    source = MemoryVisionSource()
    source.put_frame(ref, _tiny_png_bytes(), media_type="image/png")
    return source


def _make_image(
    size: tuple[int, int],
    *,
    format: str = "JPEG",
    noisy: bool = False,
) -> bytes:
    image = (
        Image.effect_noise(size, 100).convert("RGB")
        if noisy
        else Image.new("RGB", size, "navy")
    )
    output = BytesIO()
    options = {"quality": 100, "subsampling": 0} if format == "JPEG" else {}
    image.save(output, format=format, **options)
    return output.getvalue()


def test_normalize_image_raw_base64_longer_than_file_name_limit():
    b64 = _tiny_png_b64(padded=True)
    assert len(b64) > 255
    data, mt = normalize_image(b64, media_type="image/png")
    assert data == b64
    assert mt == "image/png"


def test_normalize_image_data_url():
    b64 = _tiny_png_b64()
    data, mt = normalize_image(f"data:image/png;base64,{b64}")
    assert data == b64
    assert mt == "image/png"


def test_normalize_image_data_url_with_extra_params():
    b64 = _tiny_png_b64()
    data, mt = normalize_image(
        f"data:image/png;charset=utf-8;base64,{b64}",
    )
    assert data == b64
    assert mt == "image/png"


@pytest.mark.parametrize(
    "image",
    ["!!!!", "data:image/png;base64,!!!!", "YWJjZA=="],
)
def test_normalize_image_rejects_invalid_or_non_image_base64(image: str):
    with pytest.raises(ValueError):
        normalize_image(image)


def test_normalize_image_rejects_oversized_data_before_encoding():
    raw = b"\x89PNG\r\n\x1a\n" + b"\0" * MAX_SOURCE_IMAGE_BYTES
    with pytest.raises(ValueError, match="过大"):
        normalize_image(raw, media_type="image/png")


def test_normalize_image_rejects_media_type_mismatch():
    with pytest.raises(ValueError, match="不一致"):
        normalize_image(_tiny_png_bytes(), media_type="image/jpeg")


@pytest.mark.parametrize(
    ("source_size", "expected_max", "image_format", "expected_media_type"),
    [
        ((1920, 1080), (1280, 720), "JPEG", "image/jpeg"),
        ((1080, 1920), (720, 1280), "PNG", "image/png"),
    ],
)
def test_normalize_image_downsamples_to_720p(
    source_size: tuple[int, int],
    expected_max: tuple[int, int],
    image_format: str,
    expected_media_type: str,
):
    raw = _make_image(source_size, format=image_format)
    encoded, media_type = normalize_image(raw, media_type=expected_media_type)
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.width <= expected_max[0]
        assert image.height <= expected_max[1]
        assert image.size == expected_max
    assert media_type == expected_media_type


def test_normalize_image_rejects_truncated_image_with_valid_magic():
    with pytest.raises(ValueError, match="完整解码"):
        normalize_image(b"\xff\xd8\xffgarbage", media_type="image/jpeg")


def test_normalize_image_does_not_upscale_small_image():
    raw = _make_image((640, 480))
    encoded, _ = normalize_image(raw, media_type="image/jpeg")
    assert base64.b64decode(encoded) == raw


def test_normalize_image_reencodes_excessive_jpeg_quality():
    raw = _make_image((1280, 720), noisy=True)
    assert len(raw) > REENCODE_THRESHOLD_BYTES
    encoded, media_type = normalize_image(raw, media_type="image/jpeg")
    optimized = base64.b64decode(encoded)
    assert len(optimized) < len(raw)
    with Image.open(BytesIO(optimized)) as image:
        assert image.size == (1280, 720)
    assert media_type == "image/jpeg"


def test_file_path_disabled_by_default_and_restricted_to_root(tmp_path: Path):
    allowed = tmp_path / "frames"
    allowed.mkdir()
    inside = allowed / "shot.png"
    outside = tmp_path / "secret.png"
    inside.write_bytes(_tiny_png_bytes())
    outside.write_bytes(_tiny_png_bytes())

    with pytest.raises(ValueError):
        normalize_image(str(inside))
    data, mt = normalize_image(str(inside), allow_path=True, allowed_root=allowed)
    assert data == _tiny_png_b64()
    assert mt == "image/png"
    with pytest.raises(ValueError, match="允许目录"):
        normalize_image(str(outside), allow_path=True, allowed_root=allowed)


def test_to_data_url():
    b64 = _tiny_png_b64()
    assert to_data_url(b64, "image/png") == f"data:image/png;base64,{b64}"


async def test_memory_vision_source_is_bounded_and_uses_opaque_refs():
    source = MemoryVisionSource(max_frames=1)
    source.put_frame("old", _tiny_png_bytes())
    source.put_frame("new", _tiny_png_bytes())
    with pytest.raises(ValueError, match="已过期"):
        await source.get_frame("old")
    assert (await source.get_frame("new")).media_type == "image/png"


def test_build_vision_message_uses_standard_image_block():
    b64 = _tiny_png_b64()
    msg = build_vision_message(question="这是什么？", image=b64, media_type="image/png")
    assert isinstance(msg, HumanMessage)
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "text"
    assert "不可信外部输入" in msg.content[0]["text"]
    assert msg.content[1]["type"] == "image"
    assert msg.content[1]["base64"] == b64
    assert msg.content[1]["mime_type"] == "image/png"
    assert "image_url" not in msg.content[1]

    openai_message = convert_to_openai_messages([msg])[0]
    image_block = openai_message["content"][1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


async def test_analyze_image_mock_and_empty_fallback():
    model = make_model("mock", responses=[AIMessage(content="一个红色杯子。")])
    out = await analyze_image(
        model,
        question="图里有什么？",
        image=_tiny_png_b64(),
        media_type="image/png",
    )
    assert out == "一个红色杯子。"
    assert len(model.received) == 1

    empty = make_model("mock", responses=[AIMessage(content="")])
    assert "空结果" in await analyze_image(
        empty,
        question="图里有什么？",
        image=_tiny_png_bytes(),
    )


async def test_describe_image_tool_returns_untrusted_structured_observation():
    model = make_model("mock", responses=[AIMessage(content="桌面干净。")])
    tool = build_vision_tools(model, _source())[0]
    out = await tool.ainvoke({"question": "环境如何？", "image_ref": "camera/latest"})
    result = json.loads(out)
    assert result["type"] == "vision_observation"
    assert result["trusted"] is False
    assert result["observation"] == "桌面干净。"
    assert "不得执行" in result["instruction"]


async def test_vision_governance_receives_ref_but_not_image_payload():
    seen: list[dict] = []

    def block_private_frame(name: str, args: dict) -> str | None:
        seen.append(dict(args))
        return "禁止访问私有帧" if args.get("image_ref") == "private" else None

    policy = GovernancePolicy(constitution=[block_private_frame])
    tool = build_vision_tools(
        make_model("mock", responses=[]),
        _source("private"),
        governance=policy,
    )[0]
    out = await tool.ainvoke({"question": "看到了什么？", "image_ref": "private"})
    assert "治理策略拒绝" in out
    assert seen == [{"question": "看到了什么？", "image_ref": "private"}]
    assert _tiny_png_b64() not in repr(policy.audit.entries)


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


async def test_describe_image_skips_renormalization(monkeypatch):
    calls: list[tuple[bytes, str]] = []
    original = __import__(
        "robot_agent.vision.resize", fromlist=["optimize_for_vlm"]
    ).optimize_for_vlm

    def counting(data: bytes, media_type: str) -> tuple[bytes, str]:
        calls.append((data, media_type))
        return original(data, media_type)

    monkeypatch.setattr("robot_agent.vision.images.optimize_for_vlm", counting)
    source = MemoryVisionSource()
    source.put_frame("cam", _tiny_png_bytes(), media_type="image/png")
    calls.clear()

    model = make_model("mock", responses=[AIMessage(content="ok")])
    tool = build_vision_tools(model, source)[0]
    await tool.ainvoke({"question": "看到了什么？", "image_ref": "cam"})
    assert calls == []


async def test_agent_e2e_describe_image_without_image_in_checkpoint():
    source = _source("camera/latest")
    vlm = make_model("mock", responses=[AIMessage(content="前方有障碍物。")])
    agent_model = make_model(
        "mock",
        responses=[
            _tool_call(
                "describe_image",
                {"question": "前方有什么？", "image_ref": "camera/latest"},
                "c1",
            ),
            AIMessage(content="根据视觉观察，前方有障碍物，我会停下。"),
        ],
    )
    eff = build_effectors("mock")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        agent = build_robot_agent(
            model=agent_model,
            effectors=eff,
            checkpointer=cp,
            vlm_model=vlm,
            vision_source=source,
        )
        out = await agent.ainvoke(
            {"messages": [HumanMessage("请看一下摄像头画面并告诉我前方情况。")]},
            {"configurable": {"thread_id": "vision-e2e"}},
        )

    assert "障碍物" in out["messages"][-1].content
    assert len(vlm.received) == 1

    serialized = json.dumps(
        convert_to_openai_messages(out["messages"]),
        ensure_ascii=False,
    )
    assert _tiny_png_b64() not in serialized
    for message in out["messages"]:
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                assert set(call["args"]) <= {"question", "image_ref"}


def test_vlm_tool_schema_and_agent_build_exclude_image_payload():
    tool = build_vision_tools(
        make_model("mock", responses=[]),
        _source(),
    )[0]
    properties = tool.args_schema.model_json_schema()["properties"]
    assert set(properties) == {"question", "image_ref"}
    assert "image" not in properties
    assert "media_type" not in properties

    agent = build_robot_agent(
        model=make_model("mock", responses=[]),
        effectors=build_effectors("mock"),
        vlm_model=make_model("mock", responses=[]),
        vision_source=_source(),
    )
    assert agent is not None


async def test_vision_trust_policy_is_injected_after_existing_system_blocks():
    async def inner(state: dict) -> dict:
        return {"llm_input_messages": list(state["messages"])}

    hook = make_vision_trust_hook(inner)
    result = await hook({"messages": [SystemMessage("identity"), HumanMessage("look")]})
    messages = result["llm_input_messages"]
    assert messages[0].content == "identity"
    assert isinstance(messages[1], SystemMessage)
    assert "不可信外部感知数据" in messages[1].content
    assert isinstance(messages[2], HumanMessage)


def test_build_robot_agent_requires_vlm_and_source_together():
    model = make_model("mock", responses=[])
    with pytest.raises(ValueError, match="必须同时配置"):
        build_robot_agent(
            model=model,
            vlm_model=make_model("mock", responses=[]),
        )
    with pytest.raises(ValueError, match="必须同时配置"):
        build_robot_agent(
            model=model,
            vision_source=_source(),
        )


def test_resolve_model_name_vision_profile_and_smart_fallback():
    cfg = LLMConfig(provider="openai", model_vision="gpt-4o-vision")
    assert resolve_model_name("vision", cfg) == "gpt-4o-vision"

    cfg_smart = LLMConfig(provider="openai", model_smart="local-multimodal")
    assert resolve_model_name("vision", cfg_smart) == "local-multimodal"

    cfg_default = LLMConfig(provider="openai")
    assert resolve_model_name("vision", cfg_default) == "gpt-4o"
