"""提示词集中管理（robot_agent/prompts）的回归测试。

覆盖：registry 加载、每个模板可渲染、占位符↔params 一致性、identity_default 深拷贝、
未知 id / 缺参数 / 占位符不一致的 fail-fast。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_agent import prompts
from robot_agent.prompts import PromptError, render

_REGISTRY = json.loads(
    (Path(prompts.__file__).parent / "registry.json").read_text(encoding="utf-8")
)
_ENTRIES = _REGISTRY["prompts"]

# 每个提示词的样例参数（占位符全覆盖），用于验证可渲染。
_SAMPLE = {
    "identity": {
        "name": "小巡",
        "persona": "服务机器人",
        "values": "安全第一",
        "good_at": "导航",
        "bad_at": "精细操作",
    },
    "memory_header": {"items": "- [facts] x: 1"},
    "goal_turn": {"intent": "巡视", "steps": "1) A；2) B"},
    "metacog_warn": {"reason": "检测到循环"},
    "plan": {"intent": "把东西搬过去"},
    "distill": {"episodes": "回合1：…"},
    "conflict": {"memories": "1: [键=a] v"},
    "vision_describe": {"question": "图里有什么？"},
    "vision_trust_policy": {},
}


def test_all_registered_prompts_render():
    """registry 里每条提示词都能用样例参数渲染出非空文本。"""
    for pid in _ENTRIES:
        assert pid in _SAMPLE, f"测试样例缺少 {pid}，请补充 _SAMPLE"
        text = render(pid, **_SAMPLE[pid])
        assert text and isinstance(text, str)


def test_placeholders_match_declared_params():
    """每个模板内的 {占位符} 与 registry 声明的 params 严格一致（loader 已校验，这里冗余兜底）。"""
    import re

    base = Path(prompts.__file__).parent
    for pid, meta in _ENTRIES.items():
        template = (base / meta["file"]).read_text(encoding="utf-8")
        found = set(re.findall(r"\{(\w+)\}", template))
        assert found == set(meta.get("params", [])), pid


def test_identity_default_contains_key_fields():
    data = prompts.identity_default()
    assert data["name"] == "小巡"
    assert "capabilities" in data and "good_at" in data["capabilities"]


def test_identity_default_is_deep_copy():
    """改动返回值不得污染下一次取到的默认身份。"""
    d1 = prompts.identity_default()
    d1["values"].append("篡改")
    d1["name"] = "X"
    d2 = prompts.identity_default()
    assert "篡改" not in d2["values"]
    assert d2["name"] == "小巡"


def test_render_unknown_id_raises():
    with pytest.raises(PromptError):
        render("no_such_prompt")


def test_render_missing_param_raises():
    with pytest.raises(PromptError):
        render("metacog_warn")  # 缺 reason


def test_render_substrings_preserved():
    """关键文案保留，确保下游 substring 断言（身份/计划/告警）不回归。"""
    assert "我是谁" in render("identity", **_SAMPLE["identity"])
    assert "计划步骤" in render("goal_turn", **_SAMPLE["goal_turn"])
    assert "元认知告警" in render("metacog_warn", **_SAMPLE["metacog_warn"])
    assert "已知的长期记忆" in render("memory_header", **_SAMPLE["memory_header"])
