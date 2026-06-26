"""P0 验收：工程脚手架与依赖（对应 docs/IMPLEMENTATION_PLAN.md §P0）。

覆盖 P0 验收清单：
- 冒烟导入通过（StateGraph / create_react_agent / AsyncSqliteSaver / AsyncSqliteStore）；
- 应用层 robot_agent 可 import；
- LLM 工厂 make_model 的 Mock 路径不接真实 LLM 即可工作；
- 不接真硬件、不接真 LLM 也能 import 成功（langchain-anthropic 惰性导入）。
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage


def test_core_smoke_imports():
    """底座关键符号可导入（瘦身不变量的子集，P0 冒烟）。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401
    from langgraph.graph import StateGraph  # noqa: F401
    from langgraph.prebuilt import create_react_agent  # noqa: F401
    from langgraph.store.sqlite.aio import AsyncSqliteStore  # noqa: F401


def test_robot_agent_importable():
    """应用层空骨架可 import，并暴露 LLM 工厂。"""
    import robot_agent

    assert hasattr(robot_agent, "make_model")
    assert hasattr(robot_agent, "MockChatModel")


def test_make_model_mock_profile_returns_chat_model():
    from robot_agent import make_model

    model = make_model("mock", responses=[AIMessage(content="hi")])
    assert isinstance(model, BaseChatModel)


def test_make_model_responses_implies_mock_without_profile():
    """只要传了 responses，即便用默认 profile 也走 Mock（离线友好）。"""
    from robot_agent import MockChatModel, make_model

    model = make_model(responses=[AIMessage(content="ok")])
    assert isinstance(model, MockChatModel)


def test_mock_model_replays_scripted_responses():
    from robot_agent import make_model

    model = make_model("mock", responses=[AIMessage(content="第一")])
    out = model.invoke([HumanMessage("问")])
    assert out.content == "第一"


def test_mock_model_exhausted_raises():
    """脚本回放次数不足时给出明确断言错误，便于定位测试脚本漏写。"""
    from robot_agent import make_model

    model = make_model("mock", responses=[])
    with pytest.raises(AssertionError, match="响应已耗尽"):
        model.invoke([HumanMessage("问")])


def test_make_model_unknown_profile_raises():
    from robot_agent import make_model

    with pytest.raises(ValueError, match="未知 profile"):
        make_model("nonexistent-profile")


def test_make_model_real_profile_helpful_error_when_client_missing(monkeypatch):
    """真实档位且未装 langchain-anthropic 时，给出可操作的 ImportError。

    通过隐藏该模块来**确定性**复现「未安装」，不依赖真实环境是否装了客户端，
    也保证 robot_agent 自身 import 不受影响（依赖惰性）。
    """
    import builtins

    from robot_agent import make_model

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_anthropic" or name.startswith("langchain_anthropic."):
            raise ImportError("simulated missing langchain-anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="langchain-anthropic"):
        make_model("smart")


def test_make_model_real_profile_builds_client_when_present(monkeypatch):
    """真实档位可用时，按档位映射的模型 id 构建客户端并透传 overrides。

    注入一个假的 ChatAnthropic（不触真实凭证 / 网络），避免在装了客户端但无
    ANTHROPIC_API_KEY 的环境里破坏「离线可测」承诺。
    """
    import sys
    import types

    from robot_agent import make_model
    from robot_agent.llm import PROFILE_MODELS

    captured: dict = {}

    class _FakeChatAnthropic(BaseChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

        def _generate(self, *a, **k):  # pragma: no cover - 不会被调用
            raise NotImplementedError

        @property
        def _llm_type(self) -> str:
            return "fake-anthropic"

    fake_mod = types.ModuleType("langchain_anthropic")
    fake_mod.ChatAnthropic = _FakeChatAnthropic
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake_mod)

    model = make_model("fast", api_key="dummy")
    assert isinstance(model, _FakeChatAnthropic)
    assert captured["model"] == PROFILE_MODELS["fast"]
    assert captured["api_key"] == "dummy"
