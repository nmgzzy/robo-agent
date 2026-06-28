"""P0 验收：工程脚手架与依赖（对应 docs/IMPLEMENTATION_PLAN.md §P0）。

覆盖 P0 验收清单：
- 冒烟导入通过（StateGraph / create_react_agent / AsyncSqliteSaver / AsyncSqliteStore）；
- 应用层 robot_agent 可 import；
- LLM 工厂 make_model 的 Mock 路径不接真实 LLM 即可工作；
- 不接真硬件、不接真 LLM 也能 import 成功（远程客户端惰性导入）。
"""

from __future__ import annotations

import os
from unittest.mock import patch

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
    assert hasattr(robot_agent, "load_llm_config_from_env")


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


def test_load_llm_config_from_env(tmp_path):
    from robot_agent.env import load_env
    from robot_agent.llm import LLMConfig, load_llm_config_from_env

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai_compatible",
                "LLM_API_KEY=sk-test",
                "LLM_BASE_URL=http://localhost:8000/v1",
                "LLM_MODEL=qwen-plus",
                "LLM_MODEL_FAST=qwen-turbo",
                "LLM_MODEL_SMART=qwen-max",
                "LLM_MODEL_VISION=qwen-vl-max",
            ]
        ),
        encoding="utf-8",
    )
    with patch.dict(os.environ, {}, clear=True):
        load_env(path=dotenv, override=True)
        cfg = load_llm_config_from_env()
        assert cfg == LLMConfig(
            provider="openai",
            model="qwen-plus",
            model_fast="qwen-turbo",
            model_smart="qwen-max",
            model_vision="qwen-vl-max",
            api_key="sk-test",
            base_url="http://localhost:8000/v1",
        )


def test_load_env_does_not_override_existing_shell_vars(tmp_path):
    from robot_agent.env import load_env

    dotenv = tmp_path / ".env"
    dotenv.write_text("LLM_API_KEY=from-dotenv\n", encoding="utf-8")
    with patch.dict(os.environ, {"LLM_API_KEY": "from-shell"}, clear=True):
        load_env(path=dotenv, override=False)
        assert os.environ["LLM_API_KEY"] == "from-shell"
        load_env(path=dotenv, override=True)
        assert os.environ["LLM_API_KEY"] == "from-dotenv"


def test_load_env_parses_quoted_values(tmp_path):
    from robot_agent.env import load_env

    dotenv = tmp_path / ".env"
    dotenv.write_text('LLM_BASE_URL="http://localhost:8000/v1"\n', encoding="utf-8")
    with patch.dict(os.environ, {}, clear=True):
        load_env(path=dotenv, override=True)
        assert os.environ["LLM_BASE_URL"] == "http://localhost:8000/v1"


def test_find_dotenv_does_not_use_unrelated_working_directory(monkeypatch, tmp_path):
    import robot_agent.env as env_module

    repo_root = tmp_path / "repo"
    working_dir = tmp_path / "other-project"
    repo_root.mkdir()
    working_dir.mkdir()
    (working_dir / ".env").write_text("LLM_API_KEY=wrong\n", encoding="utf-8")
    monkeypatch.setattr(env_module, "_repo_root", lambda: repo_root)
    monkeypatch.chdir(working_dir)

    assert env_module.find_dotenv() is None


def test_load_llm_config_falls_back_to_openai_api_key(monkeypatch):
    from robot_agent.llm import load_llm_config_from_env

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    cfg = load_llm_config_from_env()
    assert cfg.api_key == "sk-openai"


def test_explicit_provider_uses_matching_fallback_api_key(monkeypatch):
    from robot_agent.llm import load_llm_config_from_env

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    cfg = load_llm_config_from_env(provider="anthropic")
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "sk-anthropic"


def test_normalize_provider_rejects_unknown_openai_prefix():
    from robot_agent.llm import normalize_provider

    with pytest.raises(ValueError, match="未知 LLM provider"):
        normalize_provider("openai_typo")


def test_resolve_model_name_priority():
    from robot_agent.llm import LLMConfig, resolve_model_name

    cfg = LLMConfig(
        provider="openai",
        model="shared",
        model_fast="fast-only",
        model_smart="smart-only",
    )
    assert resolve_model_name("fast", cfg) == "fast-only"
    assert resolve_model_name("smart", cfg) == "smart-only"

    cfg_shared = LLMConfig(provider="openai", model="shared")
    assert resolve_model_name("fast", cfg_shared) == "shared"


def test_explicit_model_overrides_profile_model():
    from robot_agent.llm import LLMConfig, merge_llm_config, resolve_model_name

    base = LLMConfig(
        provider="openai",
        model_fast="env-fast",
        model_smart="env-smart",
        model_vision="env-vision",
    )
    merged = merge_llm_config(base, model="explicit-model")

    assert resolve_model_name("fast", merged) == "explicit-model"
    assert resolve_model_name("smart", merged) == "explicit-model"
    assert resolve_model_name("vision", merged) == "explicit-model"


def test_make_model_real_profile_helpful_error_when_client_missing(monkeypatch):
    """真实档位且未装 langchain-openai 时，给出可操作的 ImportError。"""
    import builtins

    from robot_agent import make_model

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_openai" or name.startswith("langchain_openai."):
            raise ImportError("simulated missing langchain-openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="langchain-openai"):
        make_model("smart", provider="openai")


def test_make_model_openai_builds_client_when_present(monkeypatch):
    """OpenAI 兼容档位：注入假 ChatOpenAI，验证 model / api_key / base_url 透传。"""
    import sys
    import types

    from robot_agent import make_model
    from robot_agent.llm import LLMConfig

    captured: dict = {}

    class _FakeChatOpenAI(BaseChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

        def _generate(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        @property
        def _llm_type(self) -> str:
            return "fake-openai"

    fake_mod = types.ModuleType("langchain_openai")
    fake_mod.ChatOpenAI = _FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_mod)

    cfg = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="dummy",
        base_url="http://localhost:11434/v1",
    )
    model = make_model("fast", config=cfg)
    assert isinstance(model, _FakeChatOpenAI)
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "dummy"
    assert captured["base_url"] == "http://localhost:11434/v1"


def test_make_model_anthropic_builds_client_when_present(monkeypatch):
    """Anthropic 档位：注入假 ChatAnthropic，验证按档位映射的模型 id。"""
    import sys
    import types

    from robot_agent import make_model
    from robot_agent.llm import LLMConfig, PROFILE_MODELS

    captured: dict = {}

    class _FakeChatAnthropic(BaseChatModel):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

        def _generate(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        @property
        def _llm_type(self) -> str:
            return "fake-anthropic"

    fake_mod = types.ModuleType("langchain_anthropic")
    fake_mod.ChatAnthropic = _FakeChatAnthropic
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake_mod)

    model = make_model(
        "fast",
        config=LLMConfig(provider="anthropic"),
        api_key="dummy",
    )
    assert isinstance(model, _FakeChatAnthropic)
    assert captured["model"] == PROFILE_MODELS["fast"]
    assert captured["api_key"] == "dummy"
