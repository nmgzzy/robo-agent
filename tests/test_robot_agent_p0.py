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


def test_make_model_real_profile_lazy_imports_anthropic():
    """真实档位需要 langchain-anthropic；未安装时给出可操作的 ImportError，

    且不影响 robot_agent 本身的 import（依赖惰性）。
    """
    from robot_agent import make_model

    # 显式区分「已装 / 未装」两种合法结果，二者都不应让 robot_agent 自身 import 失败。
    try:
        import langchain_anthropic  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="langchain-anthropic"):
            make_model("smart")
    else:
        model = make_model("smart")
        assert isinstance(model, BaseChatModel)
