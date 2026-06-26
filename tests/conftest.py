"""共享测试夹具：瘦身后「嵌入式机器人 Agent 底座」的验收/回归测试。

这些测试不依赖任何远程 LLM 或外部服务，全部在本地（含内存/临时文件 SQLite）运行，
以匹配嵌入式、依赖少、可离线自测的目标。
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# create_react_agent 已被上游标注 deprecated（迁到 langchain.agents），
# 但瘦身版刻意不引入更重的 langchain 包，继续用 langgraph.prebuilt 版本。
# 测试里静音该弃用告警，避免噪声。
warnings.filterwarnings("ignore", message=".*create_react_agent.*")


class FakeToolCallingModel(BaseChatModel):
    """可脚本化的假聊天模型：支持 bind_tools，按预设顺序逐条返回 AIMessage。

    用于在不接真实远程 LLM 的前提下，确定性地驱动 create_react_agent 的工具调用往返。
    """

    responses: list[AIMessage]
    idx: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self.responses[self.idx]
        object.__setattr__(self, "idx", self.idx + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingModel":
        # 脚本化模型忽略实际工具绑定，直接回放预设响应。
        return self


@pytest.fixture
def make_model():
    """返回一个工厂：传入 AIMessage 序列，得到脚本化假模型。"""

    def _make(responses: list[AIMessage]) -> FakeToolCallingModel:
        return FakeToolCallingModel(responses=responses)

    return _make
