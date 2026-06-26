"""共享测试夹具：瘦身后「嵌入式机器人 Agent 底座」的验收/回归测试。

这些测试不依赖任何远程 LLM 或外部服务，全部在本地（含内存/临时文件 SQLite）运行，
以匹配嵌入式、依赖少、可离线自测的目标。
"""

from __future__ import annotations

import warnings

import pytest
from langchain_core.messages import AIMessage

# 脚本化假模型已上提到应用层 LLM 工厂，测试与生产共用同一实现（避免重复）。
from robot_agent.llm import MockChatModel

# create_react_agent 已被上游标注 deprecated（迁到 langchain.agents），
# 但瘦身版刻意不引入更重的 langchain 包，继续用 langgraph.prebuilt 版本。
# 测试里静音该弃用告警，避免噪声。
warnings.filterwarnings("ignore", message=".*create_react_agent.*")


@pytest.fixture
def make_model():
    """返回一个工厂：传入 AIMessage 序列，得到脚本化假模型（robot_agent.llm.MockChatModel）。"""

    def _make(responses: list[AIMessage]) -> MockChatModel:
        return MockChatModel(responses=responses)

    return _make
