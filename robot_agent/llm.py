"""LLM 工厂：`make_model(profile)` → `BaseChatModel`，支持模型分层与 Mock。

对齐设计 §4（核心闭环的「决策大脑」）与 FR-1（模型分层：高频轻任务用小模型，
复杂规划用大模型）。两条关键纪律：

- **Mock 优先**（NFR-5 / AC-1 / AC-7）：不接真实远程 LLM 即可离线跑通闭环、单测、回归。
- **依赖惰性**：只有请求真实模型档位时才导入 `langchain-anthropic`，核心 `import` 不受其影响。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

# 模型分层（FR-1）：profile → 远程模型 id。高频轻量决策用 haiku，复杂规划用 opus。
PROFILE_MODELS: dict[str, str] = {
    "fast": "claude-haiku-4-5",
    "smart": "claude-opus-4-8",
}
DEFAULT_PROFILE = "smart"
# 触发 Mock 的特殊档位名。
MOCK_PROFILE = "mock"


class MockChatModel(BaseChatModel):
    """脚本化假聊天模型：按预设顺序逐条返回 `AIMessage`，支持 `bind_tools`。

    用于在不接真实远程 LLM 的前提下，确定性地驱动 `create_react_agent` 的工具调用
    往返与整条闭环（对齐 NFR-5 / AC-1 / AC-7）。`bind_tools` 仅记录而不改变回放序列：
    脚本里写好「第 N 次该返回什么」即可。
    """

    responses: list[AIMessage]
    idx: int = 0
    # 记录每次被调用时收到的输入消息序列，便于断言「记忆是否被注入到了 LLM 输入」（AC-3 雏形）。
    received: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # received 是可变列表字段，原地 append 即可（无需绕过 pydantic 校验）。
        self.received.append(list(messages))
        if self.idx >= len(self.responses):
            raise AssertionError(
                f"MockChatModel 响应已耗尽（已回放 {self.idx} 条，无更多脚本）："
                "请检查预设响应数量是否覆盖了全部 LLM 调用次数。"
            )
        msg = self.responses[self.idx]
        # BaseChatModel 是 pydantic 模型，常规赋值会被校验拦截，用 object.__setattr__ 自增游标。
        object.__setattr__(self, "idx", self.idx + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 机器人闭环是 async-only（async 工具 + ainvoke）。显式提供原生异步实现，
        # 避免回退到 BaseChatModel 默认的「线程池跑 _generate」路径（事件循环关闭时可能挂起）。
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "MockChatModel":
        # 脚本化模型忽略实际工具绑定，直接回放预设响应。
        return self


def make_model(
    profile: str = DEFAULT_PROFILE,
    *,
    responses: Sequence[AIMessage] | None = None,
    **overrides: Any,
) -> BaseChatModel:
    """构建一个 `ChatModel`。

    - `profile="mock"` 或传入 `responses`：返回脚本化 `MockChatModel`（离线 / 测试 / 回归）。
    - `profile in {"fast", "smart"}`：惰性构建 `ChatAnthropic`（需安装 `langchain-anthropic`，
      并配置 `ANTHROPIC_API_KEY` / `base_url` 等）。`overrides` 透传给 `ChatAnthropic`。

    设计 §4.1 蓝图里的 `ChatAnthropic(model=...)` 即由本工厂统一产出，便于按档位切换与 Mock。
    """
    if profile == MOCK_PROFILE or responses is not None:
        return MockChatModel(responses=list(responses or []))

    try:
        model_name = PROFILE_MODELS[profile]
    except KeyError:
        valid = sorted(PROFILE_MODELS) + [MOCK_PROFILE]
        raise ValueError(f"未知 profile={profile!r}，可选：{valid}") from None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:  # 离线 / 未装客户端：给出可操作的提示而非裸 ImportError。
        raise ImportError(
            f"profile={profile!r} 需要远程 LLM 客户端 `langchain-anthropic`（未安装）；"
            "离线 / 测试请改用 make_model('mock', responses=[...])。"
        ) from e

    return ChatAnthropic(model=model_name, **overrides)
