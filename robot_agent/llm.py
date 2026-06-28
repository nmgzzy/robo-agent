"""LLM 工厂：`make_model(profile)` → `BaseChatModel`，支持模型分层与 Mock。

对齐设计 §4（核心闭环的「决策大脑」）与 FR-1（模型分层：高频轻任务用小模型，
复杂规划用大模型）。两条关键纪律：

- **Mock 优先**（NFR-5 / AC-1 / AC-7）：不接真实远程 LLM 即可离线跑通闭环、单测、回归。
- **依赖惰性**：只有请求真实模型档位时才导入对应客户端（`langchain-openai` /
  `langchain-anthropic`），核心 `import` 不受其影响。

真实模型可通过仓库根目录 `.env`（见 `.env.example`）、环境变量，或
`make_model(..., provider=..., model=..., api_key=..., base_url=...)` 显式参数配置
（显式参数优先于环境变量；shell 已导出变量优先于 `.env`）：

| 变量 | 说明 |
|------|------|
| `LLM_PROVIDER` | `openai` / `openai_compatible` / `anthropic`（默认 `openai`） |
| `LLM_API_KEY` | API 密钥；未设时回退 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| `LLM_BASE_URL` | OpenAI 兼容端点 base URL（如本地 vLLM / OneAPI） |
| `LLM_MODEL` | 默认模型名（两档未单独指定时共用） |
| `LLM_MODEL_FAST` | `fast` 档位模型 |
| `LLM_MODEL_SMART` | `smart` 档位模型 |
| `LLM_MODEL_VISION` | `vision` 档位模型（内置 VLM，默认与 smart 同档多模态模型） |
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

ProviderKind = Literal["openai", "anthropic"]

# 环境变量名
ENV_PROVIDER = "LLM_PROVIDER"
ENV_API_KEY = "LLM_API_KEY"
ENV_BASE_URL = "LLM_BASE_URL"
ENV_MODEL = "LLM_MODEL"
ENV_MODEL_FAST = "LLM_MODEL_FAST"
ENV_MODEL_SMART = "LLM_MODEL_SMART"
ENV_MODEL_VISION = "LLM_MODEL_VISION"

DEFAULT_PROVIDER: ProviderKind = "openai"

# 模型分层（FR-1）：profile → 远程模型 id（Anthropic 档位默认值，OpenAI 见 `_default_models`）。
PROFILE_MODELS: dict[str, str] = {
    "fast": "claude-haiku-4-5",
    "smart": "claude-opus-4-8",
}
DEFAULT_PROFILE = "smart"
MOCK_PROFILE = "mock"

_OPENAI_PROVIDER_ALIASES = frozenset({"openai", "openai_compatible", "openai_compat"})
_ANTHROPIC_PROVIDER_ALIASES = frozenset({"anthropic", "claude"})

_DEFAULT_MODELS: dict[ProviderKind, dict[str, str]] = {
    "anthropic": {**PROFILE_MODELS, "vision": "claude-sonnet-4-6"},
    "openai": {"fast": "gpt-4o-mini", "smart": "gpt-4o", "vision": "gpt-4o"},
}

_CLIENT_PACKAGES: dict[ProviderKind, tuple[str, str]] = {
    "openai": ("langchain_openai", "langchain-openai"),
    "anthropic": ("langchain_anthropic", "langchain-anthropic"),
}


@dataclass(frozen=True)
class LLMConfig:
    """从环境变量或显式参数解析出的 LLM 连接配置。"""

    provider: ProviderKind
    model: str | None = None
    model_fast: str | None = None
    model_smart: str | None = None
    model_vision: str | None = None
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(raw: str) -> ProviderKind:
    """把用户/provider 字符串规范为 `openai` 或 `anthropic`。"""
    normalized = raw.strip().lower().replace("-", "_")
    if normalized in _OPENAI_PROVIDER_ALIASES:
        return "openai"
    if normalized in _ANTHROPIC_PROVIDER_ALIASES:
        return "anthropic"
    valid = sorted(_OPENAI_PROVIDER_ALIASES | _ANTHROPIC_PROVIDER_ALIASES)
    raise ValueError(f"未知 LLM provider={raw!r}，可选：{valid}")


def load_llm_config_from_env(*, provider: str | None = None) -> LLMConfig:
    """读取 `.env` 与 `LLM_*` 环境变量；密钥可回退到 provider 惯例变量名。

    `provider` 用于显式覆盖环境中的 provider，并确保惯例密钥从匹配的
    `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY` 读取。
    """
    from robot_agent.env import ensure_env_loaded

    ensure_env_loaded()
    resolved_provider = normalize_provider(
        provider if provider is not None else os.getenv(ENV_PROVIDER, DEFAULT_PROVIDER)
    )
    api_key = os.getenv(ENV_API_KEY)
    if not api_key:
        if resolved_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv(ENV_BASE_URL) or None
    return LLMConfig(
        provider=resolved_provider,
        model=os.getenv(ENV_MODEL) or None,
        model_fast=os.getenv(ENV_MODEL_FAST) or None,
        model_smart=os.getenv(ENV_MODEL_SMART) or None,
        model_vision=os.getenv(ENV_MODEL_VISION) or None,
        api_key=api_key,
        base_url=base_url,
    )


def merge_llm_config(
    base: LLMConfig,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMConfig:
    """显式参数覆盖 `base` 中的同名字段。"""
    resolved_provider = (
        normalize_provider(provider) if provider is not None else base.provider
    )
    return LLMConfig(
        provider=resolved_provider,
        model=model if model is not None else base.model,
        model_fast=base.model_fast,
        model_smart=base.model_smart,
        model_vision=base.model_vision,
        api_key=api_key if api_key is not None else base.api_key,
        base_url=base_url if base_url is not None else base.base_url,
    )


def resolve_model_name(profile: str, config: LLMConfig) -> str:
    """按 profile 与配置解析最终模型 id。"""
    if profile not in {"fast", "smart", "vision"}:
        valid = sorted({*PROFILE_MODELS, "vision"}) + [MOCK_PROFILE]
        raise ValueError(f"未知 profile={profile!r}，可选：{valid}")

    per_profile = {
        "fast": config.model_fast,
        "smart": config.model_smart,
        "vision": config.model_vision,
    }[profile]
    if per_profile:
        return per_profile
    if config.model:
        return config.model
    if profile == "vision" and config.model_smart:
        return config.model_smart
    return _DEFAULT_MODELS[config.provider][profile]


class MockChatModel(BaseChatModel):
    """脚本化假聊天模型：按预设顺序逐条返回 `AIMessage`，支持 `bind_tools`。

    用于在不接真实远程 LLM 的前提下，确定性地驱动 `create_react_agent` 的工具调用
    往返与整条闭环（对齐 NFR-5 / AC-1 / AC-7）。`bind_tools` 仅记录而不改变回放序列：
    脚本里写好「第 N 次该返回什么」即可。
    """

    responses: list[AIMessage]
    idx: int = 0
    received: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        if self.idx >= len(self.responses):
            raise AssertionError(
                f"MockChatModel 响应已耗尽（已回放 {self.idx} 条，无更多脚本）："
                "请检查预设响应数量是否覆盖了全部 LLM 调用次数。"
            )
        msg = self.responses[self.idx]
        object.__setattr__(self, "idx", self.idx + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "MockChatModel":
        return self


def _import_client_class(provider: ProviderKind) -> type[BaseChatModel]:
    _, package_name = _CLIENT_PACKAGES[provider]
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic
    except ImportError as e:
        raise ImportError(
            f"provider={provider!r} 需要远程 LLM 客户端 `{package_name}`（未安装）；"
            "离线 / 测试请改用 make_model('mock', responses=[...])。"
        ) from e


def _build_remote_model(
    config: LLMConfig,
    model_name: str,
    **overrides: Any,
) -> BaseChatModel:
    client_cls = _import_client_class(config.provider)
    kwargs: dict[str, Any] = {"model": model_name, **overrides}
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    return client_cls(**kwargs)


def make_model(
    profile: str = DEFAULT_PROFILE,
    *,
    responses: Sequence[AIMessage] | None = None,
    config: LLMConfig | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **overrides: Any,
) -> BaseChatModel:
    """构建一个 `ChatModel`。

    - `profile="mock"` 或传入 `responses`：返回脚本化 `MockChatModel`（离线 / 测试 / 回归）。
    - `profile in {"fast", "smart", "vision"}`：按 provider 惰性构建远程客户端。
      连接参数来自显式参数（优先）或 `LLM_*` 环境变量；`overrides` 透传给客户端构造。
    """
    if profile == MOCK_PROFILE or responses is not None:
        return MockChatModel(responses=list(responses or []))

    resolved = merge_llm_config(
        config or load_llm_config_from_env(provider=provider),
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    model_name = resolve_model_name(profile, resolved)
    return _build_remote_model(resolved, model_name, **overrides)
