"""可靠性封装（设计 §7「可靠性」，实现计划 §P2）：重试 / 超时 / 降级 / 清理。

嵌入式机器人长跑，远程 LLM 与网络工具天然会抖（断网、超时、限流）。本模块把这些
易抖动的调用包成「失败可重试、超时可中断、彻底不可用时降级到保守安全策略」的形态：

- `ResilientChatModel`：包住任意 `BaseChatModel`（或 `bind_tools` 后的 Runnable），
  对 `_agenerate` 做有限次退避重试 + 单次超时；重试耗尽则返回**降级** `AIMessage`
  （不带任何工具调用 → 闭环安全收束，对应「停在原地等待」）。
- `cleanup_threads`：删除过期线程的 checkpoint（长跑控制 `agent.db` 体积）。
  WAL 模式已由 `AsyncSqliteSaver` 在建表时开启（见其 `setup()`），此处不重复设置。

为何不用 Pregel 自带 `RetryPolicy`：`create_react_agent` 未暴露按节点配置重试的入口，
且其默认谓词 `default_retry_on` 惰性 import `httpx`/`requests`（见 `SLIMMING_NOTES.md §六`，
瘦身后不保证可用）。在「决策大脑」这一层用自定义谓词包装，既精准又不引依赖。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from langgraph.checkpoint.base import BaseCheckpointSaver

# 远程 LLM 客户端（如 ChatAnthropic）抛的瞬态异常**不是** ConnectionError/TimeoutError 的子类，
# 而是 APIConnectionError / APITimeoutError / RateLimitError / OverloadedError 等。瘦身纪律下
# 不硬 import `langchain-anthropic` / `anthropic`，改按**类名**鸭子匹配（沿 MRO 兜住子类），
# 既覆盖真实 provider 故障，又不引依赖。其余异常（鉴权错误、代码 bug、控制流）不重试。
_TRANSIENT_EXC_NAMES: frozenset[str] = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
    }
)


def default_retry_on(exc: BaseException) -> bool:
    """默认重试谓词：内置网络/超时异常 + 主流 provider 的瞬态异常（按类名匹配）。"""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return any(c.__name__ in _TRANSIENT_EXC_NAMES for c in type(exc).__mro__)


# 降级话术：决策大脑彻底不可用时的保守兜底，明确「不动」语义且不带工具调用。
DEFAULT_FALLBACK_TEXT = "（降级）暂时无法连接决策大脑，按保守策略停在原地等待。"


class ResilientChatModel(BaseChatModel):
    """给「决策大脑」加重试 / 超时 / 降级的包装模型（async 主路径，对齐闭环 async-only）。

    - `bind_tools` 透传给 `inner` 并返回新的包装（共享 `calls` 记录，便于在外层断言）。
    - 重试只针对 `retry_on` 列出的瞬态异常；退避间隔 `initial_interval * backoff_factor**(n-1)`，
      `initial_interval=0`（默认）则不 sleep（离线测试零等待）。
    - 全部尝试失败 → 返回 `fallback_text` 的 `AIMessage`（无 tool_calls），闭环安全收束。
    """

    inner: Any
    max_attempts: int = 3
    timeout_s: float | None = None
    initial_interval: float = 0.0
    backoff_factor: float = 2.0
    # 谓词函数 `(exc) -> bool`，或异常类型 / 类型元组（isinstance 判定）。默认 default_retry_on。
    retry_on: (
        Callable[[BaseException], bool]
        | type[BaseException]
        | tuple[type[BaseException], ...]
    ) = default_retry_on
    fallback_text: str = DEFAULT_FALLBACK_TEXT
    # 每次 _agenerate 的可观测记录：{"attempts": n, "degraded": bool}，供测试断言重试/降级发生。
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "resilient-chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ResilientChatModel":
        # 把工具绑定下放给被包模型；浅拷贝共享 calls 记录，外层仍可断言内部重试行为。
        return self.model_copy(update={"inner": self.inner.bind_tools(tools, **kwargs)})

    def _should_retry(self, exc: BaseException) -> bool:
        ro = self.retry_on
        if isinstance(ro, (type, tuple)):  # 异常类型 / 类型元组
            return isinstance(exc, ro)
        return bool(ro(exc))  # 谓词函数

    async def _acall_inner(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        # 透传 stop 与 provider 专属 kwargs，保证包装前后语义一致（重试时亦然）。
        coro = self.inner.ainvoke(messages, stop=stop, **kwargs)
        if self.timeout_s is not None:
            return await asyncio.wait_for(coro, self.timeout_s)
        return await coro

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        interval = self.initial_interval
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                msg = await self._acall_inner(messages, stop=stop, **kwargs)
            # 仅捕获 Exception：放过 CancelledError / KeyboardInterrupt 等 BaseException 控制流。
            except Exception as exc:
                if not self._should_retry(exc):
                    raise  # 非瞬态（鉴权错误 / 代码 bug）：原样抛出，不吞成降级
                last_exc = exc
                if attempt < self.max_attempts:
                    if interval > 0:
                        await asyncio.sleep(interval)
                        interval *= self.backoff_factor
                    continue
                break  # 次数耗尽 → 降级
            else:
                self.calls.append({"attempts": attempt, "degraded": False})
                return ChatResult(generations=[ChatGeneration(message=msg)])

        # 所有尝试失败：降级为保守安全回复（无 tool_calls，闭环到此安全收束）。
        self.calls.append(
            {"attempts": self.max_attempts, "degraded": True, "error": repr(last_exc)}
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.fallback_text))]
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 闭环是 async-only；同步路径仅为接口完整性，复用异步实现。
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        )


def make_resilient(inner: BaseChatModel, **opts: Any) -> ResilientChatModel:
    """把一个 `BaseChatModel` 包成带重试 / 超时 / 降级的 `ResilientChatModel`。

    用法：`build_robot_agent(model=make_resilient(make_model("smart"), timeout_s=20))`。
    `opts` 透传给 `ResilientChatModel`（`max_attempts` / `timeout_s` / `retry_on` / `fallback_text` 等）。
    """
    return ResilientChatModel(inner=inner, **opts)


async def cleanup_threads(saver: BaseCheckpointSaver, thread_ids: Iterable[str]) -> int:
    """删除给定线程的全部 checkpoint（长跑清理，设计 §7「性能」）。

    返回成功删除的线程数。调用方负责挑选「过期」线程（如已完成且超出保留窗口的回合）。
    WAL 已由 `AsyncSqliteSaver` 建表时开启，此处只做体积回收。
    """
    deleted = 0
    for tid in thread_ids:
        await saver.adelete_thread(tid)
        deleted += 1
    return deleted
