"""记忆体系（设计 §6）：长期记忆 namespace 约定 + 主循环衔接 hook。

- **namespace 约定**（§6.2）：P1 先落 `facts` / `episodic` / `prefs`，按 `(robot_id, kind)` 组织。
- **`pre_model_hook`（§6.3）**：调 LLM 前，从 Store 检索长期记忆，并在高水位时把
  较老 `messages` 滚动压缩为 checkpoint 内的会话摘要。
- **事实回写**（§6.3）：`remember_fact` 工具把新学到的事实/偏好写回 Store，
  实现「越用越懂这个环境」。

检索用结构化 `asearch`（无额外算力开销，嵌入式首选；语义检索属可选增强，见 §6.2）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.config import get_store
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore

from robot_agent import prompts
from robot_agent.context import (
    DEFAULT_CONTEXT_POLICY,
    ContextPolicy,
    prepare_context,
    trim_llm_input_messages,
)
from robot_agent.identity import load_identity_text

# 长期记忆 namespace 种类（设计 §6.2，P1 范围）。
KIND_FACTS = "facts"
KIND_EPISODIC = "episodic"
KIND_PREFS = "prefs"
P1_KINDS: tuple[str, ...] = (KIND_FACTS, KIND_EPISODIC, KIND_PREFS)

# 注入上下文时默认检索的种类：facts（环境事实）+ prefs（用户偏好），与决策强相关。
DEFAULT_RECALL_KINDS: tuple[str, ...] = (KIND_FACTS, KIND_PREFS)


def ns(robot_id: str, kind: str) -> tuple[str, str]:
    """构造长期记忆 namespace：`(robot_id, kind)`（设计 §6.2）。"""
    return (robot_id, kind)


def _unwrap_value(value: Any) -> Any:
    """还原 `remember_fact` 的 `{"value": ...}` 包装为原值；其它形状的记录原样返回。

    兼容外部直接写入的任意 dict（如 `{"x": 0, "y": 0}`），不误拆。
    """
    if isinstance(value, dict) and set(value) == {"value"}:
        return value["value"]
    return value


def _format_memory(items_by_kind: Mapping[str, Sequence[Any]]) -> str | None:
    """把检索到的记忆条目格式化为注入用的文本块；无记忆则返回 None。"""
    lines: list[str] = []
    for kind, items in items_by_kind.items():
        for it in items:
            lines.append(f"- [{kind}] {it.key}: {_unwrap_value(it.value)}")
    if not lines:
        return None
    return prompts.render("memory_header", items="\n".join(lines))


def make_inject_memory(
    robot_id: str,
    *,
    kinds: Sequence[str] = DEFAULT_RECALL_KINDS,
    inject_identity: bool = True,
    summary_model: BaseChatModel | None = None,
    context_policy: ContextPolicy | None = None,
):
    """构造 `pre_model_hook`：会话压缩 + 身份/长期记忆注入（设计 §6.3 / §8.8）。

    返回的 hook 是 async（用 `asearch`，兼容 `AsyncSqliteStore`）。高水位压缩成功时会
    原子替换已归档的 `messages`，并把 `context_summary` 一并写入 checkpoint；低水位时只返回
    本次 `llm_input_messages`。摘要失败只累计失败计数、不覆盖原消息或摘要，并临时退回硬窗口。
    身份/长期记忆注入后若总输入仍超 `hard_limit_tokens`，会保留前导 system 块并裁剪会话尾部。

    注入顺序（均为 system 块，置于普通历史之前）：身份（稳定锚点，最前）→ State 原始
    system/会话摘要 → 长期记忆（动态检索）。硬限额下优先保住稳定规则和当前会话连续性，
    长期记忆块预算不足时可省略。
    `inject_identity=True` 时若 `(robot_id, "identity")` 有身份则注入（设计 §P3.2）。
    `context_policy=None` 时关闭 LLM 滚动摘要，但仍按 `hard_limit_tokens` 做硬窗口兜底。
    """
    kinds = tuple(kinds)

    async def inject_memory(state: dict) -> dict:
        identity_blocks: list[BaseMessage] = []
        memory_blocks: list[BaseMessage] = []

        try:
            store: BaseStore | None = get_store()
        except RuntimeError:
            store = None

        if store is not None:
            if inject_identity:
                identity_text = await load_identity_text(store, robot_id)
                if identity_text is not None:
                    identity_blocks.append(SystemMessage(identity_text))

            items_by_kind: dict[str, list[Any]] = {}
            try:
                for kind in kinds:
                    items_by_kind[kind] = await store.asearch(ns(robot_id, kind))
            except Exception:
                items_by_kind = {}
            memory_text = _format_memory(items_by_kind)
            if memory_text is not None:
                memory_blocks.append(SystemMessage(memory_text))

        reserved = count_tokens_approximately([*identity_blocks, *memory_blocks])
        hard_limit = (
            context_policy.hard_limit_tokens
            if context_policy is not None
            else DEFAULT_CONTEXT_POLICY.hard_limit_tokens
        )
        context = await prepare_context(
            state,
            summary_model=summary_model if context_policy is not None else None,
            policy=context_policy,
            reserved_tokens=reserved if context_policy is not None else 0,
        )

        context_system_count = 0
        while context_system_count < len(context.messages) and isinstance(
            context.messages[context_system_count], SystemMessage
        ):
            context_system_count += 1
        context_systems = context.messages[:context_system_count]
        context_tail = context.messages[context_system_count:]
        protected_prefix = [*identity_blocks, *context_systems]
        llm_input = trim_llm_input_messages(
            [*protected_prefix, *memory_blocks, *context_tail],
            preserve_prefix=len(protected_prefix),
            hard_limit_tokens=hard_limit,
        )

        return {
            **context.state_update,
            "llm_input_messages": llm_input,
        }

    return inject_memory


def build_memory_tools(robot_id: str, *, governance: Any = None) -> list[Any]:
    """构造记忆回写/读取工具（事实回写路径，设计 §6.3）。

    返回 `[remember_fact, recall]`，经 `InjectedStore` 直接拿到长期记忆，无需 LLM 传 store。
    配 `governance` 时，写回（副作用）也过治理校验——避免记忆写入成为绕过 P9 策略的缺口
    （读取 `recall` 无副作用，不拦）。
    """

    @tool
    async def remember_fact(
        key: str,
        value: str,
        store: Annotated[BaseStore, InjectedStore()],
        kind: str = KIND_FACTS,
    ) -> str:
        """把一条长期记忆写回 Store。kind ∈ {facts, episodic, prefs}（默认 facts）。"""
        if kind not in P1_KINDS:
            return f"拒绝写入：未知 kind={kind!r}，可选 {list(P1_KINDS)}。"
        if governance is not None:
            ok, reason = governance.check(
                "remember_fact", {"key": key, "value": value, "kind": kind}
            )
            if not ok:
                return f"remember_fact 被治理策略拒绝：{reason}"
        await store.aput(ns(robot_id, kind), key, {"value": value})
        return f"已记住 [{kind}] {key} = {value}"

    @tool
    async def recall(
        key: str,
        store: Annotated[BaseStore, InjectedStore()],
        kind: str = KIND_FACTS,
    ) -> str:
        """按 key 从长期记忆读取一条（kind 默认 facts）。"""
        item = await store.aget(ns(robot_id, kind), key)
        return str(_unwrap_value(item.value)) if item is not None else "none"

    return [remember_fact, recall]
