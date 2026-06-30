"""会话上下文管理：高水位触发滚动摘要，保留最近原文窗口。

较老的已完成消息会被压缩进 `RobotState.context_summary`，随后从 `messages` 中移除；
摘要与最近消息都由 checkpoint 持久化。压缩失败时只累计失败计数，不覆盖原消息或摘要；
本次模型输入改用硬窗口兜底，避免远程摘要服务故障破坏可恢复执行。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from robot_agent import prompts
from robot_agent.env import ensure_env_loaded

logger = logging.getLogger(__name__)

SUMMARY_MARKER = "CONTEXT_SUMMARY_V1"

ENV_CONTEXT_HIGH_WATERMARK = "CONTEXT_HIGH_WATERMARK_TOKENS"
ENV_CONTEXT_RECENT_WINDOW = "CONTEXT_RECENT_WINDOW_TOKENS"
ENV_CONTEXT_MAX_SUMMARY = "CONTEXT_MAX_SUMMARY_TOKENS"
ENV_CONTEXT_HARD_LIMIT = "CONTEXT_HARD_LIMIT_TOKENS"
ENV_CONTEXT_SUMMARY_BATCH = "CONTEXT_SUMMARY_BATCH_TOKENS"


@dataclass(frozen=True)
class ContextPolicy:
    """滚动上下文压缩策略，token 数均为近似值。

    `high_watermark_tokens` 触发压缩；压缩后保留约 `recent_window_tokens` 的最近原文，
    摘要最多 `max_summary_tokens`。`hard_limit_tokens` 仅在摘要失败或单条消息过大时兜底。
    """

    high_watermark_tokens: int = 10000
    recent_window_tokens: int = 5000
    max_summary_tokens: int = 1000
    hard_limit_tokens: int = 50000
    summary_batch_tokens: int = 3000

    def __post_init__(self) -> None:
        values = {
            "high_watermark_tokens": self.high_watermark_tokens,
            "recent_window_tokens": self.recent_window_tokens,
            "max_summary_tokens": self.max_summary_tokens,
            "hard_limit_tokens": self.hard_limit_tokens,
            "summary_batch_tokens": self.summary_batch_tokens,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数。")
        if self.recent_window_tokens >= self.high_watermark_tokens:
            raise ValueError(
                "recent_window_tokens 必须小于 high_watermark_tokens，压缩后才有回落空间。"
            )
        if self.high_watermark_tokens > self.hard_limit_tokens:
            raise ValueError(
                "high_watermark_tokens 不能大于 hard_limit_tokens，否则会先截断再压缩。"
            )
        if (
            self.recent_window_tokens + self.max_summary_tokens
            >= self.high_watermark_tokens
        ):
            raise ValueError(
                "recent_window_tokens + max_summary_tokens 必须小于 "
                "high_watermark_tokens，压缩后才不会立刻再次触发。"
            )


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是正整数，当前值为 {raw!r}。") from exc
    if value <= 0:
        raise ValueError(f"环境变量 {name} 必须是正整数，当前值为 {raw!r}。")
    return value


def load_context_policy_from_env() -> ContextPolicy:
    """从根目录 `.env` / 进程环境加载上下文限额，环境变量优先。"""
    ensure_env_loaded()
    defaults = ContextPolicy()
    try:
        return ContextPolicy(
            high_watermark_tokens=_env_positive_int(
                ENV_CONTEXT_HIGH_WATERMARK, defaults.high_watermark_tokens
            ),
            recent_window_tokens=_env_positive_int(
                ENV_CONTEXT_RECENT_WINDOW, defaults.recent_window_tokens
            ),
            max_summary_tokens=_env_positive_int(
                ENV_CONTEXT_MAX_SUMMARY, defaults.max_summary_tokens
            ),
            hard_limit_tokens=_env_positive_int(
                ENV_CONTEXT_HARD_LIMIT, defaults.hard_limit_tokens
            ),
            summary_batch_tokens=_env_positive_int(
                ENV_CONTEXT_SUMMARY_BATCH, defaults.summary_batch_tokens
            ),
        )
    except ValueError as exc:
        if "环境变量" in str(exc):
            raise
        raise ValueError(f"上下文限额环境变量组合无效：{exc}") from exc


# 应用导入时解析一次：shell 已导出值优先于仓库根 `.env`。
DEFAULT_CONTEXT_POLICY = load_context_policy_from_env()


@dataclass(frozen=True)
class PreparedContext:
    """一次上下文整理的结果。"""

    messages: list[BaseMessage]
    state_update: dict[str, Any]


def _message_tokens(messages: list[BaseMessage]) -> int:
    return count_tokens_approximately(messages)


def _summary_message(summary: str) -> SystemMessage:
    return SystemMessage(
        prompts.render(
            "context_summary",
            summary_json=json.dumps(summary, ensure_ascii=False),
        )
    )


def _summary_injection_overhead_tokens() -> int:
    """注入模板固定开销（不含摘要正文）。"""
    return _message_tokens([_summary_message("")])


def _inject_summary(messages: list[BaseMessage], summary: str) -> list[BaseMessage]:
    """把摘要放在稳定的前导 system 指令之后、普通历史之前。"""
    if not summary:
        return messages
    insert_at = 0
    while insert_at < len(messages) and isinstance(messages[insert_at], SystemMessage):
        insert_at += 1
    return [
        *messages[:insert_at],
        _summary_message(summary),
        *messages[insert_at:],
    ]


def _safe_content(content: Any) -> Any:
    """序列化历史供摘要模型读取，但不复制图片/base64 等非文本载荷。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return repr(content)

    blocks: list[str] = []
    for block in content:
        if isinstance(block, str):
            blocks.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            blocks.append(str(block.get("text", "")))
        elif isinstance(block, dict):
            block_type = str(block.get("type", "non_text"))[:64]
            blocks.append(f"[非文本载荷未送入上下文摘要模型：{block_type}]")
        else:
            blocks.append(f"[非文本载荷未送入：{type(block).__name__}]")
    return "\n".join(blocks)


def _safe_tool_calls(tool_calls: list[dict[str, Any]]) -> Any:
    """保留正常工具调用；参数总量异常时只留调用身份与载荷长度。"""
    encoded = json.dumps(tool_calls, ensure_ascii=False, default=str)
    if len(encoded) <= 4096:
        return tool_calls
    return [
        {
            "name": str(call.get("name", ""))[:128],
            "id": str(call.get("id", ""))[:128],
            "args": {"note": "过长工具参数已省略"},
        }
        for call in tool_calls[:32]
    ]


def _message_record(message: BaseMessage) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": message.type,
        "content": _safe_content(message.content),
    }
    if message.name:
        record["name"] = message.name
    if isinstance(message, AIMessage) and message.tool_calls:
        record["tool_calls"] = _safe_tool_calls(message.tool_calls)
    if isinstance(message, ToolMessage):
        record["tool_call_id"] = message.tool_call_id
        if status := getattr(message, "status", None):
            record["status"] = status
    return record


def _atomic_groups(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """把 AI tool-call 与其连续 tool results 组成不可拆分单元。"""
    groups: list[list[BaseMessage]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, AIMessage) and message.tool_calls:
            call_ids = {call["id"] for call in message.tool_calls}
            group = [message]
            index += 1
            while index < len(messages) and isinstance(messages[index], ToolMessage):
                tool_message = messages[index]
                if tool_message.tool_call_id not in call_ids:
                    break
                group.append(tool_message)
                index += 1
            groups.append(group)
            continue
        groups.append([message])
        index += 1
    return groups


def _split_old_and_recent(
    messages: list[BaseMessage], recent_window_tokens: int
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """按原子消息组切分，且保留输入中连续的前导 system 消息。"""
    leading_system: list[BaseMessage] = []
    first_non_system = 0
    while first_non_system < len(messages) and isinstance(
        messages[first_non_system], SystemMessage
    ):
        leading_system.append(messages[first_non_system])
        first_non_system += 1

    groups = _atomic_groups(messages[first_non_system:])
    if len(groups) <= 1:
        return [], messages

    # 当前回合必须完整保留：元认知的循环/步数检测依赖最后一条 human 作为回合锚点，
    # 而且把一个仍在执行的回合压进摘要会丢失精确的工具轨迹。
    current_turn_from = len(groups)
    for group_index, group in enumerate(groups):
        if any(message.type == "human" for message in group):
            current_turn_from = group_index

    budget = max(1, recent_window_tokens - _message_tokens(leading_system))
    keep_from = len(groups) - 1
    kept_tokens = _message_tokens(groups[keep_from])
    while keep_from > 0:
        candidate = groups[keep_from - 1]
        candidate_tokens = _message_tokens(candidate)
        if kept_tokens + candidate_tokens > budget:
            break
        keep_from -= 1
        kept_tokens += candidate_tokens
    keep_from = min(keep_from, current_turn_from)

    old = [message for group in groups[:keep_from] for message in group]
    recent = [
        *leading_system,
        *(message for group in groups[keep_from:] for message in group),
    ]
    return old, recent


def _records_to_batches(
    messages: list[BaseMessage], max_tokens: int
) -> list[list[dict[str, Any]]]:
    """把待归档消息分批，极端长单条文本按字符切块，避免摘要调用本身溢出。"""
    max_chars = max_tokens * 4
    records: list[dict[str, Any]] = []
    for message in messages:
        record = _message_record(message)
        content = record.get("content")
        if isinstance(content, str) and len(content) > max_chars:
            for offset in range(0, len(content), max_chars):
                chunk = dict(record)
                chunk["content"] = content[offset : offset + max_chars]
                chunk["chunk"] = f"{offset // max_chars + 1}"
                records.append(chunk)
        else:
            records.append(record)

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for record in records:
        encoded = json.dumps(record, ensure_ascii=False, default=str)
        if current and current_chars + len(encoded) > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(record)
        current_chars += len(encoded)
    if current:
        batches.append(current)
    return batches


def _parse_summary(response: BaseMessage, policy: ContextPolicy) -> str:
    if isinstance(response, AIMessage) and response.tool_calls:
        raise ValueError("摘要模型返回了工具调用，而不是纯文本摘要。")
    text = str(response.text).strip()
    first_line, separator, summary = text.partition("\n")
    if first_line.strip() != SUMMARY_MARKER or not separator or not summary.strip():
        raise ValueError(f"摘要模型输出缺少 {SUMMARY_MARKER} 标记或摘要正文为空。")
    summary = summary.strip()
    if _message_tokens([_summary_message(summary)]) > policy.max_summary_tokens:
        raise ValueError(
            "摘要模型输出超过 max_summary_tokens，拒绝用超长摘要替换历史。"
        )
    return summary


async def _roll_summary(
    model: BaseChatModel,
    previous_summary: str,
    old_messages: list[BaseMessage],
    policy: ContextPolicy,
) -> str:
    summary = previous_summary
    output_budget = policy.max_summary_tokens - _summary_injection_overhead_tokens()
    if output_budget <= 0:
        raise ValueError(
            "max_summary_tokens 不足以容纳会话摘要注入模板，请提高该限额。"
        )
    for batch in _records_to_batches(old_messages, policy.summary_batch_tokens):
        prompt = prompts.render(
            "context_summarize",
            previous_summary_json=json.dumps(summary, ensure_ascii=False),
            conversation_json=json.dumps(batch, ensure_ascii=False, default=str),
        )
        response = await model.ainvoke(
            [SystemMessage(prompt)],
            max_tokens=output_budget,
        )
        summary = _parse_summary(response, policy)
    return summary


def _hard_window(
    messages: list[BaseMessage], summary: str, hard_limit_tokens: int
) -> list[BaseMessage]:
    """仅作故障兜底；保留原始 system/摘要，再给最近原文分配剩余预算。"""
    prepared = _inject_summary(messages, summary)
    system_count = 0
    while system_count < len(prepared) and isinstance(
        prepared[system_count], SystemMessage
    ):
        system_count += 1
    return trim_llm_input_messages(
        prepared,
        preserve_prefix=system_count,
        hard_limit_tokens=hard_limit_tokens,
    )


def trim_llm_input_messages(
    messages: list[BaseMessage],
    *,
    preserve_prefix: int,
    hard_limit_tokens: int,
) -> list[BaseMessage]:
    """优先保留前导 system 块，并把总输入裁到硬上限内。

    `preserve_prefix` 指定必须保留的前导 system 数量（通常是身份锚点）。其后的连续
    system 块（长期记忆、原始规则、会话摘要）会在预算允许时依次保留；放不下的块被省略。
    必须保留的前缀本身超限时无法同时满足两项约束，会明确报错而不是返回超限输入。
    """
    if hard_limit_tokens <= 0:
        raise ValueError("hard_limit_tokens 必须是正整数。")
    if preserve_prefix < 0:
        raise ValueError("preserve_prefix 不能为负。")
    if preserve_prefix > len(messages):
        raise ValueError("preserve_prefix 不能大于消息数量。")
    if any(
        not isinstance(message, SystemMessage) for message in messages[:preserve_prefix]
    ):
        raise ValueError("preserve_prefix 范围内只能包含前导 SystemMessage。")

    prefix = list(messages[:preserve_prefix])
    prefix_tokens = _message_tokens(prefix)
    if prefix_tokens > hard_limit_tokens:
        raise ValueError(
            "必须保留的 system 前缀已超过 hard_limit_tokens，无法构造合法模型输入。"
        )

    # 其余连续前导 system 块是高优先级但可降级的数据；逐块纳入，单块放不下时省略，
    # 避免一条异常大的长期记忆挤掉身份锚点或让整个请求超过模型上限。
    rest_start = preserve_prefix
    while rest_start < len(messages) and isinstance(
        messages[rest_start], SystemMessage
    ):
        candidate = messages[rest_start]
        candidate_tokens = _message_tokens([candidate])
        if prefix_tokens + candidate_tokens <= hard_limit_tokens:
            prefix.append(candidate)
            prefix_tokens += candidate_tokens
        else:
            logger.warning(
                "前导 system 块超过剩余上下文预算，已省略（index=%s, tokens=%s）",
                rest_start,
                candidate_tokens,
            )
        rest_start += 1

    rest = list(messages[rest_start:])
    budget = hard_limit_tokens - prefix_tokens
    if budget <= 0 or not rest:
        return prefix
    if _message_tokens(rest) <= budget:
        return [*prefix, *rest]
    try:
        trimmed = trim_messages(
            rest,
            max_tokens=budget,
            token_counter=count_tokens_approximately,
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
        if _message_tokens(list(trimmed)) > budget:
            raise ValueError("trim_messages 返回结果仍超过剩余预算。")
    except Exception as exc:
        logger.warning("硬窗口标准裁剪失败，改用原子消息组兜底: %s", exc)
        groups = _atomic_groups(rest)
        kept: list[list[BaseMessage]] = []
        used = 0
        for group in reversed(groups):
            group_tokens = _message_tokens(group)
            if group_tokens > budget - used:
                break
            kept.append(group)
            used += group_tokens
        trimmed = [message for group in reversed(kept) for message in group]
    return [*prefix, *list(trimmed)]


async def prepare_context(
    state: dict[str, Any],
    *,
    summary_model: BaseChatModel | None,
    policy: ContextPolicy | None,
    reserved_tokens: int = 0,
) -> PreparedContext:
    """准备本次 LLM 历史，并在高水位时返回可持久化的压缩状态更新。

    `reserved_tokens` 为 hook 已确定会注入的前导 system 块（身份/记忆）预算，会从触发
    水位中扣除，避免「messages 未超水位、加上注入后撞模型上限」。
    """
    if reserved_tokens < 0:
        raise ValueError("reserved_tokens 不能为负。")
    messages: list[BaseMessage] = list(state.get("messages") or [])
    previous_summary = str(state.get("context_summary") or "").strip()

    if policy is None or summary_model is None:
        hard_limit = (
            policy.hard_limit_tokens
            if policy is not None
            else DEFAULT_CONTEXT_POLICY.hard_limit_tokens
        )
        return PreparedContext(_hard_window(messages, previous_summary, hard_limit), {})

    current = _inject_summary(messages, previous_summary)
    effective_watermark = max(1, policy.high_watermark_tokens - reserved_tokens)
    if _message_tokens(current) <= effective_watermark:
        return PreparedContext(current, {})

    effective_recent_window = max(
        1,
        min(
            policy.recent_window_tokens,
            effective_watermark - policy.max_summary_tokens,
        ),
    )
    old_messages, recent_messages = _split_old_and_recent(
        messages, effective_recent_window
    )
    if not old_messages:
        return PreparedContext(
            _hard_window(messages, previous_summary, policy.hard_limit_tokens), {}
        )

    try:
        summary = await _roll_summary(
            summary_model, previous_summary, old_messages, policy
        )
    except Exception as exc:
        # 摘要是增强能力，不得因远程模型异常破坏主闭环或覆盖 checkpoint。
        failures = int(state.get("context_compaction_failures") or 0) + 1
        logger.warning(
            "上下文滚动摘要失败，退回硬窗口（failures=%s）: %s",
            failures,
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return PreparedContext(
            _hard_window(messages, previous_summary, policy.hard_limit_tokens),
            {"context_compaction_failures": failures},
        )

    prepared = _inject_summary(recent_messages, summary)
    if _message_tokens(prepared) > policy.hard_limit_tokens:
        prepared = _hard_window(recent_messages, summary, policy.hard_limit_tokens)

    state_update = {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *recent_messages,
        ],
        "context_summary": summary,
        "context_compaction_count": int(state.get("context_compaction_count") or 0) + 1,
        "context_archived_messages": int(state.get("context_archived_messages") or 0)
        + len(old_messages),
    }
    return PreparedContext(prepared, state_update)


__all__ = [
    "DEFAULT_CONTEXT_POLICY",
    "ContextPolicy",
    "PreparedContext",
    "load_context_policy_from_env",
    "prepare_context",
    "trim_llm_input_messages",
]
