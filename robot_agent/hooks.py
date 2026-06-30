"""`pre_model_hook` 装饰器共用的消息编排辅助。"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage


def insert_after_leading_system_messages(
    messages: list[BaseMessage],
    block: SystemMessage,
) -> list[BaseMessage]:
    """在连续前导 `SystemMessage` 块（身份锚点、记忆头等）之后插入一条 system 块。"""
    insert_at = 0
    while insert_at < len(messages) and isinstance(messages[insert_at], SystemMessage):
        insert_at += 1
    return [*messages[:insert_at], block, *messages[insert_at:]]
