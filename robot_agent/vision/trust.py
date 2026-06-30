"""把视觉信任边界作为稳定 system 约束注入主模型输入。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.messages import BaseMessage, SystemMessage

from robot_agent import prompts
from robot_agent.hooks import insert_after_leading_system_messages

PreModelHook = Callable[[dict], Awaitable[dict]]


def make_vision_trust_hook(inner: PreModelHook) -> PreModelHook:
    """在已有 hook 的 system 锚点之后加入视觉不可信数据约束。

    一旦配置了 `vlm_model` 即对每个回合注入（纵深防御，不依赖本回合上下文是否含图），
    属固定且很小的 token 开销（提示词正文见 `prompts/vision_trust_policy.md`）。注入是
    临时 `llm_input_messages`、不落 `state.messages`，故不会跨回合累积。
    """

    async def inject_trust_boundary(state: dict) -> dict:
        result = await inner(state)
        hook_messages = result.get("llm_input_messages")
        messages: list[BaseMessage] = list(
            hook_messages if hook_messages is not None else state.get("messages") or []
        )
        policy = SystemMessage(prompts.render("vision_trust_policy"))
        return {
            **result,
            "llm_input_messages": insert_after_leading_system_messages(
                messages, policy
            ),
        }

    return inject_trust_boundary


__all__ = ["make_vision_trust_hook"]
