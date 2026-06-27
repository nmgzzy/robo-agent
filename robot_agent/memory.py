"""记忆体系（设计 §6）：长期记忆 namespace 约定 + 主循环衔接 hook。

- **namespace 约定**（§6.2）：P1 先落 `facts` / `episodic` / `prefs`，按 `(robot_id, kind)` 组织。
- **`pre_model_hook`（§6.3）**：调 LLM 前，从 Store 检索相关长期记忆拼进上下文，
  同时裁剪过长的 `messages`（控制 token 与时延，嵌入式关键）。
- **事实回写**（§6.3）：`remember_fact` 工具把新学到的事实/偏好写回 Store，
  实现「越用越懂这个环境」。

检索用结构化 `asearch`（无额外算力开销，嵌入式首选；语义检索属可选增强，见 §6.2）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.tools import tool

from langgraph.config import get_store
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from robot_agent.identity import load_identity_text

# 长期记忆 namespace 种类（设计 §6.2，P1 范围）。
KIND_FACTS = "facts"
KIND_EPISODIC = "episodic"
KIND_PREFS = "prefs"
P1_KINDS: tuple[str, ...] = (KIND_FACTS, KIND_EPISODIC, KIND_PREFS)

# 注入上下文时默认检索的种类：facts（环境事实）+ prefs（用户偏好），与决策强相关。
DEFAULT_RECALL_KINDS: tuple[str, ...] = (KIND_FACTS, KIND_PREFS)

# 默认裁剪预算（近似 token）：嵌入式下控制单次 LLM 输入规模。
DEFAULT_MAX_TOKENS = 4096


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
    return "已知的长期记忆（供决策参考，可能不完整）：\n" + "\n".join(lines)


def make_inject_memory(
    robot_id: str,
    *,
    kinds: Sequence[str] = DEFAULT_RECALL_KINDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    inject_identity: bool = True,
):
    """构造 `pre_model_hook`：注入身份 + 检索长期记忆 + 裁剪 `messages`（设计 §6.3 / §8.8）。

    返回的 hook 是 async（用 `asearch`，兼容 `AsyncSqliteStore`）。它返回 `llm_input_messages`
    —— 只作为本次 LLM 输入、**不**改写 State 里的 `messages`（保留完整短期记忆给落盘/复盘）。

    注入顺序（均为 system 块，置于历史之前）：身份（稳定锚点，最前）→ 长期记忆（动态检索）。
    `inject_identity=True` 时若 `(robot_id, "identity")` 有身份则注入（设计 §P3.2）。
    """
    kinds = tuple(kinds)

    async def inject_memory(state: dict) -> dict:
        messages: list[BaseMessage] = list(state.get("messages") or [])

        # 1) 裁剪历史：超预算时保留最近的若干条，控制 token 与时延。
        #    start_on="human" 尽量避免裁出孤立的 tool 结果；裁剪失败则回退原始历史。
        try:
            trimmed = trim_messages(
                messages,
                max_tokens=max_tokens,
                token_counter=count_tokens_approximately,
                strategy="last",
                start_on="human",
                include_system=False,
                allow_partial=False,
            )
        except Exception:
            trimmed = messages

        # 2) 检索长期记忆。Store 未配置时（get_store 抛错或返回 None）安全跳过，只做裁剪。
        try:
            store: BaseStore | None = get_store()
        except RuntimeError:
            store = None
        if store is None:
            return {"llm_input_messages": trimmed}

        system_blocks: list[BaseMessage] = []

        # 2a) 身份：稳定锚点，置于所有 system 块最前（设计 §8.8）。
        if inject_identity:
            identity_text = await load_identity_text(store, robot_id)
            if identity_text is not None:
                system_blocks.append(SystemMessage(identity_text))

        # 2b) 长期记忆：动态检索，置于身份之后、历史之前。
        items_by_kind: dict[str, list[Any]] = {}
        for kind in kinds:
            items_by_kind[kind] = await store.asearch(ns(robot_id, kind))
        memory_text = _format_memory(items_by_kind)
        if memory_text is not None:
            system_blocks.append(SystemMessage(memory_text))

        return {"llm_input_messages": [*system_blocks, *trimmed]}

    return inject_memory


def build_memory_tools(robot_id: str) -> list[Any]:
    """构造记忆回写/读取工具（事实回写路径，设计 §6.3）。

    返回 `[remember_fact, recall]`，经 `InjectedStore` 直接拿到长期记忆，无需 LLM 传 store。
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
