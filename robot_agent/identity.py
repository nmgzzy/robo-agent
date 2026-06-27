"""身份 / 自我模型（设计 §8.8，实现计划 §P3）：稳定注入「我是谁」。

长跑下若没有一个稳定的身份锚，决策语气与边界会随上下文漂移。本模块把身份
（persona / 价值观 / 能力自知）存入长期记忆 namespace `(robot_id, "identity")`，
由 `pre_model_hook` 每次调 LLM 前注入为 system context（成本几乎为零，最先立起来）。

与长期记忆（facts/episodic/prefs，§6）的区别：身份是**稳定锚点**，注入时置于所有
system 块**最前**，且不随检索波动；记忆是动态检索的环境知识，置于身份之后。

读取/更新接口（§P3.3）：初期由人工 `set_identity` 写入；后续可由复盘闭环（P6）更新。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.store.base import BaseStore

from robot_agent import prompts

# 身份 namespace 种类与固定 key（整份身份存为单条记录，便于整体读写）。
KIND_IDENTITY = "identity"
IDENTITY_KEY = "self"

# 默认身份：结构化默认值的唯一来源在 prompts/registry.json 的 identity.default_data，
# 此处取深拷贝常量供 ensure_default_identity 种入。可被 set_identity 覆盖。
DEFAULT_IDENTITY: dict[str, Any] = prompts.identity_default()


def identity_ns(robot_id: str) -> tuple[str, str]:
    """构造身份 namespace：`(robot_id, "identity")`（设计 §8.8）。"""
    return (robot_id, KIND_IDENTITY)


async def set_identity(
    store: BaseStore, robot_id: str, identity: Mapping[str, Any]
) -> None:
    """写入/覆盖整份身份（§P3.3 读取/更新接口的写侧）。

    深拷贝后落库：避免与 `InMemoryStore` 共享嵌套 list/dict，调用方后续改入参不污染已存值。
    """
    await store.aput(identity_ns(robot_id), IDENTITY_KEY, copy.deepcopy(dict(identity)))


async def get_identity(store: BaseStore, robot_id: str) -> dict[str, Any] | None:
    """读取整份身份；未设置则返回 None。

    深拷贝返回：`InMemoryStore` 下直接返回存储对象，调用方改动会回灌污染 store，故隔离。
    """
    item = await store.aget(identity_ns(robot_id), IDENTITY_KEY)
    return copy.deepcopy(dict(item.value)) if item is not None else None


async def ensure_default_identity(
    store: BaseStore,
    robot_id: str,
    identity: Mapping[str, Any] = DEFAULT_IDENTITY,
) -> dict[str, Any]:
    """身份缺失时种入默认值；已存在则原样返回（幂等，适合启动时调用）。

    返回值深拷贝：防止调用方改动回灌污染 store 或进程级 `DEFAULT_IDENTITY`。
    """
    existing = await get_identity(store, robot_id)
    if existing is not None:
        return existing
    await set_identity(store, robot_id, identity)
    return copy.deepcopy(dict(identity))


def _format_list(items: Sequence[Any]) -> str:
    return "；".join(str(x) for x in items)


def format_identity(identity: Mapping[str, Any]) -> str:
    """把身份渲染为稳定的 system context 文本块（注入用）。

    模板在 `prompts/identity.md`；此处只做结构→参数的扁平化（list 拼成文本）。
    所有字段均渲染（缺失则为空），与纯模板取向一致；默认身份字段齐全。
    """
    caps = identity.get("capabilities") or {}
    return prompts.render(
        "identity",
        name=identity.get("name", ""),
        persona=identity.get("persona", ""),
        values=_format_list(identity.get("values") or []),
        good_at=_format_list(caps.get("good_at") or []),
        bad_at=_format_list(caps.get("bad_at") or []),
    )


async def load_identity_text(store: BaseStore, robot_id: str) -> str | None:
    """检索并格式化身份文本；未设置则返回 None（供 `pre_model_hook` 调用）。"""
    identity = await get_identity(store, robot_id)
    if not identity:
        return None
    return format_identity(identity)
