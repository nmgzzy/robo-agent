"""提示词集中管理（JSON 索引 + Markdown 正文）。

把散落在各模块的 LLM 可见文案（身份锚点、记忆头部、开回合指令、元认知告警，
以及目标分解 / 复盘蒸馏 / 记忆冲突消解 / 视觉理解与信任边界提示词）统一外置到本包：

- `registry.json`：索引。每条提示词登记 `description` / `file`（对应 `.md`）/ `params`
  （声明的 `{占位符}`）；`identity` 额外带 `default_data`（结构化默认身份的唯一来源）。
- `<id>.md`：提示词正文，正文里用 `{占位符}` 标参数，渲染时 `str.format` 填充。

设计取向（贴合嵌入式「依赖少、可靠」）：

- **启动即加载**：首次 import 时一次性读 registry + 全部 MD 并缓存在进程内；长跑不再碰磁盘。
- **fail-fast**：registry 缺失 / JSON 损坏 / MD 文件缺失 / 占位符与 `params` 不一致 → 加载时
  直接抛 `PromptError`。这些文件随代码打包，缺了即打包 bug，宁可启动炸也不带病上线。

对外只暴露两个函数：`render(id, **params)` 渲染最终文本；`identity_default()` 取结构化默认身份。
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent
_REGISTRY_PATH = _DIR / "registry.json"

# 匹配模板里的 `{占位符}`（占位符为合法标识符；用于与声明的 params 做一致性校验）。
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class PromptError(RuntimeError):
    """提示词加载 / 渲染出错（registry 损坏、文件缺失、占位符不一致、参数缺失）。"""


def _load() -> dict[str, dict[str, Any]]:
    """加载并校验 registry + 全部 MD 模板；返回 id → {template, params, data}。

    在 import 期一次性执行（见模块底部），任何不一致都在此 fail-fast。
    """
    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise PromptError(f"提示词索引缺失：{_REGISTRY_PATH}") from e
    except json.JSONDecodeError as e:
        raise PromptError(f"提示词索引 JSON 损坏：{_REGISTRY_PATH}：{e}") from e

    entries = raw.get("prompts")
    if not isinstance(entries, dict) or not entries:
        raise PromptError("registry.json 缺少非空的 `prompts` 字段。")

    compiled: dict[str, dict[str, Any]] = {}
    for pid, meta in entries.items():
        file = meta.get("file")
        if not file:
            raise PromptError(f"提示词 `{pid}` 未声明 `file`。")
        path = _DIR / file
        try:
            # 去掉尾部换行：MD 文件惯例以换行结尾，而模板正文不应携带尾换行
            # （原内联字符串均无尾换行，保持渲染结果一致）。
            template = path.read_text(encoding="utf-8").rstrip("\n")
        except FileNotFoundError as e:
            raise PromptError(f"提示词 `{pid}` 的正文文件缺失：{path}") from e

        declared = set(meta.get("params", []))
        found = set(_PLACEHOLDER_RE.findall(template))
        if declared != found:
            raise PromptError(
                f"提示词 `{pid}` 占位符与 params 不一致："
                f"params={sorted(declared)} 模板内={sorted(found)}。"
            )

        compiled[pid] = {
            "template": template,
            "params": declared,
            "data": meta.get("default_data"),
        }
    return compiled


_PROMPTS = _load()


def render(prompt_id: str, **params: Any) -> str:
    """取 `prompt_id` 的模板，用 `params` 渲染为最终文本。

    `prompt_id` 不存在或缺参数 → `PromptError`（定位清晰，避免静默拿到半成品提示词）。
    """
    entry = _PROMPTS.get(prompt_id)
    if entry is None:
        raise PromptError(f"未知提示词 id：{prompt_id!r}（可选：{sorted(_PROMPTS)}）。")
    try:
        return entry["template"].format(**params)
    except KeyError as e:
        raise PromptError(f"渲染提示词 `{prompt_id}` 缺少参数：{e}") from e


def identity_default() -> dict[str, Any]:
    """返回 identity 的结构化默认身份（深拷贝，防调用方改动回灌污染进程级数据）。"""
    entry = _PROMPTS.get("identity")
    data = entry.get("data") if entry else None
    if not isinstance(data, dict):
        raise PromptError("identity 未在 registry.json 中声明 `default_data`。")
    return copy.deepcopy(data)
