# 提示词集中管理（JSON 索引 + Markdown 正文）

日期：2026-06-27
状态：已实现

## 背景与目标

裁剪自 LangGraph 的「嵌入式机器人 Agent 底座」，其 LLM 可见文案（系统提示词 / 任务提示词）
原先以内联字符串散落在 7 个模块里，优化时要逐个翻代码、改一处看不到全貌。本次把它们**集中
外置**为「JSON 索引 + Markdown 正文」，让提示词成为可独立查看、对比、迭代的数据。

本仓库**没有**一个写死的全局 system prompt——系统上下文是每回合由 `pre_model_hook` 动态拼装
注入的。本设计覆盖全部 7 处文案，分两类：

- **A. 注入对话的 system 上下文**：身份锚点、长期记忆头部、目标开回合指令、元认知告警。
- **B. 辅助 LLM 调用的任务提示词**：目标分解、复盘蒸馏、记忆冲突消解。

## 方案

`JSON 做索引，指向 MD 正文`：一个 `registry.json` 登记每条提示词的元数据，正文各自一个 `.md`。
结构化默认数据（identity）放 JSON，自然语言模板放 MD。

### 目录结构

新增 `robot_agent/prompts/` 包（应用层走 `pythonpath=.`，MD/JSON 用 `Path(__file__).parent`
读同级文件，无需打包 data）：

```
robot_agent/prompts/
├── __init__.py        # loader：render(id, **kwargs) / identity_default()
├── registry.json      # 索引：id → 描述 / md 文件 / params /（identity 的）default_data
├── identity.md        # 身份渲染模板（system 锚点）
├── memory_header.md   # 长期记忆注入头部
├── goal_turn.md       # 开回合指令（带 plan 分支，仅 with-plan 走模板）
├── metacog_warn.md    # 元认知告警
├── plan.md            # 目标分解（辅助调用）
├── distill.md         # 复盘蒸馏（辅助调用）
└── conflict.md        # 记忆冲突消解（辅助调用）
```

### registry.json

每条登记 `description` / `file` / `params`（声明的 `{占位符}`）；`identity` 额外带
`default_data`（结构化默认身份的**唯一来源**，替代旧 `DEFAULT_IDENTITY` 常量）。

- `params` 与 MD 内 `{占位符}` 在加载时**严格一致性校验**（集合相等），拼错/漏填当场报错。

### loader API（`robot_agent/prompts/__init__.py`）

```python
render(prompt_id: str, **params) -> str   # 取模板 .format(**params) 渲染
identity_default() -> dict                 # registry 里 identity 的 default_data（深拷贝）
```

健壮性（贴合嵌入式「依赖少、可靠」）：

- **启动即加载**：首次 import 一次性读 registry + 全部 MD 并缓存进程内；长跑不再碰磁盘。
- **fail-fast**：registry 缺失 / JSON 损坏 / MD 缺失 / 占位符与 params 不一致 → 加载即抛
  `PromptError`。这些文件随代码打包，缺了即打包 bug，宁可启动炸也不带病上线。
- 模板尾换行在加载时 `rstrip("\n")`，与原内联字符串（无尾换行）渲染结果一致。

### 调用点改写（行为保持不变）

| 文件 | 改法 |
|---|---|
| `identity.py` | `DEFAULT_IDENTITY = prompts.identity_default()`；`format_identity` → `render("identity", …)` |
| `memory.py` | `_format_memory` 头部 → `render("memory_header", items=…)` |
| `goals/policy.py` | `_format_goal_prompt` 保留 `if goal.plan` 分支，有 plan 时 `render("goal_turn", …)` |
| `metacog/monitor.py` | warn → `render("metacog_warn", reason=…)` |
| `goals/planning.py` | `PLAN_PROMPT` → `render("plan", intent=…)` |
| `reflect/distill.py` | `DISTILL_PROMPT` → `render("distill", episodes=…)` |
| `governance/compaction.py` | `CONFLICT_PROMPT` → `render("conflict", memories=…)` |

### 已接受的行为变化

旧 `format_identity()` 会**跳过缺失字段**；改为纯模板后所有字段都渲染（缺失显示为空）。默认
身份字段齐全，无影响；仅运行时塞入残缺身份才有差别。已确认接受，换取 identity.md 是干净模板。

## 测试

- 新增 `tests/test_prompts.py`：registry 加载、每模板可渲染、占位符↔params 一致性、
  `identity_default()` 深拷贝与关键字段、未知 id / 缺参数的 fail-fast、关键文案 substring 保留。
- 既有测试（identity/memory/goals/reflect/metacog/governance）全部保持通过——行为不变是硬约束。

## 后续优化入口

要调系统提示词，直接编辑对应 `.md`；要调默认身份，改 `registry.json` 的
`identity.default_data`；要新增提示词，在 registry 登记一条 + 建 `.md`，调用处 `render(id, …)`。
