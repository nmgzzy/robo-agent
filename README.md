<div align="center">
  <h1>嵌入式机器人 Agent 底座</h1>
  <h3>基于裁剪后的 LangGraph，构建有状态机器人 Agent 的低层编排框架</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/langgraph" alt="License"></a>
</div>

<br>

本仓库是从 [LangGraph](https://github.com/langchain-ai/langgraph) 官方 monorepo **裁剪**而来的精简底座，
面向**在嵌入式 Linux 上运行的机器人 Agent**：负责思考、决策、调用工具控制机器人、管理短期/长期记忆。
设计目标是**依赖少、性能优、可长期可靠运行、可调试**；LLM 推理以**远程 API** 为主。

> 完整背景与裁剪改动见 [`docs/SLIMMING_NOTES.md`](docs/SLIMMING_NOTES.md)；
> 机器人 Agent 架构设计见 [`docs/ROBOT_AGENT_DESIGN.md`](docs/ROBOT_AGENT_DESIGN.md)。

## 为什么用它？

它为**任何长时运行、有状态**的机器人工作流 / Agent 提供底层支撑：

- **可恢复执行（Durable execution）** —— Agent 可穿越故障/断电：状态落盘 checkpoint，重启后按 `thread_id` 从中断处精确续跑。
- **人在环（Human-in-the-loop）** —— 借助 `interrupt`，在执行任意环节暂停以检查/修改 Agent 状态，可用于安全/急停门控。
- **分层记忆** —— checkpoint 保存最近原文并在高水位自动滚动摘要较老会话，store 保存
  跨会话长期记忆；不会在正常路径上直接丢弃较老对话语义。
- **可调试** —— `stream_mode` 逐步观察状态流转；本地 SQLite（`*.db`）可用 `sqlite3` CLI 离线审查；全程无云依赖。

## 安装

本仓库以 monorepo 形式提供 4 个本地库（editable 安装）：

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements-app.txt
uv pip install -e libs/checkpoint -e libs/checkpoint-sqlite -e libs/prebuilt -e libs/langgraph
```

接入远程 LLM 时，另装聊天模型客户端（推荐 [Claude](https://www.anthropic.com/)）：

```bash
uv add langchain-anthropic   # 决策用 claude-opus-4-8；高频轻量任务用 claude-haiku-4-5
```

会话滚动摘要限额可在仓库根 `.env` 配置：

```dotenv
CONTEXT_HIGH_WATERMARK_TOKENS=10000
CONTEXT_RECENT_WINDOW_TOKENS=5000
CONTEXT_MAX_SUMMARY_TOKENS=1000
CONTEXT_HARD_LIMIT_TOKENS=50000
CONTEXT_SUMMARY_BATCH_TOKENS=3000
```

必须满足 `最近窗口 + 最大摘要 < 触发水位 <= 故障硬上限`。代码显式传入
`ContextPolicy(...)` 会覆盖环境默认值；传 `context_policy=None` 可关闭自动摘要。

## 库一览

| 库 | 作用 |
|----|------|
| `libs/langgraph` | 核心引擎：StateGraph + Pregel 执行循环 + channels + types/runtime |
| `libs/checkpoint` | checkpointer / store 基础接口与内存实现 |
| `libs/checkpoint-sqlite` | 短期记忆 `SqliteSaver` + 长期记忆 `SqliteStore`（本地落盘） |
| `libs/prebuilt` | `create_react_agent` / `ToolNode` 等高层 Agent 构建 API |

## 测试

```bash
make test          # 根目录，等价于 pytest tests/
```

验收/回归测试位于 [`tests/`](tests/)，全部本地运行、无需远程 LLM 或外部服务。详见 [`tests/README.md`](tests/README.md)。

## 致谢

LangGraph 受 [Pregel](https://research.google/pubs/pub37252/) 与 [Apache Beam](https://beam.apache.org/) 启发，
其公开接口借鉴了 [NetworkX](https://networkx.org/documentation/latest/)。LangGraph 由 LangChain Inc 构建，可脱离 LangChain 使用。
本仓库是其精简衍生版，遵循 MIT 许可证（见 [LICENSE](LICENSE)）。
