# 仓库协作说明（AGENTS）

本仓库是一个 monorepo，是从 LangGraph 官方仓库**裁剪**而来的「嵌入式机器人 Agent 底座」。
每个库位于 `libs/` 下的子目录中。背景与裁剪细节见 `docs/SLIMMING_NOTES.md`，
机器人 Agent 架构设计见 `docs/ROBOT_AGENT_DESIGN.md`。

当你修改任意库中的代码时，在创建 PR 前请在该库目录下运行：

- `make format` —— 运行代码格式化
- `make lint` —— 运行静态检查

验收/回归测试集中在仓库根目录的 `tests/`（瘦身时删除了上游各库自带的庞大单测）：

```txt
make test            # 根目录运行，等价于 pytest tests/
TEST=tests/test_memory.py make test   # 仅跑某个测试文件（每个库的 test 目标也支持 TEST 变量）
```

`TEST` 变量里也可以附加其它 pytest 参数。

## 库

裁剪后仅保留 4 个 Python 库：

- **checkpoint** —— LangGraph checkpointer 与 store 的基础接口（含内存实现）。
- **checkpoint-sqlite** —— checkpoint saver 与 store 的 SQLite 实现（短期/长期记忆，本地落盘）。
- **langgraph** —— 构建有状态、多 actor agent 的核心框架（StateGraph + Pregel 引擎）。
- **prebuilt** —— 创建与运行 agent / 工具的高层 API（`create_react_agent`、`ToolNode` 等）。

> 已删除的外围库：`cli`、`sdk-py`、`sdk-js`、`checkpoint-postgres`、`checkpoint-conformance`。

### 依赖关系图

下图按各库 `pyproject.toml` 声明的生产依赖，列出每个库的下游（依赖它的库）。

```text
checkpoint
├── checkpoint-sqlite
├── prebuilt
└── langgraph

prebuilt
└── langgraph
```

对某个库的改动可能影响上图中它的所有下游。

- 不要使用 Sphinx 风格的双反引号（` ``code`` `）。在 docstring 与注释里引用行内代码请用单反引号（`` `code` ``）。
