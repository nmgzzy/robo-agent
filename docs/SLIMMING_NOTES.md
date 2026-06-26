# 瘦身记录（SLIMMING_NOTES）

本仓库已从 LangGraph 官方完整 monorepo 裁剪为「嵌入式机器人 Agent 底座」。
目标：依赖少、性能优、可长期可靠运行、可调试，LLM 走远程 API。

本文件记录所有改动点，便于将来跟随上游升级时手动 rebase。

## 一、保留的库

| 库 | 作用 |
|----|------|
| `libs/langgraph` | 核心引擎：StateGraph + Pregel 执行循环 + channels + types/runtime |
| `libs/checkpoint` | `BaseCheckpointSaver` / `BaseStore` 接口 + 内存实现 |
| `libs/checkpoint-sqlite` | 短期记忆 `SqliteSaver` + 长期记忆 `SqliteStore`（本地落盘） |
| `libs/prebuilt` | `create_react_agent` / `ToolNode` / `tools_condition` |

## 二、物理删除（Part A）

整目录删除：
- `libs/cli`、`libs/sdk-py`、`libs/checkpoint-postgres`、`libs/checkpoint-conformance`、`libs/sdk-js`
- `examples/`、`docs/`
- 各保留库内的 `tests/`、`bench/`（含 `conftest.py`）

体积：约 15M → 4.4M（源码部分约 3M，其余为安装产生的 egg-info/缓存）。

## 三、langgraph core 源码裁剪（Part B）

去掉对 `langgraph-sdk` 的硬依赖与远程执行能力。**这些是偏离上游的补丁点：**

1. **删除文件**
   - `libs/langgraph/langgraph/pregel/remote.py`（`RemoteGraph` 远程图客户端）
   - `libs/langgraph/langgraph/pregel/_remote_run_stream.py`（远程流）
   - 说明：`RemoteGraph` 不在任何 `__init__` 导出（命名空间包，用户需 `from langgraph.pregel.remote import RemoteGraph` 直接导入），内部无其它引用，删除是安全的。

2. **`libs/langgraph/langgraph/runtime.py`**
   - 删除 `from langgraph_sdk.auth.types import BaseUser`。
   - 新增本地最小 `BaseUser` Protocol（`@runtime_checkable`，含 `is_authenticated` / `display_name` / `identity`）作为类型桩，保持 `__all__` 与 `ServerInfo.user` 的公开 API 不变。
   - 升级提示：若上游 `BaseUser` 协议新增成员，按需同步此桩。

3. **`libs/langgraph/pyproject.toml`**
   - 运行时 `dependencies` 删除 `langgraph-sdk`。
   - `dependency-groups.test` 删除 `langgraph-checkpoint-postgres`、`langgraph-sdk`、`psycopg[binary]`、`langgraph-cli`（两条）。
   - `[tool.uv.sources]` 删除指向已删目录的 `langgraph-checkpoint-postgres` / `langgraph-sdk` / `langgraph-cli` 路径项。

4. **`libs/prebuilt/pyproject.toml`**
   - `test` 组删除 `langgraph-checkpoint-postgres`、`psycopg-binary`。
   - `[tool.uv.sources]` 删除 `langgraph-checkpoint-postgres` 路径项。

5. **`uv.lock`**
   - 删除了 `libs/langgraph/uv.lock` 与 `libs/prebuilt/uv.lock`（旧 lock 仍指向已删库）。
   - **需重新生成**：在对应库目录执行 `uv lock`（需联网）。`uv pip install -e <dir>` 不依赖 lock，可直接用。

## 四、运行时依赖（裁剪后）

`langgraph` 元数据 `Requires`：
```
langchain-core, langgraph-checkpoint, langgraph-prebuilt, pydantic, xxhash
```
加上 `checkpoint` 的 `ormsgpack`、`checkpoint-sqlite` 的 `aiosqlite` + `sqlite-vec` + `orjson`。

**直接依赖层面：已无 `langgraph-sdk` / `psycopg`**，也不再有 `sdk-py` 引入的 `websockets`，以及 `remote.py` 里对 `httpx` 的直接使用。

> 说明（准确起见）：`httpx` / `requests` 仍会**经 `langchain-core → langsmith` 传递性引入**，且核心的默认重试谓词
> `langgraph/_internal/_retry.py::default_retry_on` 会在被调用时**惰性 import** 它们来识别 5xx。
> 也就是说我们去掉的是「远程图执行 / Server SDK」这条重链路，但 LangChain 生态自带的轻量 HTTP 客户端仍在依赖树中。
> 若要进一步剔除，可考虑 `langsmith` 的可选裁剪或禁用遥测（属后续优化，见 todo）。

LLM 网络访问由用户另装的聊天模型客户端（如 `langchain-anthropic`）提供。

## 五、安装与验证

```bash
# 干净环境安装（editable，本地优先）
uv venv && source .venv/bin/activate
uv pip install -e libs/checkpoint -e libs/checkpoint-sqlite -e libs/prebuilt -e libs/langgraph

# 冒烟
python -c "from langgraph.graph import StateGraph; from langgraph.prebuilt import create_react_agent, ToolNode; from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; from langgraph.store.sqlite.aio import AsyncSqliteStore; print('OK')"
```
已验证：未安装 `langgraph-sdk` 时全部导入正常；最小图挂 `AsyncSqliteSaver`+`AsyncSqliteStore`，重启后按 `thread_id` 恢复短期状态、跨会话读长期记忆均通过。

> 注意：`pip install -e libs/prebuilt` **单独**安装不自洽——`langgraph-prebuilt` 在 import 期会用到核心模块，但其依赖刻意不声明 `langgraph`（避免与 `langgraph → langgraph-prebuilt` 形成硬循环，这是上游既定设计）。务必按上面顺序四个库一起装。

## 六、后续可选优化（todo）

- [ ] 进一步剔除传递性 `httpx`/`requests`：评估能否不安装 / 裁剪 `langsmith`（`langchain-core` 的传递依赖），或至少禁用遥测（`LANGCHAIN_TRACING_V2=false`）以减小依赖面与启动开销。
- [ ] `langgraph/_internal/_retry.py::default_retry_on` 惰性 import `httpx`/`requests`；若最终不安装它们，需确认默认重试谓词不会被触发，或替换为自定义 `RetryPolicy`。
- [ ] 各库 `uv.lock` 已删/过期，联网后执行一次 `uv lock` 重生成（`uv pip install -e` 不受影响）。
