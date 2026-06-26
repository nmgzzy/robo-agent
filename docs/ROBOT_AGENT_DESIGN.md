# 机器人 Agent 架构设计（基于裁剪后的 LangGraph 底座）

> 本文是**设计文档**，描述如何在已瘦身的 LangGraph 底座上构建一个跑在嵌入式 Linux 上的机器人 Agent。
> 职责范围：**思考、决策、调用工具控制机器人、管理短期/长期记忆**。
> 不在范围内：机器人的感知与底层控制实现（由外部模块提供，本 Agent 只通过「工具」调用它们、通过「状态」读取它们的结果）。
> 本次仅交付设计，不落业务代码。下文代码片段为蓝图示意。

---

## 1. 设计约束与总体取舍

| 约束 | 设计决策 |
|------|----------|
| 依赖少 | 已裁剪到 `langchain-core / langgraph-checkpoint / langgraph-prebuilt / pydantic / xxhash / ormsgpack / aiosqlite / sqlite-vec / orjson`。无数据库服务依赖、无 Server/远程图链路。（注：`langchain-core → langsmith` 仍传递性带入轻量 `httpx`/`requests`，详见 SLIMMING_NOTES.md。） |
| LLM 远程 API 为主 | 通过 `langchain-core` 的 ChatModel 抽象接入远程模型（推荐 `langchain-anthropic` + Claude）。网络/密钥都封装在该客户端，底座自身不再直接依赖 `langgraph-sdk` 的 httpx/websockets 远程链路。 |
| 性能优 | 异步底座（`aiosqlite`），LLM 网络 I/O 与机器人控制并发；冷启动快；SQLite WAL 模式。 |
| 长期可靠运行 | 短期记忆 checkpoint 落盘 → 崩溃/断电后按 `thread_id` 恢复续跑；节点重试 + 超时；interrupt 做安全/急停门控。 |
| 可调试 | `stream_mode=debug/updates/values` 逐步观察；本地 `*.db` 可用 `sqlite3` CLI 离线审查；全程无云依赖。 |

---

## 2. 分层架构

```
┌──────────────────────────────────────────────────────────────┐
│  应用层：任务编排 / 多回合会话 / 安全策略                          │
├──────────────────────────────────────────────────────────────┤
│  Agent 主循环（本设计核心）                                       │
│   思考(LLM 决策) ─▶ 决策路由 ─▶ 工具调用(控制机器人) ─┐            │
│        ▲                                            │            │
│        └────────────── 回灌结果/记忆注入 ◀───────────┘            │
├───────────────┬───────────────────────┬──────────────────────┤
│ LLM 接入       │ 记忆管理               │ 工具层                  │
│ ChatModel(远程)│ 短期: SqliteSaver      │ ToolNode 包装机器人动作  │
│                │ 长期: SqliteStore      │（移动/抓取/查询状态…）   │
├───────────────┴───────────────────────┴──────────────────────┤
│  裁剪后的 LangGraph 引擎：StateGraph + Pregel + channels         │
├──────────────────────────────────────────────────────────────┤
│  外部（不在本 Agent 范围）：感知、定位、运动控制、硬件驱动          │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 主循环：思考 → 决策 → 行动

### 3.1 首选：复用 `create_react_agent`

最省代码、直接满足「思考+决策+工具调用+记忆」四件事。位置：
`libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py`。它原生支持：
- `model`：远程 ChatModel（决策大脑）
- `tools`：机器人控制工具列表 → 内部用 `ToolNode` 执行
- `checkpointer`：短期记忆（线程内状态、可恢复）
- `store`：长期记忆（跨会话）
- `pre_model_hook` / `post_model_hook`：**记忆注入与裁剪**的挂载点（见 §5）

> 注：`langgraph.prebuilt.create_react_agent` 已被上游标注 deprecated（迁移到 `langchain.agents.create_agent`，属更重的 `langchain` 包）。
> 本瘦身构建**不含 `langchain`**，继续用 `langgraph.prebuilt` 版本可避免新增依赖（仅有一条 DeprecationWarning，可按需静音）；
> 若想完全规避弃用项，直接用 §3.2 的自定义 `StateGraph` 即可——能力等价。

蓝图：
```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
# from langchain_anthropic import ChatAnthropic   # 远程 LLM 客户端（用户另装）

agent = create_react_agent(
    model=ChatAnthropic(model="claude-opus-4-8"),  # 决策；高频轻量任务可用 claude-haiku-4-5
    tools=[move_to, grasp, get_world_state],        # 机器人控制动作（占位，见 §4）
    checkpointer=saver,      # 短期记忆
    store=store,             # 长期记忆
    pre_model_hook=inject_memory,    # 调 LLM 前注入长期记忆 + 裁剪历史
    # post_model_hook=safety_gate,   # 决策后安全门控（可选）
)
out = await agent.ainvoke(
    {"messages": [("user", "把桌上的杯子拿给我")]},
    {"configurable": {"thread_id": "episode-42"}},
)
```

### 3.2 需要更强控制时：下沉到 `StateGraph`

当需要自定义状态通道（并发感知输入、世界模型）、显式安全门控节点、或非 ReAct 的循环结构时，用
`StateGraph`（`libs/langgraph/langgraph/graph/state.py`）手写：
```
START ─▶ think(LLM) ─▶ [tools_condition] ─▶ act(ToolNode) ─▶ think ...
                              └─▶ END
```
`tools_condition` / `ToolNode` 均来自 `libs/prebuilt`。两种方案可平滑切换，建议从 3.1 起步。

---

## 4. 状态与工具

### 4.1 状态（State）
- `messages`：对话/推理历史 = **工作记忆（短期）**，用 `add_messages` reducer 累积。
- 注入式只读世界状态：机器人感知结果（位姿、检测到的物体、电量等）由外部写入 State 的自定义字段；本 Agent 只读不算它怎么来的。

### 4.2 工具（控制机器人的唯一出口）
用 `ToolNode`（`libs/prebuilt/langgraph/prebuilt/tool_node.py`）把「机器人动作」暴露为工具，LLM 通过 tool-call 触发：
```python
@tool
def move_to(x: float, y: float) -> str:
    """移动底盘到坐标 (x, y)。"""
    return robot.navigate(x, y)   # ← 此处接外部运动控制，不在本 Agent 实现
```
要点：
- 工具是**思考与物理世界之间的受控边界**——所有副作用都过工具，便于审计与安全拦截。
- 工具可用 `InjectedState` / `InjectedStore`（来自 prebuilt）拿到当前状态与长期记忆，无需 LLM 显式传参。
- 危险动作建议配合 `interrupt`（见 §6）做执行前人工/规则确认。

---

## 5. 记忆管理

### 5.1 短期记忆（线程内 / 单次任务）— Checkpoint
- 实现：`AsyncSqliteSaver`（`libs/checkpoint-sqlite/.../checkpoint/sqlite/aio.py`），本地 `agent.db`。
- 语义：每一步状态快照按 `thread_id` 落盘。一个机器人「任务/回合」用一个 `thread_id`。
- 价值：进程崩溃/断电重启后，用同 `thread_id` 即可恢复到中断点继续（durable execution）；天然支持 interrupt/resume。
- 已验证：重启新连接后 `aget_state(cfg)` 能取回中断前的完整状态。

### 5.2 长期记忆（跨任务 / 跨会话）— Store
- 实现：`AsyncSqliteStore`（`libs/checkpoint-sqlite/.../store/sqlite/aio.py`）。接口见 `libs/checkpoint/langgraph/store/base/__init__.py`。
- 命名空间组织建议：
  - `(robot_id, "facts")`：环境/任务事实（如「充电桩在 (3,2)」）
  - `(robot_id, "episodic")`：历史经验（做过什么、结果如何）
  - `(robot_id, "prefs")`：用户偏好
- 检索：
  - 结构化/键值检索：`aget` / `asearch`（无额外算力开销，嵌入式首选）。
  - 语义检索（可选）：远程 embedding 写入 + `sqlite-vec` 本地向量检索；算力紧张时关闭，退回关键字。

### 5.3 记忆与主循环的衔接
- `pre_model_hook`：调 LLM 前，从 Store 检索相关长期记忆拼进上下文；同时**裁剪/摘要**过长的 `messages`，控制 token 与时延（嵌入式关键）。
- `post_model_hook` 或工具内：把新学到的事实/经验写回 Store，实现「越用越懂这个环境」。

---

## 6. 嵌入式可靠性 / 性能 / 可调试

**可靠性**
- 崩溃恢复：SQLite 落盘 checkpoint（§5.1）。
- 重试：Pregel 自带重试（`libs/langgraph/langgraph/pregel/_retry.py`），给易抖动的节点（远程 LLM 调用、网络工具）配 `RetryPolicy`。
- 超时与降级：节点级超时；远程 LLM 不可用时可回退到「保守安全策略 / 停在原地等待」。
- 安全门控：用 `interrupt`（`langgraph.types`）在危险动作前暂停，等规则或人工确认后 `Command(resume=...)` 继续。

**性能**
- 异步全链路（`aiosqlite`），LLM 网络等待期间不阻塞其它协程。
- 依赖少 → 冷启动快、内存占用低、攻击面小。
- SQLite 开 WAL 模式；定期清理过期 checkpoint 控制 `agent.db` 体积（长跑机器人务必做）。
- 模型分层：高频轻决策用小模型（`claude-haiku-4-5`），复杂规划用大模型（`claude-opus-4-8`），省时延与成本。

**可调试**
- 流式观测：`graph.astream(..., stream_mode="debug")` 看每个超步的节点输入/输出/写入通道；`updates` 看增量，`values` 看全量状态。
- 离线审查：`agent.db` / store DB 直接用 `sqlite3` CLI 查历史状态与记忆，无需云端。
- 结构化日志：在工具与 hook 里打点（决策、tool-call、记忆命中），便于现场排障。

---

## 7. 后续落地步骤（不在本次范围）

1. 选定并安装远程 LLM 客户端（建议 `langchain-anthropic`，配置 API key / base_url）。
2. 定义机器人控制工具集（move/grasp/query…）对接外部控制接口。
3. 定义 State schema（messages + 世界状态字段）与 `pre_model_hook` 记忆策略。
4. 用 `create_react_agent` 起步跑通闭环，再按需下沉到自定义 `StateGraph`。
5. 加 RetryPolicy / 超时 / interrupt 安全门 / checkpoint 清理任务。
6. 压测时延与内存，按嵌入式目标平台调优（模型分层、WAL、并发度）。
