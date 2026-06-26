# 项目结构图 & 与原版 LangGraph 对比

本文用 Mermaid 图描述裁剪后「嵌入式机器人 Agent 底座」的结构，并与原版 LangGraph monorepo 做对比。
（在 GitHub / VS Code / 支持 Mermaid 的查看器中可直接渲染。）

---

## 一、当前项目分层结构

```mermaid
flowchart TB
    ROBOT["🤖 机器人 Agent 业务<br/>（感知 / 控制由外部提供，不在底座内）"]:::ext

    subgraph PB["libs/prebuilt — 高层 Agent API"]
        REACT["create_react_agent"]
        TOOLNODE["ToolNode / tools_condition"]
    end

    subgraph CORE["libs/langgraph — 核心引擎"]
        GRAPH["graph：StateGraph"]
        PREGEL["pregel：执行循环 / 重试 / interrupt"]
        CHAN["channels：状态通道"]
        STREAM["stream：流式输出"]
        RT["runtime / types<br/>（本地 BaseUser 桩，已去 sdk）"]
    end

    subgraph MEMIF["libs/checkpoint — 记忆接口"]
        CPBASE["BaseCheckpointSaver"]
        STBASE["BaseStore"]
        MEM["InMemorySaver / InMemoryStore"]
    end

    subgraph SQL["libs/checkpoint-sqlite — 本地落盘记忆"]
        SAVER["SqliteSaver / AsyncSqliteSaver<br/>🧠 短期记忆（崩溃恢复）"]
        STORE["SqliteStore / AsyncSqliteStore<br/>📚 长期记忆（跨会话 + 向量检索）"]
    end

    LLM["☁️ 远程 LLM（langchain-anthropic + Claude，另装）"]:::ext

    ROBOT --> REACT
    REACT --> TOOLNODE
    REACT --> CORE
    TOOLNODE --> CORE
    CORE --> MEMIF
    SQL -.实现.-> MEMIF
    REACT -. checkpointer .-> SAVER
    REACT -. store .-> STORE
    REACT --> LLM

    classDef ext fill:#eee,stroke:#999,stroke-dasharray:4 3,color:#333;
```

**思考 → 决策 → 行动 → 记忆**的闭环：`create_react_agent`（或自定义 `StateGraph`）用远程 LLM 决策，
经 `ToolNode` 调用工具控制机器人，状态由 `SqliteSaver` 落盘（短期、可恢复），
经验/事实写入 `SqliteStore`（长期、跨会话）。

---

## 二、库依赖关系（裁剪后，仅 4 库）

```mermaid
flowchart LR
    LG["langgraph"]
    CP["checkpoint"]
    PB["prebuilt"]
    CPS["checkpoint-sqlite"]

    LC["langchain-core"]
    MSG["ormsgpack"]
    AIO["aiosqlite"]
    VEC["sqlite-vec"]
    OJ["orjson"]
    PYD["pydantic"]
    XXH["xxhash"]

    LG --> CP
    LG --> PB
    PB --> CP
    CPS --> CP

    LG --> LC
    CP --> LC
    CP -->|序列化| MSG
    CPS --> AIO
    CPS --> VEC
    CPS --> OJ
    LG --> PYD
    LG --> XXH

    classDef lib fill:#d6f5d6,stroke:#2e7d32,color:#143d16;
    classDef dep fill:#e3f0ff,stroke:#1565c0,color:#0d2c52;
    class LG,CP,PB,CPS lib;
    class LC,MSG,AIO,VEC,OJ,PYD,XXH dep;
```

> 注：`httpx` / `requests` 仍经 `langchain-core → langsmith` 传递性带入（轻量）；
> 已无 `langgraph-sdk`（连带 websockets）与 `psycopg`。详见 [SLIMMING_NOTES.md](./SLIMMING_NOTES.md)。

---

## 三、与原版 LangGraph 对比

### 3.1 库层面：保留 vs 删除

```mermaid
flowchart TB
    subgraph KEEP["✅ 保留（4）"]
        K1["langgraph<br/>核心引擎"]
        K2["checkpoint<br/>记忆接口"]
        K3["checkpoint-sqlite<br/>本地落盘记忆"]
        K4["prebuilt<br/>高层 Agent API"]
    end

    subgraph DROP["❌ 删除（5 库 + 外围）"]
        D1["cli<br/>命令行工具"]
        D2["sdk-py<br/>远程 API 客户端"]
        D3["sdk-js<br/>JS/TS SDK"]
        D4["checkpoint-postgres<br/>Postgres 后端"]
        D5["checkpoint-conformance<br/>一致性测试"]
        D6["examples / docs / 各库 tests"]
    end

    KEEP --- NOTE1["+ core 内部裁剪：<br/>删 pregel/remote.py、_remote_run_stream.py<br/>runtime.py 本地 BaseUser 桩"]:::note

    classDef keep fill:#d6f5d6,stroke:#2e7d32,color:#143d16;
    classDef drop fill:#ffd9d9,stroke:#c62828,color:#5c1212;
    classDef note fill:#fff7d6,stroke:#f9a825,color:#5c4a12;
    class K1,K2,K3,K4 keep;
    class D1,D2,D3,D4,D5,D6 drop;
```

### 3.2 对比表

| 维度 | 原版 LangGraph monorepo | 当前底座（裁剪后） |
|------|--------------------------|---------------------|
| Python 库数量 | 8（+1 JS 桩） | **4** |
| 源码体积 | ~13M（含 tests/examples） | **~2.5M**（4 库源码） |
| `langgraph` 运行时依赖 | langchain-core、langgraph-checkpoint、**langgraph-sdk**、langgraph-prebuilt、xxhash、pydantic | langchain-core、langgraph-checkpoint、langgraph-prebuilt、xxhash、pydantic（**去 sdk**） |
| 远程执行 | `RemoteGraph`（remote.py，依赖 sdk + httpx/websockets） | **已移除** |
| 记忆持久化后端 | 内存 / SQLite / **Postgres** | 内存 / **SQLite（本地落盘）** |
| 重量级依赖 | psycopg、websockets、httpx（直接） | **均已去除直接依赖** |
| 服务端 / 部署 | CLI + dev server + Server SDK | **无**（仅本地嵌入式运行） |
| 测试 | 各库庞大单测 + docker compose（pg/redis） | 根 `tests/` 轻量验收套件（22 用例，纯本地） |
| LLM 接入 | 任意（含 Server 链路） | **远程 API 为主**（langchain-anthropic + Claude） |
| 目标场景 | 通用 / 云端 / 多机部署 | **嵌入式 Linux 机器人**：依赖少、可靠、可调试 |

### 3.3 能力对比（保留 / 弱化 / 移除）

```mermaid
flowchart LR
    subgraph GKEEP["✅ 完整保留"]
        A1["StateGraph / Pregel 编排"]
        A2["可恢复执行（checkpoint）"]
        A3["人在环 / interrupt 安全门"]
        A4["短期 + 长期记忆"]
        A5["create_react_agent / ToolNode"]
        A6["流式 stream_mode 调试"]
    end
    subgraph GWEAK["🔸 弱化 / 依赖外部"]
        B1["可观测性：去内置遥测，靠日志/stream"]
        B2["长期记忆向量检索：sqlite-vec / 远程 embedding"]
    end
    subgraph GDROP["❌ 移除"]
        C1["RemoteGraph 远程执行"]
        C2["Postgres 后端"]
        C3["CLI / dev server / 部署平台"]
        C4["Server 鉴权链路（仅留类型桩）"]
    end

    classDef keep fill:#d6f5d6,stroke:#2e7d32,color:#143d16;
    classDef weak fill:#fff7d6,stroke:#f9a825,color:#5c4a12;
    classDef drop fill:#ffd9d9,stroke:#c62828,color:#5c1212;
    class A1,A2,A3,A4,A5,A6 keep;
    class B1,B2 weak;
    class C1,C2,C3,C4 drop;
```
