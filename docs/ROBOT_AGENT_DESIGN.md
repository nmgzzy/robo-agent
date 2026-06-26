# 机器人 Agent 需求与技术总纲（基于裁剪后的 LangGraph 底座）

> 本文是**需求 + 技术总纲**：先界定「要造什么、为谁造、做到什么算成功」（第一部分 · 需求），
> 再给出「用裁剪后的 LangGraph 底座怎么造」（第二部分 · 技术总纲）。
> 配套的**分阶段实现计划**见 [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)。
>
> **职责范围**：本 Agent 负责**思考、决策、调用工具控制机器人、管理短期/长期记忆、作为长期运行的数字个体自主运转**。
> **不在范围内**：机器人的感知与底层控制实现（由外部模块提供，本 Agent 只通过「工具」调用、通过「状态」读取其结果）。
> 本文为设计交付，不含业务代码；文中代码片段均为**蓝图示意**。

---

## 文档导航

| 部分 | 章节 | 内容 |
|------|------|------|
| **一 · 需求** | §1 | 使命、场景、角色、功能性/非功能性需求、验收标准、非目标 |
| **二 · 技术总纲** | §2 | 设计约束与总体取舍 |
| | §3 | 分层架构（各层输入/输出契约） |
| | §4 | 核心闭环：思考 → 决策 → 行动 → 记忆 |
| | §5 | 状态、工具与硬件抽象层（HAL） |
| | §6 | 记忆体系（短期 Checkpoint / 长期 Store） |
| | §7 | 嵌入式可靠性 · 性能 · 可调试 |
| | §8 | 数字个体能力域（引擎层 / 治理层 / 成长层 / 运维层） |
| | §9 | 术语表与模块依赖图 |

---

# 第一部分 · 需求

## 1. 需求定义

### 1.1 使命（Mission）

在一台**嵌入式 Linux 机器人**上，运行一个**长期在线、可靠、可调试**的数字个体：
它接收外部感知输入与人类指令，自主思考与决策，通过受控的工具边界驱动机器人完成任务，
并在持续运行中**积累记忆、复盘改进、保持身份稳定**——做到「越用越懂这个环境」。

### 1.2 目标场景（Scenarios）

| # | 场景 | 说明 |
|---|------|------|
| S1 | **指令驱动任务** | 人下达自然语言指令（"把桌上的杯子拿给我"）→ Agent 规划 → 调工具控制机器人 → 反馈结果。 |
| S2 | **事件驱动响应** | 传感器/环境事件（检测到新物体、低电告警、语音唤醒）汇入收件箱 → Agent 自主决策响应。 |
| S3 | **空闲自主运转** | 无外部任务时，Agent 按空闲策略自我驱动（巡检、整理记忆、复盘、待机）。 |
| S4 | **崩溃/断电恢复** | 进程崩溃或断电后，按 `thread_id` 从落盘 checkpoint 恢复到中断点续跑。 |
| S5 | **长期演化** | 跨天/跨周运行，记忆持续巩固与治理，技能逐步沉淀，身份与偏好保持一致。 |

### 1.3 角色（Actors）

- **人类用户**：下达指令、接收反馈、在危险动作前做人工确认（人在环）。
- **外部感知模块**：产出观测（位姿、检测、电量、ASR 文本…），不在本 Agent 内实现。
- **外部控制模块**：执行意图级指令（导航到点、设速度、抓取、播报），算法（SLAM/避障/运动学/TTS）在其内部。
- **远程 LLM 服务**：Agent 的「决策大脑」，经 ChatModel 抽象访问（推荐 Claude）。
- **本 Agent（数字个体）**：思考、决策、记忆、自主调度——本文设计主体。

### 1.4 功能性需求（FR）

| ID | 需求 | 关联设计 |
|----|------|----------|
| FR-1 | 接入远程 LLM 做决策，支持模型分层（高频轻任务用小模型，复杂规划用大模型） | §4 |
| FR-2 | 以**工具**为唯一出口控制机器人，所有副作用经工具审计 | §5.2 |
| FR-3 | 经**硬件抽象层（HAL）**接入感知/执行，核心代码不依赖任何硬件 SDK，每个接口配 Mock | §5.3 |
| FR-4 | **短期记忆**：每步状态按 `thread_id` 落盘，支持崩溃恢复与 interrupt/resume | §6.1 |
| FR-5 | **长期记忆**：跨会话存取事实/经验/偏好，支持键值与（可选）语义检索 | §6.2 |
| FR-6 | 调 LLM 前注入长期记忆并裁剪历史；事后把新知识回写长期记忆 | §6.3 |
| FR-7 | **自主引擎**：常驻 driver + 收件箱 + 事件总线 + 空闲策略，让个体在「没人说话」时也能转 | §8.1 |
| FR-8 | **目标系统**：目标栈、分解、优先级、deadline、可重规划、多目标仲裁 | §8.2 |
| FR-9 | **复盘闭环**：回合/周期后自评（intended vs actual），episodic→semantic 蒸馏 | §8.3 |
| FR-10 | **记忆治理**：遗忘衰减、去重、冲突消解、重要性排序、巩固 | §8.4 |
| FR-11 | **元认知**：循环检测、卡死识别、置信度、预算感知、不确定即上报 | §8.5 |
| FR-12 | **安全/对齐策略层**：宪章硬约束、工具权限范围、危险动作限流、覆盖审计 | §8.6 |
| FR-13 | **技能库**：把成功计划存为数据、检索复用、组合新技能，配合动态工具加载 | §8.7 |
| FR-14 | **身份/自我模型**：稳定注入 persona / 价值观 / 能力自知 | §8.8 |
| FR-15 | **运维可观测**：决策日记、运行时自省、健康度指标、远程巡检 | §8.9 |

### 1.5 非功能性需求（NFR）

| ID | 维度 | 目标 |
|----|------|------|
| NFR-1 | **依赖少 / 攻击面小** | 核心依赖树保持裁剪后的最小集（见 §2）；硬件 SDK/ROS/OpenCV/控制算法只许出现在实现包内 |
| NFR-2 | **性能** | 全链路异步；LLM 网络等待不阻塞其它协程；冷启动快、内存占用低 |
| NFR-3 | **可靠性** | 崩溃/断电可恢复；节点重试 + 超时 + 降级；危险动作 interrupt 门控 |
| NFR-4 | **可调试** | `stream_mode=debug/updates/values` 可逐步观察；本地 `*.db` 可离线审查；全程无云依赖（LLM 除外） |
| NFR-5 | **可测试** | 不接真硬件即可在 Mock 下跑通闭环、可单测、可回归 |
| NFR-6 | **长期可维护** | 长跑数周不被脏记忆拖垮；体积可控（定期清理过期 checkpoint） |

### 1.6 验收标准（Acceptance Criteria）

- **AC-1（核心闭环）**：在 Mock HAL 下，给定一条指令，Agent 产出预期的工具调用序列并落盘状态。
- **AC-2（恢复）**：杀进程后用同 `thread_id` 重启，`aget_state(cfg)` 取回中断前完整状态并续跑。
- **AC-3（记忆）**：跨会话写入的事实/偏好能在新会话被检索注入并影响决策。
- **AC-4（自主）**：无外部输入时，driver 能按空闲策略自发开启一个回合并写出复盘。
- **AC-5（安全）**：危险动作在执行前被 interrupt 拦截，需规则/人工确认方可继续。
- **AC-6（治理）**：注入矛盾事实后，记忆治理能检出冲突并按策略消解，检索质量不退化。
- **AC-7（回归）**：给定一串脚本化观测，断言 Agent 下发的指令序列与期望一致（Mock 回归）。

### 1.7 非目标（Non-Goals）

- ❌ 不实现感知/定位/运动控制/SLAM/TTS 等算法（在 HAL 实现侧，不进底座）。
- ❌ 不引入数据库服务、Server/远程图链路、云端部署平台。
- ❌ 不做多机分布式编排（单机嵌入式运行）。
- ❌ 本次不交付业务代码（仅需求 + 技术总纲 + 实现计划）。

---

# 第二部分 · 技术总纲

## 2. 设计约束与总体取舍

| 约束 | 设计决策 |
|------|----------|
| 依赖少 | 已裁剪到 `langchain-core / langgraph-checkpoint / langgraph-prebuilt / pydantic / xxhash / ormsgpack / aiosqlite / sqlite-vec / orjson`。无数据库服务依赖、无 Server/远程图链路。（注：`langchain-core → langsmith` 仍传递性带入轻量 `httpx`/`requests`，详见 `SLIMMING_NOTES.md`。） |
| LLM 远程 API 为主 | 通过 `langchain-core` 的 ChatModel 抽象接入远程模型（推荐 `langchain-anthropic` + Claude）。网络/密钥都封装在该客户端，底座自身不再直接依赖 `langgraph-sdk` 的 httpx/websockets 远程链路。 |
| 性能优 | 异步底座（`aiosqlite`），LLM 网络 I/O 与机器人控制并发；冷启动快；SQLite WAL 模式。 |
| 长期可靠运行 | 短期记忆 checkpoint 落盘 → 崩溃/断电后按 `thread_id` 恢复续跑；节点重试 + 超时；interrupt 做安全/急停门控。 |
| 可调试 | `stream_mode=debug/updates/values` 逐步观察；本地 `*.db` 可用 `sqlite3` CLI 离线审查；全程无云依赖。 |

---

## 3. 分层架构

```
┌──────────────────────────────────────────────────────────────┐
│  应用层 / 数字个体策略层（§8）                                    │
│   常驻 driver · 收件箱/事件总线 · 目标系统 · 复盘 · 治理 · 安全策略 │
├──────────────────────────────────────────────────────────────┤
│  Agent 主循环（§4，本设计核心）                                   │
│   思考(LLM 决策) ─▶ 决策路由 ─▶ 工具调用(控制机器人) ─┐            │
│        ▲                                            │            │
│        └────────────── 回灌结果/记忆注入 ◀───────────┘            │
├───────────────┬───────────────────────┬──────────────────────┤
│ LLM 接入       │ 记忆管理（§6）         │ 工具层 / HAL（§5）       │
│ ChatModel(远程)│ 短期: SqliteSaver      │ ToolNode 包装机器人动作  │
│                │ 长期: SqliteStore      │（移动/抓取/查询状态…）   │
├───────────────┴───────────────────────┴──────────────────────┤
│  裁剪后的 LangGraph 引擎：StateGraph + Pregel + channels         │
├──────────────────────────────────────────────────────────────┤
│  外部（不在本 Agent 范围）：感知、定位、运动控制、硬件驱动          │
└──────────────────────────────────────────────────────────────┘
```

**各层的输入/输出契约**（自上而下，依赖倒置：上层依赖抽象，实现侧注入）：

| 层 | 输入 | 输出 | 关键契约 |
|----|------|------|----------|
| 策略层（§8） | 收件箱事件、目标栈、时钟心跳 | 「现在该开哪个回合」的决策 → `graph.ainvoke` | driver 决定**何时**唤醒；图决定**怎么**思考 |
| 主循环（§4） | `messages` + 注入的长期记忆 + 只读世界状态 | tool-call / 终态 | think→act 循环；`tools_condition` 路由 |
| 记忆（§6） | `pre_model_hook` 检索请求、回写请求 | 注入上下文 / 落盘快照 | Saver 按 `thread_id`；Store 按 namespace |
| 工具/HAL（§5） | LLM 的 tool-call（意图级指令） | 执行结果 / 观测快照 | `Actuator.execute` / `SensorSource.stream`；副作用全过工具 |
| 引擎 | 节点函数 + channels | 超步推进、重试、interrupt | Pregel 执行语义 |

---

## 4. 核心闭环：思考 → 决策 → 行动 → 记忆

### 4.1 首选：复用 `create_react_agent`

最省代码、直接满足「思考 + 决策 + 工具调用 + 记忆」四件事。位置：
`libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py`。原生支持：

- `model`：远程 ChatModel（决策大脑）
- `tools`：机器人控制工具列表 → 内部用 `ToolNode` 执行
- `checkpointer`：短期记忆（线程内状态、可恢复）
- `store`：长期记忆（跨会话）
- `pre_model_hook` / `post_model_hook`：**记忆注入与裁剪 / 安全门控**的挂载点（见 §6、§8）

> 注：`langgraph.prebuilt.create_react_agent` 已被上游标注 deprecated（迁移到 `langchain.agents.create_agent`，属更重的 `langchain` 包）。
> 本瘦身构建**不含 `langchain`**，继续用 `langgraph.prebuilt` 版本可避免新增依赖（仅有一条 DeprecationWarning，可按需静音）；
> 若想完全规避弃用项，直接用 §4.2 的自定义 `StateGraph` 即可——能力等价。

蓝图：

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
# from langchain_anthropic import ChatAnthropic   # 远程 LLM 客户端（用户另装）

agent = create_react_agent(
    model=ChatAnthropic(model="claude-opus-4-8"),  # 决策；高频轻量任务可用 claude-haiku-4-5
    tools=[move_to, grasp, get_world_state],        # 机器人控制动作（占位，见 §5）
    checkpointer=saver,      # 短期记忆
    store=store,             # 长期记忆
    pre_model_hook=inject_memory,    # 调 LLM 前注入长期记忆 + 裁剪历史
    # post_model_hook=safety_gate,   # 决策后安全门控（可选，见 §8.6）
)
out = await agent.ainvoke(
    {"messages": [("user", "把桌上的杯子拿给我")]},
    {"configurable": {"thread_id": "episode-42"}},
)
```

### 4.2 需要更强控制时：下沉到 `StateGraph`

当需要自定义状态通道（并发感知输入、世界模型）、显式安全门控节点、或非 ReAct 的循环结构时，用
`StateGraph`（`libs/langgraph/langgraph/graph/state.py`）手写：

```
START ─▶ think(LLM) ─▶ [tools_condition] ─▶ act(ToolNode) ─▶ think ...
                              └─▶ END
```

`tools_condition` / `ToolNode` 均来自 `libs/prebuilt`。两种方案可平滑切换，建议从 4.1 起步。

### 4.3 闭环时序（一次回合）

```
driver 唤醒(§8.1)
  └─▶ ainvoke(thread_id)
        ├─ pre_model_hook：检索长期记忆 + 裁剪 messages（§6.3）
        ├─ think：LLM 决策 → tool-call 或终态
        ├─ [tools_condition] 路由
        ├─ act：ToolNode 执行 → Actuator 下发（§5.2/5.3）→ 结果回灌 messages
        ├─ （每步）Saver 落盘快照（§6.1）
        └─ 循环直到终态
  └─▶ post：复盘 + 回写长期记忆（§6.3 / §8.3）
driver 睡下，等下一个事件/心跳
```

---

## 5. 状态、工具与硬件抽象层（HAL）

### 5.1 状态（State）

- `messages`：对话/推理历史 = **工作记忆（短期）**，用 `add_messages` reducer 累积。
- **注入式只读世界状态**：机器人感知结果（位姿、检测到的物体、电量等）由外部写入 State 的自定义字段；本 Agent 只读，不管它怎么来的。

### 5.2 工具（控制机器人的唯一出口）

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
- 危险动作建议配合 `interrupt`（见 §7）做执行前人工/规则确认。

### 5.3 硬件抽象层（HAL）：感知输入 / 执行输出 / Mock

本 Agent 的输入输出本质上都与硬件绑定——**输入**有传感器、环境/定位、摄像头、语音；
**输出**有底盘速度、目标位置、扬声器等。设计目标是让**底座实现完全不依赖具体硬件与控制算法**：
核心代码永不 `import` 任何硬件 SDK / ROS / OpenCV / 运动学/PID 库，一切经**抽象接口 + 可插拔插件**
接入，并对每个接口提供 **Mock 实现**，使得不接真硬件也能跑通闭环、可单测、可回归。

依赖方向遵循**依赖倒置**：`核心(只依赖接口) → 接口 ← 实现(real / sim / mock)`。

#### 5.3.1 接口契约一览

| 接口 | 方向 | 方法 | 产出/入参 | 实现侧承载 |
|------|------|------|-----------|-----------|
| `SensorSource` | 输入 | `stream() -> AsyncIterator[Observation]` 或 `poll()` | `Observation`（带 ts/frame/payload） | 传感器驱动、ASR、定位 |
| `Actuator` | 输出 | `async execute(command: dict) -> dict` | 意图级指令 → 执行结果/句柄 | 导航、运动学、避障、TTS |

#### 5.3.2 输入侧：感知源插件（SensorSource）

```python
from typing import Protocol, AsyncIterator
from pydantic import BaseModel

class Observation(BaseModel):
    source: str               # "camera" / "lidar" / "asr" / "battery" / "pose" ...
    ts: float                 # 单调时钟时间戳
    frame: str | None = None  # 坐标系（如适用）
    payload: dict             # 该来源的结构化数据：图像句柄 / 检测框 / ASR 文本 / 距离 / 位姿…

class SensorSource(Protocol):
    name: str
    def stream(self) -> AsyncIterator[Observation]: ...   # 流式；或拉取式 poll()->Observation|None
```

观测有两条喂入路径，按性质二选一：

- **事件型**（语音指令、检测到新物体、低电告警）→ 汇入 driver 的**事件总线/收件箱**（见 §8.1），驱动决策。
- **连续只读量**（位姿、电量、最近距离）→ 以**快照**写入 State 的世界状态字段（见 §5.1），LLM 只读。

#### 5.3.3 输出侧：执行器插件（Actuator）

所有动作的唯一落点抽象为「执行器」，**经 §5.2 的 `ToolNode` 暴露为工具**，工具只持接口引用、不关心实现：

```python
class Actuator(Protocol):
    name: str
    async def execute(self, command: dict) -> dict: ...   # 返回执行结果/句柄

@tool
async def set_velocity(vx: float, wz: float) -> str:
    """设置底盘线速度 vx(m/s)、角速度 wz(rad/s)。"""
    # Actuator.execute 是 async，工具须 async def + await，否则只是返回未执行的协程
    return await effectors["base"].execute({"vx": vx, "wz": wz})   # 经接口下发

@tool
async def speak(text: str) -> str:
    """通过扬声器播报文本。"""
    return await effectors["speaker"].execute({"text": text})
```

**控制算法的归属**：底座只下发**意图/指令级**输出（去某坐标、设某速度、播报某文本）；
路径规划、避障、运动学解算、SLAM、TTS 合成等**全部在接口背后的实现里**，不进底座。

#### 5.3.4 三档实现与 Mock

每个接口至少配三档实现，通过**注册表/配置**选择，核心代码与图结构不变：

| 档位 | 用途 | 依赖 |
|------|------|------|
| `real` | 接真硬件（ROS 桥 / 厂商 SDK / 设备驱动） | 仅在该实现包内引入，**不进核心依赖树** |
| `sim` | 接仿真器（可选，如 Gazebo/Isaac） | 同上，仅实现侧 |
| `mock` | 纯内存：脚本化产生假观测、记录收到的指令 | 无外部依赖，**单测/CI 默认** |

```python
class MockBase:                      # Actuator 的 mock：只记录，不动真硬件
    name = "base"
    def __init__(self): self.log = []
    async def execute(self, command): self.log.append(command); return {"ok": True}

class ScriptedCamera:                # SensorSource 的 mock：按脚本回放观测
    name = "camera"
    def __init__(self, frames): self._frames = frames
    async def stream(self):
        for f in self._frames:
            yield Observation(source="camera", ts=f["ts"], payload=f)
```

**依赖纪律**：硬件 SDK / ROS / OpenCV / 控制算法只允许出现在 `plugins/<impl>` 这类实现包内；
核心库（裁剪后的四个 lib）依赖树保持干净——这与 `SLIMMING_NOTES.md` 的「依赖少、攻击面小」一致。

#### 5.3.5 与可靠性 / 安全的衔接

- 接真硬件不可用时，接口应抛出明确错误 → 触发 §7 的**降级策略**（停在原地/保守等待）。
- 危险执行器动作（高速、抓取）在工具内配合 §7 的 `interrupt` 做执行前确认。
- mock 的「指令记录」与「脚本观测」天然适合做**回归测试**：给定一串观测，断言 Agent 下发的指令序列。

---

## 6. 记忆体系

### 6.1 短期记忆（线程内 / 单次任务）— Checkpoint

- 实现：`AsyncSqliteSaver`（`libs/checkpoint-sqlite/.../checkpoint/sqlite/aio.py`），本地 `agent.db`。
- 语义：每一步状态快照按 `thread_id` 落盘。一个机器人「任务/回合」用一个 `thread_id`。
- 价值：进程崩溃/断电重启后，用同 `thread_id` 即可恢复到中断点继续（durable execution）；天然支持 interrupt/resume。
- 已验证：重启新连接后 `aget_state(cfg)` 能取回中断前的完整状态。

### 6.2 长期记忆（跨任务 / 跨会话）— Store

- 实现：`AsyncSqliteStore`（`libs/checkpoint-sqlite/.../store/sqlite/aio.py`）。接口见 `libs/checkpoint/langgraph/store/base/__init__.py`。
- 命名空间组织建议（随 §8 扩展）：

  | namespace | 内容 | 引入于 |
  |-----------|------|--------|
  | `(robot_id, "facts")` | 环境/任务事实（如「充电桩在 (3,2)」） | §6 |
  | `(robot_id, "episodic")` | 历史经验（做过什么、结果如何） | §6 |
  | `(robot_id, "prefs")` | 用户偏好 | §6 |
  | `(robot_id, "goals")` | 目标/意图栈 | §8.2 |
  | `(robot_id, "skills")` | 技能库（数据化的成功计划） | §8.7 |
  | `(robot_id, "identity")` | persona / 价值观 / 能力自知 | §8.8 |

- 检索：
  - 结构化/键值检索：`aget` / `asearch`（无额外算力开销，嵌入式首选）。
  - 语义检索（可选）：远程 embedding 写入 + `sqlite-vec` 本地向量检索；算力紧张时关闭，退回关键字。

### 6.3 记忆与主循环的衔接

- `pre_model_hook`：调 LLM 前，从 Store 检索相关长期记忆拼进上下文；同时**裁剪/摘要**过长的 `messages`，控制 token 与时延（嵌入式关键）。
- `post_model_hook` 或工具内：把新学到的事实/经验写回 Store，实现「越用越懂这个环境」。
- 这是机制；**复盘闭环**（什么时候蒸馏、怎么巩固）属策略层，见 §8.3。

---

## 7. 嵌入式可靠性 / 性能 / 可调试

**可靠性**

- 崩溃恢复：SQLite 落盘 checkpoint（§6.1）。
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

## 8. 数字个体能力域

§4–§7 给的几乎都是**机制**（记忆落盘、工具边界、重试、interrupt）。要把本 Agent 当作一个
**长期运行的数字个体**——有身份、记忆、日程、目标、工具、复盘、偏好、技能库——还缺两样：

1. 一个让它**自己转起来**的外层引擎（当前底座是纯被动：必须有人 `ainvoke` 才动一下）；
2. 把上述机制**编排成「个体」的策略层**（架构图最上层目前基本是空的）。

下面把能力按四个域组织：**引擎层**（让它活着）、**治理层**（让它长期不烂）、**成长层**（让它成长）、**运维层**（无人值守监护）。每项给出：职责、底座挂载点、数据模型/接口草图。

### 8.0 能力域总览

| 能力 | 域 | 现状 | 待补的需求 | 底座挂载点 |
|------|----|------|-----------|-----------|
| 自主心跳 / 事件总线（§8.1） | 引擎 | 无，纯被动 `ainvoke` | 常驻 driver + 统一收件箱 + 空闲策略 | 新建应用层常驻进程，循环调 `graph.ainvoke` |
| 目标系统（§8.2） | 引擎 | 无「意图」一等表示 | 目标栈 + 规划 + 多目标仲裁 | Store `(robot_id,"goals")` + 规划节点 |
| 复盘闭环（§8.3） | 引擎 | 仅 §6.3 一个挂载点 | 周期/回合后自评、episodic→semantic 蒸馏 | 反思子图，由 driver 定时触发 / `post_model_hook` |
| 记忆治理（§8.4） | 治理 | 仅清理 checkpoint 体积 | 遗忘衰减、去重、冲突消解、重要性排序、巩固 | Store 之上的后台 compaction 任务 |
| 元认知 / 自我监控（§8.5） | 治理 | 仅节点级 retry/timeout | 循环检测、卡死识别、预算感知、不确定即上报 | `pre_model_hook` + 监控指标 |
| 安全 / 对齐策略层（§8.6） | 治理 | 仅 `interrupt` 机制 | 宪章硬约束、工具权限范围、危险动作限流、覆盖审计 | `post_model_hook` + `interrupt` + 工具封装 |
| 技能库（§8.7） | 成长 | 工具为启动时硬编码 | 把成功计划存为数据、检索复用、组合新技能 | Store `(robot_id,"skills")` + 动态工具加载 |
| 身份 / 自我模型（§8.8） | 成长 | 隐含在 prompt | 稳定注入的 persona / 价值观 / 能力自知 | Store `(robot_id,"identity")` + `pre_model_hook` 注入 |
| 运维可观测（§8.9） | 运维 | `stream_mode=debug`（开发期） | 决策日记、运行时自省、健康度、远程巡检 | 工具/hook 结构化打点 + 指标导出 |

### 8.1 自主心跳 + 事件总线（引擎，最高优先级）

**职责**：当前没有任何东西能在「没人说话」时唤醒它。需要一个外层常驻 driver：把定时器/cron、外部消息、传感事件统一汇进一个**收件箱**，由常驻循环消费；并定义**空闲策略**（无任务时做什么——巡检 / 整理记忆 / 待机）。`thread_id` 能恢复一个回合，但「现在该开哪个回合」需由 driver 决定。**这是被动库 → 自主个体的分界线。**

**接口草图**：

```python
class Event(BaseModel):
    kind: str          # "user_msg" / "sensor" / "timer" / "goal_due" ...
    ts: float
    payload: dict
    priority: int = 0

class Inbox(Protocol):
    async def put(self, e: Event) -> None: ...
    async def get(self, timeout: float | None) -> Event | None: ...

# 常驻 driver 主循环（伪码）
async def run_driver(graph, inbox, goals, policy):
    while True:
        e = await inbox.get(timeout=policy.idle_tick)   # 有事件则醒，超时则进入空闲策略
        thread_id = decide_thread(e, goals)             # 决定开哪个回合
        await graph.ainvoke(make_input(e), {"configurable": {"thread_id": thread_id}})
        await reflect_if_due(graph, goals)              # 空闲/回合后触发复盘（§8.3）
```

**底座挂载点**：新建应用层常驻进程，循环调 `graph.ainvoke`；事件型观测（§5.3.2）汇入 `Inbox`。

### 8.2 目标系统（引擎）

**职责**：Store 现存 facts/episodic/prefs，**没有「目标/意图」的结构化表示**。需要：长期目标 → 分解 → 当前任务 → 子任务，带优先级、deadline、状态、可重规划；以及**仲裁**——多目标竞争选哪个、被紧急事件打断后如何回到原目标。`messages` 是工作记忆，扛不住跨回合目标。

**数据模型草图**：

```python
class Goal(BaseModel):
    id: str
    parent: str | None        # 目标树
    intent: str               # 自然语言意图
    priority: int
    deadline: float | None
    status: str               # pending / active / blocked / done / abandoned
    plan: list[str] = []      # 分解出的子任务/步骤
```

**底座挂载点**：Store `(robot_id,"goals")` 持久化目标栈 + 一个**规划节点**（StateGraph 内）负责分解与重规划；driver 在空闲时做多目标仲裁。

### 8.3 复盘闭环（引擎）

**职责**：§6.3 只给了挂载点，不是机制。需要回合结束/周期触发的自评（intended vs actual）、把 episodic 蒸馏为 semantic、据此更新偏好与技能。无此闭环，「越用越懂」只是愿望。

**底座挂载点**：一个**反思子图**，由 driver 定时触发或挂 `post_model_hook`；读 `episodic`，写 `facts`/`prefs`/`skills`。

### 8.4 记忆治理（治理，长跑隐形杀手）

**职责**：§7 只管 checkpoint 体积（短期记忆运维）。语义记忆（Store）需要：遗忘/衰减、去重、**冲突消解**（两条矛盾 fact 怎么办）、重要性排序、巩固。否则跑几周后 Store 又大又脏又自相矛盾，检索质量崩塌——这是这类系统最常见的死法。

**底座挂载点**：Store 之上的**后台 compaction 任务**（由 driver 周期调度），对每个 namespace 做衰减/去重/冲突消解/重要性重排。

### 8.5 元认知 / 自我监控（治理）

**职责**：retry/timeout 是节点级容错，不是对自身认知状态的感知。需要：循环检测、卡死识别、置信度、预算感知（token / 时延 / 电量）、**不确定即上报求助**（escalation）。

**底座挂载点**：`pre_model_hook` 内做循环/预算检查 + 监控指标导出；越界则触发 interrupt 上报。

### 8.6 安全 / 对齐策略层（治理）

**职责**：`interrupt` 只是「能暂停」的机制。需要策略层：宪章式硬约束、按工具的权限范围、危险动作限流、人工覆盖审计。**物理机器人**不能只靠 prompt 里写「小心点」。

**底座挂载点**：`post_model_hook` 校验决策 + 工具封装层做权限/限流 + `interrupt` 做硬门控 + 结构化审计日志。

### 8.7 技能库（成长，作为数据而非代码）

**职责**：现在工具是启动时硬编码的 Python，运行时不能自我扩展。真正的技能库是把成功的计划/参数化过程**存成数据**、检索复用、组合成新技能（Voyager 思路），配合动态工具加载。这是「成长」的载体。

**底座挂载点**：Store `(robot_id,"skills")` 存技能定义 + 动态工具加载（运行时把技能装配进 `tools`）。

### 8.8 身份 / 自我模型（成长，零成本先立）

**职责**：现在身份只隐含在 prompt，`robot_id` 只是命名空间前缀。需要稳定注入的「我是谁」——persona、价值观、能力自知（擅长/不擅长什么）。长跑下没有这个锚，行为会漂移。成本几乎为零（一段稳定的 system context），建议最先立起来，给后续所有决策一个锚点。

**底座挂载点**：Store `(robot_id,"identity")` 存身份 + `pre_model_hook` 每次注入为 system context。

### 8.9 运维 / 可观测（运维）

**职责**：决策日记 / 审计、运行时自省（「它现在在干嘛、为什么」）、健康度指标、远程巡检。`stream_mode=debug` 面向开发期调试，不等于对一个常驻个体的生产期监护。

**底座挂载点**：工具/hook 结构化打点 + 指标导出 + 决策日记落盘（可复用 Store 或独立 db）。

### 8.10 落地优先级（指导实现计划）

1. **先做 §8.1 的引擎**——没有它，前面所有「家具」仍是被动调用的库，不是个体。
   最小闭环：**外层常驻 driver + 收件箱 + 目标栈 + 空闲时触发复盘**（醒来 → 看收件箱和目标 → 决定干啥 → 干完写复盘 → 睡下）。
2. **紧接着补 §8.4 的记忆治理**，否则引擎跑两周就被脏记忆拖垮。
3. **顺手先立 §8.8 的身份**（零成本，给所有决策一个锚点）。
4. §8.7 技能库、§8.9 运维可观测可滞后于上述三项。

> 详细的阶段拆分、交付物与验收方式见 [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)。

---

## 9. 术语表与模块依赖图

### 9.1 术语表

| 术语 | 含义 |
|------|------|
| **底座** | 裁剪后的四个 LangGraph 库（checkpoint / checkpoint-sqlite / langgraph / prebuilt）。 |
| **回合（episode）** | 一次任务/会话，对应一个 `thread_id` 下的完整 think→act 循环。 |
| **HAL** | 硬件抽象层：`SensorSource`（输入）/ `Actuator`（输出）两个 Protocol + 三档实现。 |
| **意图级指令** | 底座下发的高层命令（去某点 / 设某速度 / 播报），不含路径/运动学算法。 |
| **观测（Observation）** | 感知源产出的结构化数据，带时间戳/坐标系。 |
| **短期记忆 / Checkpoint** | 按 `thread_id` 落盘的状态快照，支持恢复与 interrupt/resume。 |
| **长期记忆 / Store** | 按 namespace 组织的跨会话知识（facts/episodic/prefs/goals/skills/identity）。 |
| **driver** | 应用层常驻进程，决定「何时唤醒、开哪个回合」。 |
| **收件箱 / 事件总线** | 汇聚定时器/外部消息/传感事件的统一队列。 |
| **复盘闭环** | 回合/周期后的自评 + episodic→semantic 蒸馏。 |
| **记忆治理** | 长期记忆的遗忘/去重/冲突消解/重要性/巩固。 |

### 9.2 模块依赖图（核心 + 计划新增）

```
                  ┌─────────────────────────────────────┐
                  │ 应用层 / 数字个体（计划新增, §8）       │
                  │  driver · inbox · goals · reflect ·   │
                  │  governance · safety · skills · ops   │
                  └───────────────┬─────────────────────┘
                                  │ 依赖（仅经接口/hook）
        ┌─────────────────────────┼──────────────────────────┐
        ▼                         ▼                          ▼
   ┌──────────┐            ┌──────────────┐           ┌──────────────┐
   │ prebuilt │            │  langgraph   │           │  HAL 接口     │
   │ react/   │──依赖──────▶│ StateGraph/  │           │ Sensor/      │
   │ ToolNode │            │ Pregel       │           │ Actuator     │
   └────┬─────┘            └──────┬───────┘           └──────┬───────┘
        │                        │                          │ 实现注入
        ▼                        ▼                   ┌───────┴────────┐
   ┌──────────┐           ┌──────────────┐           ▼       ▼        ▼
   │checkpoint│◀──实现────│checkpoint-   │        real    sim     mock
   │(接口)    │           │sqlite(落盘)  │       (硬件SDK/ROS, 不进核心)
   └──────────┘           └──────────────┘
```

> **依赖纪律**：应用层与 HAL 实现包可以引入硬件/控制依赖；**裁剪后的四个核心库依赖树保持干净**（见 `SLIMMING_NOTES.md`、`STRUCTURE.md`）。
