# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 本仓库是一个 monorepo，从 LangGraph 官方仓库**裁剪**而来的「嵌入式机器人 Agent 底座」，
> 面向在嵌入式 Linux 上长时运行的机器人 Agent（依赖少、可靠、可调试、远程 LLM 推理）。
> 背景与裁剪细节见 `docs/SLIMMING_NOTES.md`；架构设计见 `docs/ROBOT_AGENT_DESIGN.md`；
> 分阶段实现计划（P0–P10）见 `docs/IMPLEMENTATION_PLAN.md`；结构对比见 `docs/STRUCTURE.md`。

## 常用命令

```bash
# 安装：4 个本地库 editable 装入虚拟环境
uv venv && source .venv/bin/activate
uv pip install -e libs/checkpoint -e libs/checkpoint-sqlite -e libs/prebuilt -e libs/langgraph
# 或：make install（遍历 libs/* 安装）

make test                              # 根目录跑全部验收/回归（= pytest tests/）
TEST=tests/test_memory.py make test    # 只跑某个文件；TEST 可附加任意 pytest 参数
uv run --active pytest tests/test_robot_agent_p1.py -k recall   # 直接 pytest 也可

# 改了任意库代码、建 PR 前，在该库目录下跑：
make -C libs/<lib> format    # 代码格式化（ruff）
make -C libs/<lib> lint      # 静态检查（ruff + mypy）
make format && make lint     # 根目录：对所有库批量执行
```

测试全部**离线**运行（内存 / 临时 SQLite，无远程 LLM、无外部服务）。`pytest.ini` 已配
`pythonpath=.`（应用层 `robot_agent/` 免打包直接 import）与 `asyncio_mode=auto`。

## 两层结构：底座库 vs 应用层

```text
robot_agent/        ← 应用层（数字个体策略层），只经接口/hook 依赖下面 4 库
└── libs/           ← 裁剪后的 LangGraph 底座（被动框架）
    ├── checkpoint           记忆基础接口 + 内存实现（BaseCheckpointSaver / BaseStore）
    ├── checkpoint-sqlite    SQLite 落盘：SqliteSaver（短期/可恢复）+ SqliteStore（长期/跨会话+向量）
    ├── langgraph            核心引擎：StateGraph + Pregel 执行循环 + channels + stream
    └── prebuilt             高层 API：create_react_agent / ToolNode / tools_condition
```

库依赖（下游 = 依赖它的库），改动会沿箭头向下传播：

```text
checkpoint → {checkpoint-sqlite, prebuilt, langgraph}
prebuilt   → langgraph
```

> 上游被删的外围库：`cli`、`sdk-py`、`sdk-js`、`checkpoint-postgres`、`checkpoint-conformance`；
> core 内部也删了 `pregel/remote.py` 等远程执行链路。**不要**尝试 import 或恢复这些。

## 应用层 `robot_agent/`：从核心闭环到自主个体

`build_robot_agent()`（`graph.py`）用 `create_react_agent` 把核心件装配成最小可运行闭环
（参数：`model` / `effectors` / `checkpointer` / `store` / `pre_model_hook` / 可选 `safety`）。
各文件对齐 `docs/ROBOT_AGENT_DESIGN.md` 小节号（注释标了 §）；分阶段见 `IMPLEMENTATION_PLAN.md`。

**核心闭环「思考 → 决策 → 行动 → 记忆」(P0–P1)**

- **`llm.py`** — `make_model(profile)` 工厂。`profile ∈ {fast(haiku), smart(opus), mock}`；
  `MockChatModel` 按预设 `AIMessage` 序列确定性回放，驱动整条工具往返。
- **`state.py`** — `RobotState(AgentState)`：`messages`（短期工作记忆）+ 只读世界状态
  `pose/battery/detections`（由**外部**感知源快照注入，Agent 只读）。
- **`hal/`** — 硬件抽象层。`interfaces.py` 定义 `SensorSource`/`Actuator` **Protocol**（鸭子类型，
  无需继承）；`mock.py` 是纯内存实现；`registry.py` 的 `build_effectors(tier)` 按档位装配，
  P1 仅 `mock`，`real`/`sim` 留给 `hal/plugins/<impl>`。
- **`tools.py`** — `build_robot_tools(effectors)`：把执行器包成 `@tool`
  （`move_to/set_velocity/grasp/speak` + 只读 `get_world_state`）。所有副作用都过工具。
- **`memory.py`** — 长期记忆 namespace `(robot_id, kind)`，kind ∈ `{facts, episodic, prefs}`；
  `pre_model_hook` 调 LLM 前注入长期记忆 + `trim_messages` 裁剪历史；
  `remember_fact`/`recall` 工具经 `InjectedStore` 回写/读取（仅在配了 `store` 时才挂载）。

**自主个体能力域 (P2–P10)**

- **`reliability.py`** (P2) — `ResilientChatModel` 给决策大脑加重试/超时/降级（按类名识别
  provider 瞬态异常；重试耗尽返回**无工具调用**的保守回复 = 停在原地）；`cleanup_threads`
  清理过期 checkpoint。用 `make_resilient(model)` 包装后传入 `build_robot_agent`。
- **`safety.py`** (P2) — 危险动作下发前经 `interrupt` 门控，三级决策：`reject_reason` 硬拒绝
  （非有限速度，不可批准）＞ `danger_reason` 需确认（高速/抓取，`Command(resume)`，**fail-closed**
  只认显式 `True`）＞ 放行。需配 `checkpointer`。
- **`identity.py`** (P3) — 身份 namespace `(robot_id,"identity")`：persona/价值观/能力自知；
  `pre_model_hook` 把身份作为**稳定 system 锚点**注入到所有 system 块最前（先于动态记忆）。
- **`driver/`** (P4) — 自主引擎（**被动库 → 个体的分界线**）：`Event`/`Inbox`/`PriorityInbox`
  收件箱（优先级+超时）、`IdlePolicy`（`StandbyPolicy` 待机 / `PromptIdlePolicy` 自发回合）、
  `Driver` 常驻循环（`run`/`run_once`/`submit`）。被 `safety` 暂停的线程只接受显式 `resume`
  事件（`resume_event`）续跑，不灌入新消息。
- **`goals/`** (P5) — 目标系统：`Goal`（priority/deadline/status/plan，时间戳用 **UTC epoch**）、
  `GoalStore`（namespace `(robot_id,"goals")` CRUD，list 分页拉满）、`arbitrate`
  （priority＞deadline＞created_ts 仲裁）、`plan_goal`（意图→步骤分解）、`GoalDrivenIdlePolicy`
  （driver 空闲时推进目标栈，紧急事件可抢占、处理完恢复到被打断目标）。配 `planner_model` 时，
  目标首次推进会自动 `plan_goal` 分解、持久化 `plan` 并注入回合——闭合「分解→逐步执行」。
- **`reflect/`** (P6) — 复盘闭环：`Episode`/`record_episode`/`episode_from_turn` 把回合经历
  （intent→actions→outcome）写入 `episodic`；`reflect_and_distill` 读 episodic、LLM 蒸馏为
  `facts`/`prefs` 写回；`make_reflect_hook` 挂 driver `on_turn` 自动记录 + 周期蒸馏。蒸馏出的
  偏好经 `pre_model_hook` 在后续回合自动注入（「越用越懂」从愿望变机制）。
- **`governance/`** (P7+P9) — 治理层：① 记忆 compaction（`compact_namespace`/`compact_all`/
  `make_compaction_hook`——去重 + LLM 冲突消解（仅 facts/prefs）+ 衰减，AC-6）；② 安全/对齐
  策略层（`GovernancePolicy`：宪章硬约束 + 工具权限 + 限幅 + 限频，违反**直接拒绝** + `AuditLog`
  审计，在工具封装层执行）。
- **`metacog/`** (P8) — 元认知/自我监控：`detect_loop`/`steps_used` + `MetacogPolicy`/
  `make_monitor_hook` 装饰 `pre_model_hook`，循环/预算越界则 escalate（`interrupt` 上报）或
  warn（注入告警收敛）；`metrics` 导出。经 `build_robot_agent(..., metacog=...)` 接入。
- **`skills/`** (P10) — 技能库（技能即数据）：`Skill`（动作序列）+ `SkillStore`（持久化 + 检索）
  + `build_skill_tools`（动态装配为 `skill_<name>` 工具，可选过治理校验），经
  `build_robot_agent(..., extra_tools=...)` 运行时加载复用。
- **`ops/`** (P10) — 运维可观测：`DecisionJournal`/`make_journal_hook`（决策日记，`replay`
  离线还原决策链）+ `introspect`（运行时自省）+ `HealthReport`/`collect_health`（健康度聚合导出）。

## 跨切面纪律（务必遵守）

- **依赖纪律**：硬件 SDK / ROS / OpenCV / 控制算法只允许出现在 `hal/plugins/<impl>` 实现包内，
  **不进**核心四库依赖树。远程 LLM 客户端 `langchain-anthropic` 也只在请求真实档位时**惰性 import**。
- **Mock 优先**：不接真实 LLM / 真硬件即可离线跑通闭环与回归。新增功能优先保证 `mock` 路径可测，
  断言执行器 `.log` 即可验证行为。
- **闭环是 async-only**：执行器 `execute`、工具、`pre_model_hook`、记忆 hook 全是 `async`，
  入口用 `ainvoke`。下发动作的工具必须 `async def` + `await`，否则只是返回未执行的协程。
- **docstring / 注释**里引用行内代码用单反引号（`` `code` ``），**不要**用 Sphinx 双反引号（`` ``code`` ``）。
- **`CLAUDE.md` 与 `AGENTS.md` 内容保持一致**（两文件目前互为副本，改一个要同步另一个）。
