# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 本仓库是一个 monorepo，从 LangGraph 官方仓库**裁剪**而来的「嵌入式机器人 Agent 底座」，
> 面向在嵌入式 Linux 上长时运行的机器人 Agent（依赖少、可靠、可调试、远程 LLM 推理）。
> 背景与裁剪细节见 `docs/SLIMMING_NOTES.md`；架构设计见 `docs/ROBOT_AGENT_DESIGN.md`；
> 分阶段实现计划（P0–P10）见 `docs/IMPLEMENTATION_PLAN.md`；结构对比见 `docs/STRUCTURE.md`。

## 常用命令

整个项目由 **uv workspace** 管理：根 `pyproject.toml` 同时是 workspace 根与应用层
`robot_agent` 的包定义，4 个本地库（`libs/*`）为 workspace 成员，共用单一 `uv.lock`。

```bash
# 安装（PC 开发，uv 为主）：一键同步 4 库 editable + robot_agent + 开发依赖
make install                # = uv sync
make install-all            # 额外带上远程客户端 extra（openai + anthropic）

# 嵌入式无 uv 回退：用系统 Python 的 pip 按拓扑序装本地库再装应用层
make install-pip            # 在已激活的目标 venv 内执行（可选 EXTRAS=".[all]"）

# 配置 LLM：复制模板并填写密钥/端点/模型
cp .env.example .env

make test                              # 根目录跑全部验收/回归（= uv run pytest tests/）
TEST="tests/test_memory.py -k recall" make test   # 只跑某文件/附加任意 pytest 参数
uv run pytest tests/test_robot_agent_p1.py -k recall   # 直接 pytest 也可

# 改了任意代码、建 PR 前（整仓一次，ruff 覆盖含 libs/*，各目录就近读其 pyproject 配置）：
make format    # uv run ruff format . && ruff check --fix .
make lint      # uv run ruff check .
# 改了 libs/<lib> 且要类型检查（ty）时，再单独跑该库的 lint（含 ruff + ty）：
make -C libs/<lib> lint
```

远程 LLM 客户端是按需 extra（`uv sync --extra openai` / `--extra anthropic` / `--extra all`），
保持惰性 import、核心精简。测试全部**离线**运行（内存 / 临时 SQLite，无远程 LLM、无外部
服务）。`pytest.ini` 已配 `pythonpath=.` 与 `asyncio_mode=auto`。uv workspace 迁移设计见
`docs/superpowers/specs/2026-06-30-uv-workspace-migration-design.md`。

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

- **`llm.py`** — `make_model(profile)` 工厂。`profile ∈ {fast, smart, vision, mock}`；真实模型经
  仓库根 `.env`（模板 `.env.example`）/ `LLM_*` 环境变量或显式参数配置 `provider` /
  `base_url` / `api_key` / 模型名（默认 `openai` 兼容 API，`anthropic` 亦支持）；
  `MockChatModel` 按预设 `AIMessage` 序列确定性回放。
- **`state.py`** — `RobotState(AgentState)`：`messages`（近期原文）+ `context_summary`
  （较老会话的滚动摘要）+ 只读世界状态 `pose/battery/detections`（由**外部**感知源快照注入）。
- **`hal/`** — 硬件抽象层。`interfaces.py` 定义 `SensorSource`/`Actuator` **Protocol**（鸭子类型，
  无需继承）；`mock.py` 是纯内存实现；`registry.py` 的 `build_effectors(tier)` 按档位装配，
  P1 仅 `mock`，`real`/`sim` 留给 `hal/plugins/<impl>`。
- **`tools.py`** — `build_robot_tools(effectors)`：把执行器包成 `@tool`
  （`move_to/set_velocity/grasp/speak` + 只读 `get_world_state`）。所有副作用都过工具。
- **`context.py` / `memory.py`** — 中短期记忆在 token 高水位时用模型增量摘要已完成的较老
  回合，把摘要、压缩次数和最近原文写入 checkpoint；摘要失败只累计失败计数并临时硬裁剪，
  不覆盖原消息和摘要。五项 token 限额由根 `.env` 的 `CONTEXT_*_TOKENS` 配置，显式
  `ContextPolicy` 优先。
  长期记忆 namespace `(robot_id, kind)`，kind ∈ `{facts, episodic, prefs}`，由同一
  `pre_model_hook` 在调用 LLM 前注入；
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
- **`vision/`** — 内置 VLM：`make_model("vision")` 构建多模态模型；HAL / 插件通过
  `VisionSource` 供图，主模型只传不透明 `image_ref`（原图不进消息/checkpoint）；
  图片进入 VLM 前自动限制到 720p 并压缩过高质量编码；`describe_image` 经
  `build_robot_agent(..., vlm_model=..., vision_source=...)` 挂载。
- **`ops/`** (P10) — 运维可观测：`DecisionJournal`/`make_journal_hook`（决策日记，`replay`
  离线还原决策链）+ `introspect`（运行时自省）+ `HealthReport`/`collect_health`
  （健康度聚合导出，含会话压缩次数/失败/归档量，可读 Mapping 或图状态快照）。
- **`prompts/`** — 提示词集中管理（JSON 索引 + Markdown 正文）：全部 LLM 可见文案（身份锚点 /
  长短期记忆 / 会话摘要 / 开回合指令 / 元认知告警 + 目标分解 / 复盘蒸馏 /
  视觉理解与信任边界）外置到
  `registry.json`（索引：`file`/`params`/identity 的 `default_data`）+ 各自 `<id>.md`（正文）。
  loader 启动即加载并缓存、占位符与 `params` 不一致即 fail-fast；只暴露 `render(id, **params)`
  与 `identity_default()`。设计见 `docs/superpowers/specs/2026-06-27-prompts-central-management-design.md`。
- **`frontends/`** — 前端通道层（**可拔插接口**）：把常驻 agent 接到外部世界的稳定边界。
  天然分界线是 `Driver` 的 `submit(Event)`（输入）/ `on_turn(TurnResult)`（输出）——Web / IM /
  麦克风扬声器都只是这条边界上不同的「通道」。`service.py` 的 `AgentService` 是**通道无关门面**：
  `submit_user_text`/`resume`（输入）+ `subscribe`（把回合精简后**广播** fan-out 给所有订阅者）+
  只读视图 `history`（复用 `graph.aget_state`）/ `memory`（复用 `store.asearch`）/ `tools` /
  `health`（复用 `ops.collect_health`），核心闭环对通道一无所知。`channel.py` 的 `Channel`
  Protocol 是通道契约（鸭子类型）；新增前端 = 实现 `start/stop`、输入走 `submit_user_text`、
  输出 `subscribe`。`build_default_service` 一行装配离线可跑实例（默认 `_OfflineEchoModel`
  兜底、内存存储、常驻 driver）。第一个通道是 **Web 控制台**（`frontends/web/`）：纯 **stdlib**
  `http.server` + **SSE**，零第三方依赖（嵌入式友好）；同步 server 与 async 闭环间用
  `run_coroutine_threadsafe` 桥接。入口 `python -m robot_agent.frontends.web`
  （`--host/--port/--robot-id/--sqlite/--idle-prompt`），网页可对话/看历史/看记忆/看工具。

## 跨切面纪律（务必遵守）

- **依赖纪律**：硬件 SDK / ROS / OpenCV / 控制算法只允许出现在 `hal/plugins/<impl>` 实现包内，
  **不进**核心四库依赖树。远程 LLM 客户端（`langchain-openai` / `langchain-anthropic`）也只在请求真实档位时**惰性 import**。
- **Mock 优先**：不接真实 LLM / 真硬件即可离线跑通闭环与回归。新增功能优先保证 `mock` 路径可测，
  断言执行器 `.log` 即可验证行为。
- **提示词归位 `prompts/`**：任何 LLM 可见文案不要内联进逻辑代码；新增/修改提示词走
  `robot_agent/prompts/`（registry 登记 + `.md` 正文 + 调用处 `prompts.render(id, …)`），
  占位符须与 `params` 一致（否则启动 fail-fast）。优化提示词 = 直接编辑对应 `.md`。
- **闭环是 async-only**：执行器 `execute`、工具、`pre_model_hook`、记忆 hook 全是 `async`，
  入口用 `ainvoke`。下发动作的工具必须 `async def` + `await`，否则只是返回未执行的协程。
- **docstring / 注释**里引用行内代码用单反引号（`` `code` ``），**不要**用 Sphinx 双反引号（`` ``code`` ``）。
- **`CLAUDE.md` 与 `AGENTS.md` 内容保持一致**（两文件目前互为副本，改一个要同步另一个）。
