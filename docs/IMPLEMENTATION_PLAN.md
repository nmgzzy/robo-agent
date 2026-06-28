# 机器人 Agent 分阶段实现计划

> 本文是 [`ROBOT_AGENT_DESIGN.md`](./ROBOT_AGENT_DESIGN.md)（需求 + 技术总纲）的**落地实现计划**。
> 把「完整数字个体（设计 §1–§8）」拆为 **11 个阶段（P0–P10）**，每阶段可独立交付、可在 Mock 下单测、可回归。
> 阶段顺序遵循设计 §8.10 的落地优先级：**底座闭环先行 → 自主引擎 → 治理 → 成长 → 运维**。

---

## 0. 总体原则与里程碑

### 0.1 贯穿全程的纪律

- **依赖纪律**：硬件 SDK / ROS / OpenCV / 控制算法只许出现在 `plugins/<impl>` 实现包，**不进核心四库依赖树**（对齐 `SLIMMING_NOTES.md`）。
- **Mock 优先**：每个外部边界（HAL、LLM）都先有 Mock 实现，不接真硬件即可跑通闭环、单测、回归。
- **每阶段必须**：① 明确交付物；② 配验收测试（落在根 `tests/`）；③ 改任意库后在该库目录跑 `make format` + `make lint`；④ 根目录 `make test` 通过。
- **可平滑下沉**：先用 `create_react_agent` 起步，需要更强控制时再下沉到自定义 `StateGraph`（设计 §4.2），两者能力等价。

### 0.2 里程碑总览

| 阶段 | 名称 | 对应设计 | 产出可演示什么 | 依赖前序 |
|------|------|----------|----------------|----------|
| **P0** | 工程脚手架与依赖 | §2 | 装好、冒烟通过、CI 跑空测试集 | — |
| **P1** | 核心闭环 MVP | §4–§6 | Mock 下「指令→工具序列→落盘」闭环 | P0 |
| **P2** | 可靠性与安全门控 | §7 | 重试/超时/降级/危险动作 interrupt | P1 |
| **P3** | 身份 / 自我模型 | §8.8 | 稳定注入 persona，决策有锚点 | P1 |
| **P4** | 自主引擎：driver + 收件箱 | §8.1 | 无人输入时自发开回合 | P1（建议 P2/P3 后） |
| **P5** | 目标系统 | §8.2 | 目标栈、分解、多目标仲裁 | P4 |
| **P6** | 复盘闭环 | §8.3 | 回合后自评、episodic→semantic 蒸馏 | P4、P5 |
| **P7** | 记忆治理 | §8.4 | 衰减/去重/冲突消解/巩固 | P6 |
| **P8** | 元认知 / 自我监控 | §8.5 | 循环检测、预算感知、上报 | P4 |
| **P9** | 安全 / 对齐策略层 | §8.6 | 宪章约束、工具权限、限流、审计 | P2 |
| **P10** | 技能库 + 运维可观测 | §8.7、§8.9 | 技能数据化复用、决策日记/健康度 | P6、P9 |

**关键路径**：P0 → P1 → P4 → P5 → P6 → P7（数字个体最小可持续闭环）。
P2/P3/P8/P9/P10 可在 P1/P4 之后按资源穿插。

### 0.3 建议目录布局（实现期）

```
robot_agent/                 # 应用层（新建，可引入 HAL 实现依赖）
├── driver/                  # P4：常驻 driver + 收件箱 + 事件总线
├── goals/                   # P5：目标系统
├── reflect/                 # P6：复盘闭环
├── governance/              # P7/P9：记忆治理 + 安全策略
├── metacog/                 # P8：元认知/自我监控
├── skills/                  # P10：技能库
├── ops/                     # P10：运维可观测
├── hal/                     # §5.3：SensorSource/Actuator 接口
│   └── plugins/             #   real / sim / mock 实现
├── memory/                  # §6：hook（注入/裁剪/回写）、namespace 约定
├── tools/                   # §5.2：机器人控制工具
└── graph.py                 # §4：装配 create_react_agent / StateGraph
tests/                       # 根目录验收/回归测试
```

> 核心四库仍在 `libs/`，应用层只经接口/hook 依赖它们；硬件依赖封在 `hal/plugins/`。

---

## P0 · 工程脚手架与依赖

**目标**：可装、可冒烟、可跑测试的工程骨架。

**任务**
1. 按 `SLIMMING_NOTES.md §五` 装四个核心库（editable，顺序：checkpoint → checkpoint-sqlite → prebuilt → langgraph）。
2. 选定并安装远程 LLM 客户端（建议 `langchain-anthropic`，配置 API key / base_url）。
3. 重新生成各库 `uv.lock`（联网 `uv lock`）；按需 `LANGCHAIN_TRACING_V2=false` 关遥测。
4. 建 `robot_agent/` 应用层骨架（空包）+ 根 `tests/` 接入约定。
5. 抽象一层 **LLM 工厂**：`make_model(profile)` 返回 ChatModel，支持模型分层（haiku/opus 切换），便于 Mock。

**交付物**：可 `import` 的应用层空骨架；LLM 工厂（含 Mock ChatModel）。

**验收**
- 冒烟导入通过（`StateGraph` / `create_react_agent` / `AsyncSqliteSaver` / `AsyncSqliteStore`）。
- 根 `make test` 能跑（哪怕只有占位用例）。
- 不接真硬件、不接真 LLM 也能 `import` 成功。

---

## P1 · 核心闭环 MVP（思考 → 决策 → 行动 → 记忆）

**目标**：在 Mock HAL + Mock/真 LLM 下跑通完整闭环，满足 **AC-1**。

**任务**
1. **HAL 接口**（设计 §5.3）：定义 `Observation` / `SensorSource` / `Actuator` Protocol。
2. **Mock 实现**：`MockBase`（记录指令）、`ScriptedCamera`/脚本化感知源、注册表/配置选择档位。
3. **工具层**（§5.2）：把 Actuator 包成 `@tool`（`move_to` / `set_velocity` / `grasp` / `speak` / `get_world_state`），用 `InjectedState`/`InjectedStore` 拿状态/记忆。
4. **State schema**（§5.1）：`messages`（`add_messages`）+ 只读世界状态字段（pose/battery/detections）。
5. **记忆 hook**（§6.3）：`pre_model_hook=inject_memory`（检索长期记忆 + 裁剪 messages）；事实回写（工具内或 post hook）。
6. **装配**（§4.1）：`create_react_agent(model, tools, checkpointer=AsyncSqliteSaver, store=AsyncSqliteStore, pre_model_hook)`。
7. **namespace 约定**（§6.2）：先落 `facts` / `episodic` / `prefs`。
8. **内置 VLM**（§5.3.2）：通过 `VisionSource` 按不透明 `image_ref` 取帧；原图不进入
   tool-call / checkpoint；严格校验大小、格式和 MIME，自动降采样到 720p，输出标记为
   不可信感知数据。

**交付物**：`graph.py` 可 `ainvoke` 的 Agent；Mock HAL；记忆 hook。

**验收**
- **AC-1**：给定指令，Agent 产出预期工具调用序列；`MockBase.log` 与期望一致。
- **AC-7（回归雏形）**：给定脚本化观测，断言下发指令序列（落 `tests/`）。
- 跨会话写入 `prefs` 后，新 `thread_id` 能检索注入（**AC-3** 雏形）。
- VLM 工具调用只持有 `image_ref`，checkpoint 中不出现图片 base64；非法/超限图片被拒绝。

---

## P2 · 可靠性与安全门控

**目标**：满足 **NFR-3 / AC-2 / AC-5**。

**任务**
1. **崩溃恢复**：用 `AsyncSqliteSaver` 落盘；写「杀进程 → 同 `thread_id` 重启 → `aget_state` 续跑」回归。
2. **重试**：给 LLM 节点、网络工具配 `RetryPolicy`（`libs/langgraph/.../pregel/_retry.py`）。注意 `default_retry_on` 惰性 import `httpx`/`requests`（见 `SLIMMING_NOTES.md §六`），如不装则换自定义谓词。
3. **超时与降级**：节点级超时；LLM 不可用 → 回退「保守安全策略 / 停在原地等待」。HAL 接口不可用时抛明确错误触发降级。
4. **interrupt 门控**：危险动作（高速、抓取）执行前 `interrupt` → 规则/人工确认 → `Command(resume=...)`。
5. **SQLite WAL** + 过期 checkpoint 清理任务（长跑必做）。

**交付物**：RetryPolicy 配置、超时/降级封装、危险动作 interrupt 包装、checkpoint 清理任务。

**验收**
- **AC-2**：杀进程后同 `thread_id` 恢复完整状态并续跑。
- **AC-5**：危险动作被 interrupt 拦截，确认后方继续。
- 模拟 LLM 故障时进入降级而非崩溃。

---

## P3 · 身份 / 自我模型（零成本先立）

**目标**：满足 **FR-14**，给后续所有决策一个稳定锚点。

**任务**
1. Store `(robot_id,"identity")` 存 persona / 价值观 / 能力自知（擅长/不擅长）。
2. `pre_model_hook` 每次把 identity 注入为 system context（稳定、低成本）。
3. 提供 identity 的读取/更新接口（初期手工写入，后续可由复盘更新）。

**交付物**：identity namespace + 注入逻辑。

**验收**：移除/修改 identity 时，决策语气与边界可观察地变化；长跑下行为不漂移（人工抽查 + 决策日记对照）。

---

## P4 · 自主引擎：driver + 收件箱 + 事件总线（被动库 → 个体的分界线）

**目标**：满足 **FR-7 / AC-4**——无人输入时也能自发运转。

**任务**
1. **Event / Inbox**（设计 §8.1）：定义 `Event` 模型、`Inbox` Protocol（put/get + 优先级 + 超时）。
2. **事件汇入**：事件型观测（§5.3.2）从 `SensorSource` 路由进 `Inbox`；接定时器/cron。
3. **常驻 driver 主循环**：`get(timeout=idle_tick)` → `decide_thread` → `graph.ainvoke` → 空闲/回合后钩子。
4. **空闲策略**：无任务时做什么（巡检 / 整理记忆 / 待机），可配置。
5. **回合编排**：driver 决定「开哪个回合（thread_id）」；与 P2 的恢复语义衔接。

**交付物**：`robot_agent/driver/`（Inbox + 常驻循环 + 空闲策略）。

**验收**
- **AC-4**：不投递任何事件，driver 在 idle_tick 到点后自发开一个回合并执行空闲策略。
- 投递高优先级事件时能打断空闲、优先处理。

---

## P5 · 目标系统

**目标**：满足 **FR-8**——跨回合的目标/意图一等表示。

**任务**
1. **Goal 模型**（设计 §8.2）：id/parent/intent/priority/deadline/status/plan，存 Store `(robot_id,"goals")`。
2. **规划节点**：把长期目标分解为子任务/步骤；支持重规划。
3. **多目标仲裁**：driver 在空闲时按优先级/deadline 选当前目标；被紧急事件打断后能回到原目标。
4. 与 P4 衔接：driver 的 `decide_thread` 改为「看收件箱 + 看目标栈」。

**交付物**：`robot_agent/goals/`（模型 + 规划节点 + 仲裁）。

**验收**
- 注入两个竞争目标，仲裁按优先级/deadline 选对当前目标。
- 紧急事件打断后，处理完能恢复到被打断目标。

---

## P6 · 复盘闭环

**目标**：满足 **FR-9**——让「越用越懂」从愿望变机制。

**任务**
1. **反思子图**：回合结束/周期触发，做 intended vs actual 自评。
2. **蒸馏**：读 `episodic`，把经验蒸馏为 `facts`/`prefs`，必要时更新 `identity`（P3）与 `skills`（P10）。
3. **触发**：由 driver 定时触发或挂 `post_model_hook`。

**交付物**：`robot_agent/reflect/`（反思子图 + 蒸馏逻辑）。

**验收**
- 跑若干回合后，episodic 中的重复经验被蒸馏进 facts/prefs。
- 复盘产出的偏好能在后续回合被检索注入并影响决策。

---

## P7 · 记忆治理（长跑隐形杀手）

**目标**：满足 **FR-10 / NFR-6 / AC-6**——长跑不被脏记忆拖垮。

**任务**
1. **后台 compaction 任务**（由 driver 周期调度），对每个 namespace 执行：
   - **遗忘/衰减**：低重要性/久未命中的记忆衰减或归档。
   - **去重**：合并近似/重复条目。
   - **冲突消解**：检出矛盾 fact，按策略（最新/最高置信/人工）消解。
   - **重要性排序 + 巩固**：高价值记忆上浮、巩固。
2. 与 §6 检索衔接：治理后检索质量不退化。

**交付物**：`robot_agent/governance/`（compaction 任务 + 冲突消解策略）。

**验收**
- **AC-6**：注入矛盾事实后，治理检出冲突并消解；治理后检索 top-k 质量不退化（回归断言）。
- 持续注入噪声记忆，Store 体积与检索时延维持在阈值内。

---

## P8 · 元认知 / 自我监控

**目标**：满足 **FR-11**——对自身认知状态有感知。

**任务**
1. **循环/卡死检测**：`pre_model_hook` 内识别重复决策/无进展循环。
2. **预算感知**：token / 时延 / 电量预算，越界则收敛或上报。
3. **置信度 + escalation**：不确定即 `interrupt` 上报求助。
4. **指标导出**：循环计数、预算占用、置信度等。

**交付物**：`robot_agent/metacog/`（检测 + 预算 + 上报）。

**验收**
- 构造死循环场景，能被检出并中断/上报。
- 预算耗尽时进入收敛/上报而非无限消耗。

---

## P9 · 安全 / 对齐策略层

**目标**：满足 **FR-12**——物理机器人不能只靠 prompt「小心点」。

**任务**
1. **宪章硬约束**：`post_model_hook` 校验决策是否违反硬规则，违反则拦截。
2. **工具权限范围**：按工具定义权限，越权拒绝。
3. **危险动作限流**：高速/抓取等限频限幅。
4. **覆盖审计**：人工覆盖记录、结构化审计日志（与 P10 衔接）。

**交付物**：`robot_agent/governance/`（安全策略层）+ 工具封装层。

**验收**
- 违反宪章的决策被拦截并记审计。
- 越权工具调用被拒；危险动作限流生效。

---

## P10 · 技能库 + 运维可观测

**目标**：满足 **FR-13 / FR-15**——成长载体 + 无人值守监护。

**任务（技能库，§8.7）**
1. Store `(robot_id,"skills")` 存技能定义（数据化的成功计划/参数化过程）。
2. **检索复用 + 组合**：从成功回合沉淀技能（与 P6 复盘衔接），按需检索。
3. **动态工具加载**：运行时把技能装配进 `tools`。

**任务（运维可观测，§8.9）**
4. **决策日记 / 审计**：结构化打点（决策、tool-call、记忆命中），落盘可离线审查。
5. **运行时自省**：「它现在在干嘛、为什么」的查询接口。
6. **健康度指标 + 远程巡检**：指标导出。

**交付物**：`robot_agent/skills/` + `robot_agent/ops/`。

**验收**
- 把一段成功计划存为技能后，新场景能检索复用（动态加载生效）。
- 决策日记可离线还原一次回合的决策链；健康度指标可被外部读取。

---

## 附录 A · 验收标准 → 阶段映射

| 验收标准 | 落在阶段 |
|----------|----------|
| AC-1 核心闭环 | P1 |
| AC-2 崩溃恢复 | P2 |
| AC-3 跨会话记忆 | P1（雏形）→ P6/P7（增强） |
| AC-4 自主运转 | P4 |
| AC-5 安全 interrupt | P2（机制）→ P9（策略） |
| AC-6 记忆治理 | P7 |
| AC-7 Mock 回归 | P1 起，贯穿全程 |

## 附录 B · 风险与缓解

| 风险 | 缓解 |
|------|------|
| 传递性 `httpx`/`requests` 影响默认重试谓词 | P2 用自定义 `RetryPolicy` 谓词，或确认不触发（`SLIMMING_NOTES.md §六`） |
| 长跑 Store 变脏导致检索崩塌 | P7 记忆治理为关键路径，勿滞后 |
| 嵌入式算力紧张 | 模型分层（haiku/opus）、关语义检索退回关键字、裁剪 messages |
| Mock 与真硬件行为差异 | HAL 三档实现 + 接口契约固定；real/sim 仅在实现包内，核心不变 |
| `create_react_agent` 弃用 | 能力等价可下沉自定义 `StateGraph`（设计 §4.2），不增依赖 |
