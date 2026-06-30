# 用 uv workspace 统一管理整个项目（uv 为主，pip 可回退）

日期：2026-06-30
状态：已批准，实施中

## 1. 背景与目标

仓库当前没有根 `pyproject.toml`、没有 `uv.lock`；4 个裁剪后的库（`libs/*`）各自有
`pyproject.toml` 且已用 `[tool.uv.sources]` 做 path 引用；应用层 `robot_agent/` 不是包，
靠 `pytest.ini` 的 `pythonpath=.` 直接 import。安装靠 `uv venv` + `uv pip install -r
requirements-app.txt` + 逐个 `uv pip install -e libs/*`（即 uv 的 **pip 兼容接口**，
不是 `uv sync` + lockfile 的**项目接口**）。

本项目有两个使用场景：

1. **PC 上开发 / 调试**：希望以 uv 为主，`uv sync` 一键装好、`uv.lock` 锁定可复现。
2. **嵌入式 Linux 运行**（控制机器人）：uv 在 aarch64/armv7 有预编译静态二进制，
   多数现代嵌入式 Linux 可用；**若可用则统一用 uv，否则退回系统 Python + 纯 pip**。

因此目标是：把整个项目升级为 **uv workspace（单一 `uv.lock`）作为主路径**，
同时**保留一条不依赖 uv 的纯 `pip` 安装路径**给嵌入式兜底。

## 2. 硬约束

- `[tool.uv.*]` 与 `uv.lock` 是 uv 专属，pip 会忽略——纯 pip 路径不能依赖它们。
- 4 个 fork 库与上游 PyPI 同名（`langgraph` 等）。`pip install .` 会读
  `[project.dependencies]`、忽略 `[tool.uv.sources]`，若库未预装就会从 PyPI 拉到
  **上游真·langgraph**。因此纯 pip 路径必须**先按拓扑序本地安装 4 库**，再 `pip install .`：
  库已满足约束后，pip 视依赖已满足、不再去 PyPI。
- 拓扑序（下游依赖上游）：`checkpoint → {checkpoint-sqlite, prebuilt} → langgraph → robot_agent`。
  本地版本互相满足约束（checkpoint 4.1.1 / checkpoint-sqlite 3.1.0 / prebuilt 1.1.0 /
  langgraph 1.2.6），已核实。

## 3. 文件级改动

- **新建根 `pyproject.toml`**：既是 workspace 根、又是 `robot_agent` 包定义
  （hatchling，`packages = ["robot_agent"]`）。
  - `[tool.uv.workspace] members = ["libs/*"]`
  - `[tool.uv.sources]`：4 库 `{ workspace = true }`
  - `[project.dependencies]`：4 库（不加版本约束，便于 pip 路径用已装本地库满足）+ `pillow>=11,<13`
  - `[project.optional-dependencies]`：`openai` / `anthropic` / `all`
  - `[dependency-groups] dev`：pytest + pytest-asyncio + ruff
- **新建根 `.python-version`**：`3.12`（开发用；`requires-python = ">=3.10"` 不变）。
- **删除 `requirements-app.txt`**：pillow 并入根 core 依赖。
- **改写各库 `[tool.uv.sources]`**：把对兄弟库的 `{ path = "...", editable = true }`
  统一改为 `{ workspace = true }`，避免与根 sources 在 workspace 内冲突。
- **删除 per-lib `uv lock` 调用**（Makefile 的 `lock`/`lock-upgrade` 改为根级 `uv lock`）。
- **生成根 `uv.lock`**。

## 4. 依赖分层（pillow 入 core，远程客户端作 extra）

```
robot-agent（根包，包目录 robot_agent/）
├─ [project.dependencies]      = [4 库（workspace）, "pillow>=11,<13"]
├─ [optional-dependencies]
│    openai     = ["langchain-openai>=1,<2"]
│    anthropic  = ["langchain-anthropic>=1,<2"]
│    all        = ["robot-agent[openai,anthropic]"]
└─ [dependency-groups] dev     = ["pytest", "pytest-asyncio", "ruff"]
```

- 远程客户端保持**惰性 import**，按需 `uv sync --extra openai` 或 pip extra 装入。
- pillow 入 core：vision 是内置能力，且测试 `tests/test_vision.py` 顶层 `import PIL`，
  开箱即可跑 vision 与全部测试。
- 类型检查：库用 `ty`，根层暂只用 `ruff`（不引入 mypy/ty，避免横生类型债；CLAUDE.md
  原写「ruff + mypy」与库实际的 `ty` 不符，本次顺带在文档侧澄清）。

## 5. 命令入口（Makefile 重写，双路径）

| 目标 | 命令 |
|------|------|
| `make install`（PC，uv） | `uv sync`（开发依赖 + 4 库 editable） |
| `make install-all` | `uv sync --extra all` |
| `make install-pip`（嵌入式无 uv 回退） | 按拓扑序 `pip install ./libs/checkpoint ./libs/checkpoint-sqlite ./libs/prebuilt ./libs/langgraph .` |
| `make test` | `uv run pytest tests/` |
| `make lint` | `uv run ruff check .`（整仓一次） |
| `make format` | `uv run ruff format . && uv run ruff check --fix .` |
| `make lock` | `uv lock` |
| `make test-llm` / `make test-vlm` | 保留 |

`pytest.ini` 的 `pythonpath = .` 保留（robot_agent 已为包，二者不冲突）。

## 6. 迁移与验证步骤

1. 写根 `pyproject.toml` + `.python-version`，删 `requirements-app.txt`，改库 sources，改写 Makefile。
2. `uv sync --extra all` 生成 `uv.lock`，确认 4 库走 workspace（editable，指向 `libs/*`）。
3. `uv run pytest tests/` → 必须仍 **237 passed**。
4. 干净环境验证纯 pip 回退路径（系统 python venv，按拓扑序安装），跑测试通过。
5. `uv run ruff check .` 全绿。
6. 同步更新 `CLAUDE.md` / `AGENTS.md`（互为副本）/ `README.md` / `docs/STRUCTURE.md` 安装命令。

## 7. 风险

- **嵌入式 uv 可用性**未实测——靠 pip 回退兜底，不赌 uv 一定能跑。
- **纯 pip 路径必须先装本地库**：若直接 `pip install .` 而未预装 4 库，会拉到上游
  PyPI 包。已在 `make install-pip` 固化拓扑序并在文档强调。
- **lint 从逐库改整仓**：各库 ruff 配置可能不一致；根 `ruff` 配置以 `robot_agent/` 与
  `tests/` 为主，库目录沿用各自配置（ruff 就近读取目录内 pyproject）。

## 8. 顺带修复（review 小问题）

- `robot_agent/graph.py`:106-109 删除不可达的内层 `if vision_source is None` 死分支
  （第 85-86 行已 fail-fast 保证一致）。
- `robot_agent/vision/trust.py`：补注释说明信任边界每回合注入（固定 token 开销）。
- `robot_agent/vision/analyze.py`:76：`DEFAULT_FALLBACK_TEXT` 比较经评估为防御性、且
  通过导入常量自洽（改动它不会破坏链接），误判概率可忽略——保留并加注释说明。
