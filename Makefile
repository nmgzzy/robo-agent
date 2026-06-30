# uv workspace 统一管理：4 库（libs/*）+ 应用层 robot_agent 同处一个 uv.lock。
# 设计见 docs/superpowers/specs/2026-06-30-uv-workspace-migration-design.md。
#
# 两个使用场景：
#   1) PC 开发/调试：以 uv 为主（make install / test / lint / format）。
#   2) 嵌入式运行：uv 可用则同样用 uv；不便时退回系统 Python + 纯 pip（make install-pip）。

# 纯 pip 回退路径的本地库安装顺序（拓扑序：下游依赖上游）。
PIP_LIB_ORDER := libs/checkpoint libs/checkpoint-sqlite libs/prebuilt libs/langgraph

# Default target
.PHONY: all
all: lint test

# 安装（PC，uv 为主）：同步开发依赖 + 4 库 editable，写入 uv.lock。
.PHONY: install
install:
	uv sync

# 安装并带上全部可选远程客户端 extra（openai + anthropic）。
.PHONY: install-all
install-all:
	uv sync --extra all

# 嵌入式无 uv 回退：用系统 Python 的 pip，按拓扑序先装 4 本地库再装应用层。
# 先装本地库可让随后 `pip install .` 视依赖已满足、不去 PyPI 拉到上游同名包。
# 用法：在已激活的目标 venv 内执行 `make install-pip`（可选 EXTRAS=".[all]"）。
.PHONY: install-pip
install-pip:
	@for dir in $(PIP_LIB_ORDER); do \
		echo "pip install $$dir"; \
		pip install "$$dir" || exit 1; \
	done
	pip install "$(if $(EXTRAS),$(EXTRAS),.)"

# 整仓 lint（一次，不再逐库递归）。
.PHONY: lint
lint:
	uv run ruff check .

# 整仓格式化 + 自动修复。
.PHONY: format
format:
	uv run ruff format .
	uv run ruff check --fix .

# 锁定依赖（单一根 uv.lock）。
.PHONY: lock
lock:
	uv lock

.PHONY: lock-upgrade
lock-upgrade:
	uv lock --upgrade

# 验收/回归测试集中在根 tests/（瘦身后上游 per-lib 单测已移除，见 docs/SLIMMING_NOTES.md）。
# TEST 可指定单文件或附加任意 pytest 参数，例：TEST="tests/test_memory.py -k recall" make test
.PHONY: test
test:
	uv run pytest $(if $(TEST),$(TEST),tests/)

# 真实 LLM 兼容性探针（按需，使用 .env，可能产生费用）。
# 例：make test-llm ARGS="--profile smart --checks chat,forced-tool"
.PHONY: test-llm
test-llm:
	uv run python scripts/probe_live_llm.py $(ARGS)

# 真实 VLM 兼容性探针（按需，使用 .env，可能产生费用）。
# 例：make test-vlm ARGS="--model your-vision-model"
.PHONY: test-vlm
test-vlm:
	uv run python scripts/probe_live_vlm.py $(ARGS)
