# 验收 / 回归测试

瘦身时删除了上游各库自带的庞大单测，这里是面向「嵌入式机器人 Agent 底座」的一套
**轻量验收测试**，保护我们的裁剪结果与可靠性承诺。全部本地运行，不需要任何远程 LLM
或外部服务（SQLite 用内存 / tmp 临时文件）。

| 文件 | 覆盖 |
|------|------|
| `test_slimming_invariants.py` | 裁剪不变量：核心/记忆/prebuilt 符号可导入、`langgraph` 不再依赖 `langgraph-sdk`、远程模块已删、源码无残留 sdk import、本地 `BaseUser` Protocol 行为、`checkpoint-sqlite` 声明 `orjson`。 |
| `test_memory.py` | 短期记忆 checkpoint 重启恢复与线程隔离；长期记忆 store 跨会话读写、命名空间检索与删除。 |
| `test_agent_loop.py` | 思考→工具→记忆主循环：手写 `StateGraph`(`ToolNode`+`tools_condition`) 与 `create_react_agent`(脚本化假模型) 两条路径，含带 checkpointer 的重启恢复。 |

## 运行

```bash
# 在已安装四个 editable 库 + pytest/pytest-asyncio 的环境中：
make test            # 根目录，等价于 pytest tests/
# 或
python -m pytest tests/ -v
```

依赖：`pytest`、`pytest-asyncio`（异步用例靠 `pytest.ini` 的 `asyncio_mode=auto` 自动驱动）。
脚本化假模型 `FakeToolCallingModel` 见 `conftest.py`。

## 真实 LLM 兼容性测试

真实模型测试独立放在 `scripts/probe_live_llm.py`，不会被本目录的离线 pytest 自动收集。它读取根目录
`.env`，会访问远程 API 并产生费用：

```bash
make test-llm
make test-llm ARGS="--profile fast --checks chat,forced-tool"
```

探针覆盖普通对话、线程历史、自然/强制工具调用和跨线程长期记忆，并对输出 token、请求时间、
客户端重试、单用例时间和图递归步数设置上限。完整参数见 `python scripts/probe_live_llm.py --help`。

真实视觉模型使用独立探针，同样不会被离线 pytest 自动收集：

```bash
make test-vlm
make test-vlm ARGS="--model your-vision-model --json-report /tmp/vlm-report.json"
```

它生成固定测试图并走 `MemoryVisionSource → describe_image → VLM` 链路，校验视觉结果、信任标记和
`image_ref` 边界。完整参数见 `python scripts/probe_live_vlm.py --help`。
