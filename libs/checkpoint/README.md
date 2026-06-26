# LangGraph Checkpoint

[![PyPI - Version](https://img.shields.io/pypi/v/langgraph-checkpoint?label=%20)](https://pypi.org/project/langgraph-checkpoint/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langgraph-checkpoint)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langgraph-checkpoint)](https://pypistats.org/packages/langgraph-checkpoint)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

## 快速安装

```bash
uv add langgraph-checkpoint
```

## 🤔 这是什么？

本库定义了 LangGraph checkpointer 的基础接口。Checkpointer 为 LangGraph 提供持久化层：在每个超步（superstep）保存图状态，从而支持人在环、交互之间的记忆、可恢复执行等能力。

## 📖 文档

完整文档见 [API 参考](https://reference.langchain.com/python/langgraph.checkpoint)。关于持久化与记忆的概念指南见 [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)。

## 核心概念

### Checkpoint（检查点）

Checkpoint 是图状态在某一时刻的快照。Checkpoint tuple 指包含该 checkpoint 及其关联 config、metadata 和待写入（pending writes）的对象。

### Thread（线程）

Thread 让多个不同的运行各自被 checkpoint，对多租户聊天应用以及其它需要维护独立状态的场景至关重要。Thread 是 checkpointer 保存的一系列 checkpoint 所共享的唯一 ID。使用 checkpointer 时，运行图必须指定 `thread_id`，并可选地指定 `checkpoint_id`。

- `thread_id` 即一个 thread 的 ID，始终必填。
- `checkpoint_id` 可选传入，用于指向某个 thread 内的特定 checkpoint，可据此从 thread 的中途某点开始运行图。

调用图时需将它们放在 config 的 configurable 部分传入，例如：

```python
{"configurable": {"thread_id": "1"}}  # 合法的 config
{"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}  # 也合法
```

### Serde（序列化/反序列化）

`langgraph-checkpoint` 还定义了序列化/反序列化（serde）协议，并提供默认实现（`langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`），可处理多种类型，包括 LangChain 与 LangGraph 原语、datetime、enum 等。

> [!IMPORTANT]
> **Checkpoint 反序列化安全：** 默认情况下序列化器允许 checkpoint 数据中出现任意 Python 类型。新应用应设置环境变量 `LANGGRAPH_STRICT_MSGPACK=true`，或向 `JsonPlusSerializer` 传入显式的 `allowed_msgpack_modules` 列表，把反序列化限制在已知安全的类型范围内。

### Pending writes（待写入）

当某个图节点在某个超步执行到一半失败时，LangGraph 会保存同一超步中其它已成功完成节点的待写入，这样从该超步恢复执行时就不会重复运行那些已成功的节点。

## 接口

每个 checkpointer 都应符合 `langgraph.checkpoint.base.BaseCheckpointSaver` 接口，并实现以下方法：

- `.put` —— 存储一个 checkpoint 及其配置与元数据。
- `.put_writes` —— 存储关联到某个 checkpoint 的中间写入（即 pending writes）。
- `.get_tuple` —— 按给定配置（`thread_id` 与 `checkpoint_id`）获取一个 checkpoint tuple。
- `.list` —— 列出匹配给定配置与过滤条件的 checkpoint。
- `.delete_thread()` —— 删除某个 thread 关联的全部 checkpoint 与写入。
- `.get_next_version()` —— 为某个 channel 生成下一个版本 ID。

若 checkpointer 将用于异步图执行（即通过 `.ainvoke`、`.astream`、`.abatch` 执行图），则必须实现上述方法的异步版本（`.aput`、`.aput_writes`、`.aget_tuple`、`.alist`）。同理，若需要异步清理 thread，则需实现 `.adelete_thread()`。基类提供了 `.get_next_version()` 的默认实现（生成从 1 开始的整数序列），如需自定义版本方案可覆盖该方法。

## 用法

```python
from langgraph.checkpoint.memory import InMemorySaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

checkpointer = InMemorySaver()
checkpoint = {
    "v": 4,
    "ts": "2024-07-31T20:14:19.804150+00:00",
    "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
    "channel_values": {
      "my_key": "meow",
      "node": "node"
    },
    "channel_versions": {
      "__start__": 2,
      "my_key": 3,
      "start:node": 3,
      "node": 3
    },
    "versions_seen": {
      "__input__": {},
      "__start__": {
        "__start__": 1
      },
      "node": {
        "start:node": 2
      }
    },
}

# 存储 checkpoint
checkpointer.put(write_config, checkpoint, {}, {})

# 读取 checkpoint
checkpointer.get(read_config)

# 列出 checkpoint
list(checkpointer.list(read_config))
```

## 📕 发布与版本

参见我们的 [发布](https://docs.langchain.com/oss/python/release-policy) 与 [版本](https://docs.langchain.com/oss/python/versioning) 策略。
