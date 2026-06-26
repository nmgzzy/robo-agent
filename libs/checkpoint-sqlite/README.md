# LangGraph SQLite Checkpoint

[![PyPI - Version](https://img.shields.io/pypi/v/langgraph-checkpoint-sqlite?label=%20)](https://pypi.org/project/langgraph-checkpoint-sqlite/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langgraph-checkpoint-sqlite)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langgraph-checkpoint-sqlite)](https://pypistats.org/packages/langgraph-checkpoint-sqlite)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

## 快速安装

```bash
uv add langgraph-checkpoint-sqlite
```

## 🤔 这是什么？

本库提供 LangGraph checkpoint saver 的 SQLite 实现，通过 `aiosqlite` 同时支持同步与异步。当你需要由 SQLite 支撑的 LangGraph 状态持久化——用于本地开发、测试或轻量部署（如嵌入式场景）时，使用它。它同样提供长期记忆 store 的 SQLite 实现。

## 📖 文档

完整文档见 [API 参考](https://reference.langchain.com/python/langgraph.checkpoint.sqlite)。关于持久化与记忆的概念指南见 [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)。

## 安全

> [!IMPORTANT]
> 创建 checkpointer 时请设置 `LANGGRAPH_STRICT_MSGPACK=true`，或传入显式的 `allowed_msgpack_modules` 列表。这会把 checkpoint 反序列化限制在已知安全的类型范围内，避免数据库被篡改时发生代码执行。详见 [langgraph-checkpoint README](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint#serde)。

## 用法

```python
from langgraph.checkpoint.sqlite import SqliteSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

with SqliteSaver.from_conn_string(":memory:") as checkpointer:
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

### 异步

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
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
    await checkpointer.aput(write_config, checkpoint, {}, {})

    # 读取 checkpoint
    await checkpointer.aget(read_config)

    # 列出 checkpoint
    [c async for c in checkpointer.alist(read_config)]
```

## 📕 发布与版本

参见我们的 [发布](https://docs.langchain.com/oss/python/release-policy) 与 [版本](https://docs.langchain.com/oss/python/versioning) 策略。
