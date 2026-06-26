# 🦜🕸️ LangGraph

[![PyPI - Version](https://img.shields.io/pypi/v/langgraph?label=%20)](https://pypi.org/project/langgraph/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langgraph)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langgraph)](https://pypistats.org/packages/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

> 说明：本库是从上游 LangGraph 裁剪而来的精简版（移除了远程执行 / langgraph-sdk 依赖），面向嵌入式机器人 Agent 场景。详见仓库根目录 `docs/SLIMMING_NOTES.md`。

需要 JS/TS 版本？参见 [LangGraph.js](https://github.com/langchain-ai/langgraphjs)。

## 快速安装

```bash
uv add langgraph
```

## 🤔 这是什么？

LangGraph 是一个用于构建、管理和部署长时运行、有状态 agent 的低层编排框架。它为可恢复执行（durable execution）、流式输出、人在环（human-in-the-loop）、持久化、记忆等提供底层基础设施。

当你的需求较为复杂——需要混合确定性与 agentic 工作流、大量自定义、以及对时延的精细控制时，推荐使用 LangGraph。若你想用预置的 agent 架构与模型集成快速搭建基于 LLM 的 agent 和应用，可使用 [LangChain](https://docs.langchain.com/oss/python/langchain/overview)。

LangChain 的 [agents](https://docs.langchain.com/oss/python/langchain/agents) 正是构建在 LangGraph 之上，从而获得可恢复执行、流式、人在环、持久化等能力（基础的 LangChain agent 用法无需了解 LangGraph）。

## 📖 文档

完整文档见 [API 参考](https://reference.langchain.com/python/langgraph/)。概念指南、教程与示例见 [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)。可从 [LangGraph 快速上手](https://docs.langchain.com/oss/python/langgraph/quickstart) 开始。

## 📕 发布与版本

参见我们的 [发布](https://docs.langchain.com/oss/python/release-policy) 与 [版本](https://docs.langchain.com/oss/python/versioning) 策略。

## 致谢

LangGraph 受 [Pregel](https://research.google/pubs/pub37252/) 与 [Apache Beam](https://beam.apache.org/) 启发，其公开接口借鉴了 [NetworkX](https://networkx.org/documentation/latest/)。LangGraph 由 LangChain Inc 构建，可脱离 LangChain 使用。
