# LangGraph Prebuilt

[![PyPI - Version](https://img.shields.io/pypi/v/langgraph-prebuilt?label=%20)](https://pypi.org/project/langgraph-prebuilt/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langgraph-prebuilt)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langgraph-prebuilt)](https://pypistats.org/packages/langgraph-prebuilt)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

## 快速安装

```bash
uv add langgraph
```

## 🤔 这是什么？

本库定义了创建与执行 LangGraph agent 和工具的高层 API。它包含 `create_react_agent`、`ToolNode`、校验辅助类，以及 Agent Inbox 相关的 schema 等预置组件。

## 📖 文档

完整文档见 [API 参考](https://reference.langchain.com/python/langgraph.prebuilt/)。概念指南与教程见 [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)。

> [!IMPORTANT]
> 本库随 `langgraph` 一起打包；多数用户应安装 `langgraph`，而不要单独安装 `langgraph-prebuilt`（它在 import 期会用到核心模块，但依赖刻意不声明 `langgraph` 以避免循环依赖）。

## Agents

`langgraph-prebuilt` 提供了一个工具调用、ReAct 风格 agent 的[实现](https://reference.langchain.com/python/langgraph.prebuilt/chat_agent_executor/create_react_agent) —— `create_react_agent`：

```bash
uv add langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

# 定义供 agent 使用的工具
def search(query: str):
    """调用以联网搜索。"""
    # 这是占位实现，但别告诉 LLM……
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "旧金山 60 华氏度，有雾。"
    return "90 华氏度，晴。"

tools = [search]
model = ChatAnthropic(model="claude-3-7-sonnet-latest")

app = create_react_agent(model, tools)
# 运行 agent
app.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
)
```

## Tools

### ToolNode

`langgraph-prebuilt` 提供了一个执行工具调用的节点[实现](https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolNode) —— `ToolNode`：

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

def search(query: str):
    """调用以联网搜索。"""
    # 这是占位实现，但别告诉 LLM……
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "旧金山 60 华氏度，有雾。"
    return "90 华氏度，晴。"

tool_node = ToolNode([search])
tool_calls = [{"name": "search", "args": {"query": "what is the weather in sf"}, "id": "1"}]
ai_message = AIMessage(content="", tool_calls=tool_calls)
# 执行工具调用
tool_node.invoke({"messages": [ai_message]})
```

### ValidationNode

`langgraph-prebuilt` 提供了一个按 pydantic schema 校验工具调用的节点[实现](https://reference.langchain.com/python/langgraph.prebuilt/tool_validator/ValidationNode) —— `ValidationNode`：

```python
from pydantic import BaseModel, field_validator
from langgraph.prebuilt import ValidationNode
from langchain_core.messages import AIMessage


class SelectNumber(BaseModel):
    a: int

    @field_validator("a")
    def a_must_be_meaningful(cls, v):
        if v != 37:
            raise ValueError("Only 37 is allowed")
        return v

validation_node = ValidationNode([SelectNumber])
validation_node.invoke({
    "messages": [AIMessage("", tool_calls=[{"name": "SelectNumber", "args": {"a": 42}, "id": "1"}])]
})
```

## Agent Inbox

本库包含将 [Agent Inbox](https://github.com/langchain-ai/agent-inbox) 与 LangGraph agent 配合使用的 schema。用法详见[这里](https://github.com/langchain-ai/agent-inbox#interrupts)。

```python
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanResponse

def my_graph_function():
    # 从 state 的 `messages` 字段中取出最后一个工具调用
    tool_call = state["messages"][-1].tool_calls[0]
    # 创建一个 interrupt
    request: HumanInterrupt = {
        "action_request": {
            "action": tool_call['name'],
            "args": tool_call['args']
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False
        },
        "description": _generate_email_markdown(state) # 生成一段详细的 markdown 描述。
    }
    # 以列表形式发出 interrupt 请求，并取出第一个响应
    response = interrupt([request])[0]
    if response['type'] == "response":
        # 处理该响应
    ...
```

## 📕 发布与版本

参见我们的 [发布](https://docs.langchain.com/oss/python/release-policy) 与 [版本](https://docs.langchain.com/oss/python/versioning) 策略。
