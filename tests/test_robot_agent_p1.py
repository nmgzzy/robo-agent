"""P1 验收：核心闭环 MVP（思考 → 决策 → 行动 → 记忆）。

对应 docs/IMPLEMENTATION_PLAN.md §P1 与 docs/ROBOT_AGENT_DESIGN.md §4–§6。覆盖：

- **AC-1**：给定指令，Agent 产出预期工具调用序列，MockActuator.log 与期望一致。
- **AC-7（回归雏形）**：给定脚本化观测（注入世界状态），断言下发指令序列。
- **AC-3（雏形）**：跨会话写入 prefs 后，新 thread_id 能检索并注入到 LLM 输入。

外加 HAL / 工具 / 记忆 hook 的单元断言。全部离线（Mock LLM + Mock HAL + 内存 SQLite）。
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from robot_agent import build_effectors, build_robot_agent, make_model
from robot_agent.hal import (
    MockActuator,
    MockBase,
    Observation,
    ScriptedCamera,
    ScriptedSensor,
)
from robot_agent.hal.registry import EffectorRegistry
from robot_agent.memory import KIND_PREFS, ns


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# --------------------------------------------------------------------------- #
# HAL：Mock 实现与注册表
# --------------------------------------------------------------------------- #


async def test_mock_actuator_records_commands():
    base = MockActuator("base")
    res = await base.execute({"action": "move_to", "x": 1.0, "y": 2.0})
    assert res["ok"] is True and res["actuator"] == "base"
    assert base.log == [{"action": "move_to", "x": 1.0, "y": 2.0}]


def test_mock_base_factory():
    base = MockBase()
    assert base.name == "base" and base.log == []


async def test_scripted_sensor_replays_observations():
    cam = ScriptedCamera([{"ts": 1.0, "obj": "cup"}, {"ts": 2.0, "obj": "ball"}])
    obs = [o async for o in cam.stream()]
    assert [o.source for o in obs] == ["camera", "camera"]
    assert [o.ts for o in obs] == [1.0, 2.0]
    assert obs[0].payload["obj"] == "cup"
    assert isinstance(obs[0], Observation)


async def test_scripted_sensor_custom_name():
    s = ScriptedSensor("asr", [{"ts": 0.0, "text": "你好"}])
    obs = [o async for o in s.stream()]
    assert obs[0].source == "asr" and obs[0].payload["text"] == "你好"


def test_build_effectors_mock_has_default_names():
    eff = build_effectors("mock")
    assert set(eff) == {"base", "arm", "speaker"}
    assert all(isinstance(a, MockActuator) for a in eff.values())


def test_build_effectors_unknown_tier_raises():
    with pytest.raises(ValueError, match="real / sim"):
        build_effectors("real")


def test_registry_missing_effector_raises():
    reg = EffectorRegistry({"base": MockActuator("base")})
    with pytest.raises(KeyError, match="未注册执行器"):
        reg["arm"]


# --------------------------------------------------------------------------- #
# AC-1：指令 → 预期工具调用序列；MockActuator.log 与期望一致
# --------------------------------------------------------------------------- #


async def test_ac1_instruction_to_tool_sequence():
    """给定「把杯子拿给我」，Agent 依次 move_to → grasp → 终态回答，log 与期望一致。"""
    effectors = build_effectors("mock")
    model = make_model(
        responses=[
            _tool_call("move_to", {"x": 1.0, "y": 2.0}, "c1"),
            _tool_call("grasp", {"obj": "cup"}, "c2"),
            AIMessage(content="已把杯子拿给你"),
        ]
    )
    agent = build_robot_agent(model=model, effectors=effectors)

    out = await agent.ainvoke({"messages": [HumanMessage("把桌上的杯子拿给我")]})

    assert effectors["base"].log == [{"action": "move_to", "x": 1.0, "y": 2.0}]
    assert effectors["arm"].log == [{"action": "grasp", "target": "cup"}]
    assert effectors["speaker"].log == []
    assert out["messages"][-1].content == "已把杯子拿给你"
    assert [m.type for m in out["messages"]] == [
        "human",
        "ai",
        "tool",
        "ai",
        "tool",
        "ai",
    ]


# --------------------------------------------------------------------------- #
# AC-7（雏形）：给定脚本化观测（注入世界状态），断言下发指令序列
# --------------------------------------------------------------------------- #


async def test_ac7_world_state_drives_command_sequence():
    """注入检测到的物体快照 → Agent 读世界状态 → 据此移动并播报，断言指令序列。"""
    effectors = build_effectors("mock")
    model = make_model(
        responses=[
            _tool_call("get_world_state", {}, "w1"),
            _tool_call("move_to", {"x": 3.0, "y": 2.0}, "w2"),
            _tool_call("speak", {"text": "我看到杯子了"}, "w3"),
            AIMessage(content="完成"),
        ]
    )
    agent = build_robot_agent(model=model, effectors=effectors)

    out = await agent.ainvoke(
        {
            "messages": [HumanMessage("去你看到的物体那里")],
            "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "battery": 88.0,
            "detections": [{"label": "cup", "x": 3.0, "y": 2.0}],
        }
    )

    # get_world_state 是只读工具，不应下发任何执行器指令。
    assert effectors["base"].log == [{"action": "move_to", "x": 3.0, "y": 2.0}]
    assert effectors["speaker"].log == [{"action": "speak", "text": "我看到杯子了"}]
    assert effectors["arm"].log == []

    # 世界状态确实通过工具回灌给了模型（tool 消息里能看到检测结果）。
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    world = json.loads(tool_msgs[0].content)
    assert world["detections"][0]["label"] == "cup"
    assert world["battery"] == 88.0


# --------------------------------------------------------------------------- #
# AC-3（雏形）：跨会话写入 prefs → 新 thread_id 检索注入
# --------------------------------------------------------------------------- #


async def test_ac3_prefs_injected_in_new_thread(tmp_path):
    """先把用户偏好写入 prefs，新 thread_id 起一个回合时该偏好被注入 LLM 输入。"""
    robot_id = "robot-1"
    db = str(tmp_path / "agent.db")

    # 会话 A：直接把偏好写入长期记忆（模拟此前学到的偏好）。
    async with AsyncSqliteStore.from_conn_string(db) as store:
        await store.aput(ns(robot_id, KIND_PREFS), "speak_lang", {"value": "讲中文"})

    # 会话 B：全新 thread_id，模型脚本只回一句终态；hook 应已注入偏好。
    model = make_model(responses=[AIMessage(content="好的")])
    async with (
        AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver,
        AsyncSqliteStore.from_conn_string(db) as store,
    ):
        agent = build_robot_agent(
            model=model, checkpointer=saver, store=store, robot_id=robot_id
        )
        await agent.ainvoke(
            {"messages": [HumanMessage("你好")]},
            {"configurable": {"thread_id": "fresh-thread"}},
        )

    # 模型本次收到的输入里，应含注入的偏好文本（SystemMessage）。
    assert model.received, "模型未被调用"
    injected = model.received[0]
    sys_texts = [m.content for m in injected if isinstance(m, SystemMessage)]
    assert any("讲中文" in t for t in sys_texts), sys_texts


async def test_ac3_no_memory_no_injection():
    """无长期记忆时，hook 只裁剪不注入，不应凭空加入 system 记忆块。"""
    model = make_model(responses=[AIMessage(content="嗯")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        agent = build_robot_agent(model=model, store=store)
        await agent.ainvoke({"messages": [HumanMessage("在吗")]})

    injected = model.received[0]
    sys_texts = [m.content for m in injected if isinstance(m, SystemMessage)]
    assert not any("长期记忆" in t for t in sys_texts)


# --------------------------------------------------------------------------- #
# 记忆回写工具：remember_fact → recall
# --------------------------------------------------------------------------- #


async def test_remember_fact_writeback_and_recall():
    """remember_fact 写回 facts，recall 能读回（事实回写路径，§6.3）。"""
    effectors = build_effectors("mock")
    model = make_model(
        responses=[
            _tool_call("remember_fact", {"key": "dock", "value": "(3,2)"}, "m1"),
            _tool_call("recall", {"key": "dock"}, "m2"),
            AIMessage(content="充电桩在 (3,2)"),
        ]
    )
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        agent = build_robot_agent(model=model, effectors=effectors, store=store)
        out = await agent.ainvoke({"messages": [HumanMessage("充电桩在哪记一下")]})

    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert "已记住" in tool_msgs[0].content
    assert "(3,2)" in tool_msgs[1].content
    assert out["messages"][-1].content == "充电桩在 (3,2)"
