"""前端通道层（frontends）验收：通道无关门面 `AgentService` + Web 通道（stdlib + SSE）。

可拔插边界 = `Driver.submit(Event)`（输入）/ `on_turn(TurnResult)`（输出）；`AgentService`
把它封装成与通道无关的门面，Web 只是第一个 `Channel` 实现。覆盖：

- 门面：提交用户文本 → 订阅广播收到回合 → history / tools / memory / health 可读。
- Web 通道：ThreadingHTTPServer 绑随机端口，stdlib urllib 打 REST 端点冒烟。

全部离线（Mock LLM + 内存 checkpoint/store），不连真实 LLM / 真硬件 / 外部服务。
"""

from __future__ import annotations

import asyncio
import json
import urllib.request

from langchain_core.messages import AIMessage

from robot_agent.frontends import AgentService, Channel, build_default_service
from robot_agent.frontends.web import WebChannel
from robot_agent.llm import make_model


def _service(responses=None) -> AgentService:
    model = make_model(responses=responses or [AIMessage(content="收到，已记下。")])
    return build_default_service(model=model)


# --------------------------------------------------------------------------- #
# 门面：通道无关能力
# --------------------------------------------------------------------------- #


async def test_submit_broadcasts_turn_and_records_history():
    service = _service([AIMessage(content="你好，我在。")])
    await service.start()
    try:
        q = service.subscribe()
        await service.submit_user_text("在吗")
        event = await asyncio.wait_for(q.get(), timeout=5.0)
        assert event["type"] == "turn"
        assert event["thread_id"] == "user_msg"
        assert "你好" in event["reply"]

        history = await service.history("user_msg")
        roles = [m["role"] for m in history]
        assert "human" in roles and "ai" in roles
        assert any("在吗" in m["content"] for m in history)
    finally:
        await service.stop()


async def test_tools_memory_health_views():
    service = _service()
    await service.start()
    try:
        names = {t["name"] for t in service.tools()}
        # 机器人控制工具 + 只读世界状态 + 记忆工具都应在册。
        assert {"move_to", "get_world_state", "remember_fact"} <= names

        mem = await service.memory()
        assert set(mem.keys()) == {"facts", "episodic", "prefs"}
        assert all(isinstance(v, list) for v in mem.values())

        health = await service.health()
        assert "turns" in health and "pending_threads" in health
        json.dumps(health)  # 必须可序列化
    finally:
        await service.stop()


async def test_unsubscribe_stops_delivery():
    service = _service()
    await service.start()
    try:
        q = service.subscribe()
        service.unsubscribe(q)
        await service.submit_user_text("hi")
        await asyncio.sleep(0.2)
        assert q.empty()  # 已注销，不再投递
    finally:
        await service.stop()


def test_webchannel_satisfies_channel_protocol():
    assert isinstance(WebChannel(), Channel)


async def test_memory_paginates_beyond_default_limit():
    # store.asearch 默认 limit=10；写入 >10 条应全部读回（回归：分页拉满）。
    from robot_agent.memory import ns

    service = _service()
    await service.start()
    try:
        for i in range(15):
            await service.store.aput(
                ns(service.robot_id, "facts"), f"f{i}", {"value": i}
            )
        mem = await service.memory()
        assert len(mem["facts"]) == 15
    finally:
        await service.stop()


async def test_immediate_start_then_stop_is_prompt():
    # 回归：start() 紧邻 stop() 不应空等 5s（run() 重置 _running 的竞态）。
    service = _service()
    await service.start()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await service.stop()
    assert loop.time() - t0 < 3.0


# --------------------------------------------------------------------------- #
# Web 通道：REST 端点冒烟（stdlib urllib）
# --------------------------------------------------------------------------- #


def _get_json(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=5.0) as r:
        return r.status, json.loads(r.read())


def _post_json(base: str, path: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5.0) as r:
        return r.status, json.loads(r.read())


async def test_web_rest_endpoints_smoke():
    service = _service([AIMessage(content="好的。")])
    await service.start()
    channel = WebChannel(host="127.0.0.1", port=0)  # 系统分配端口
    await channel.start(service)
    host, port = channel.address
    base = f"http://{host}:{port}"
    try:
        # 同步 urllib 调用放到线程，避免阻塞 driver 所在 loop。
        status, tools = await asyncio.to_thread(_get_json, base, "/api/tools")
        assert status == 200 and any(t["name"] == "move_to" for t in tools)

        status, health = await asyncio.to_thread(_get_json, base, "/api/health")
        assert status == 200 and "turns" in health

        status, ack = await asyncio.to_thread(
            _post_json, base, "/api/chat", {"text": "测试一下"}
        )
        assert status == 200 and ack["ok"] is True

        # 回合落库后历史可读。
        for _ in range(50):
            status, hist = await asyncio.to_thread(
                _get_json, base, "/api/history?thread_id=user_msg"
            )
            if any("测试一下" in m["content"] for m in hist):
                break
            await asyncio.sleep(0.1)
        assert any("测试一下" in m["content"] for m in hist)

        status, root = None, None
        with urllib.request.urlopen(base + "/", timeout=5.0) as r:
            status = r.status
            root = r.read()
        assert status == 200 and b"Robot Agent" in root
    finally:
        await channel.stop()
        await service.stop()
