"""P6 验收：复盘闭环（FR-9）。

对应 docs/IMPLEMENTATION_PLAN.md §P6 与 docs/ROBOT_AGENT_DESIGN.md §8.3。覆盖：

- **回合记录**：episode_from_turn 从 TurnResult 提取 intent/actions/outcome；record/read/prune。
- **蒸馏（验收①）**：跑若干回合写入 episodic 后，reflect_and_distill 把经验蒸馏进 facts/prefs。
- **影响后续决策（验收②）**：蒸馏产出的 pref 在新回合被 pre_model_hook 检索注入到 LLM 输入。
- **driver 衔接**：make_reflect_hook 挂 on_turn，自动记录 + 周期触发蒸馏。

全部离线（Mock LLM + Mock HAL + 内存 SQLite）。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from robot_agent import (
    Driver,
    Episode,
    PriorityInbox,
    build_effectors,
    build_robot_agent,
    make_model,
    make_reflect_hook,
    reflect_and_distill,
    user_message,
)
from robot_agent.memory import KIND_FACTS, KIND_PREFS, ns
from robot_agent.reflect import (
    episode_from_turn,
    parse_distilled,
    prune_episodes,
    read_episodes,
    record_episode,
)
from robot_agent.reflect.distill import DistillResult


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class _Turn:
    """轻量 TurnResult 替身（episode_from_turn 只读 event/result/thread_id/index）。"""

    def __init__(self, index, event, result, thread_id):
        self.index = index
        self.event = event
        self.result = result
        self.thread_id = thread_id
        self.interrupted = False


# --------------------------------------------------------------------------- #
# 回合记录：提取 / 持久化
# --------------------------------------------------------------------------- #


def test_episode_from_turn_extracts_intent_actions_outcome():
    turn = _Turn(
        index=1,
        event=user_message("把杯子拿来"),
        result={
            "messages": [
                HumanMessage("把杯子拿来"),
                _tool_call("move_to", {"x": 1.0, "y": 2.0}, "c1"),
                AIMessage("已把杯子拿给你"),
            ]
        },
        thread_id="user_msg",
    )
    ep = episode_from_turn(turn)
    assert ep.intent == "把杯子拿来"
    assert ep.actions == ["move_to({'x': 1.0, 'y': 2.0})"]
    assert ep.outcome == "已把杯子拿给你"
    assert ep.thread_id == "user_msg"
    # id 跨 driver 重启唯一（含微秒时间戳），尾部带回合序号。
    assert ep.id.startswith("ep-") and ep.id.endswith("-000001")


def test_episode_extracts_only_current_turn_actions():
    """同 thread 复用下 result 累积全历史；episode 只取本回合（最后一条 human 之后）。"""
    turn = _Turn(
        index=2,
        event=user_message("第二个任务"),
        result={
            "messages": [
                HumanMessage("第一个任务"),
                _tool_call("move_to", {"x": 1.0, "y": 1.0}, "a"),
                AIMessage("第一回合完成"),
                HumanMessage("第二个任务"),
                _tool_call("grasp", {"obj": "cup"}, "b"),
                AIMessage("第二回合完成"),
            ]
        },
        thread_id="user_msg",
    )
    ep = episode_from_turn(turn)
    assert ep.actions == ["grasp({'obj': 'cup'})"]  # 不含上一回合的 move_to
    assert ep.outcome == "第二回合完成"


async def test_record_read_prune_episodes():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await record_episode(store, "r1", Episode(id="ep-1", ts=1.0, intent="a"))
        await record_episode(store, "r1", Episode(id="ep-2", ts=2.0, intent="b"))
        eps = await read_episodes(store, "r1")
        assert [e.id for e in eps] == ["ep-1", "ep-2"]  # 按 ts 排序

        await prune_episodes(store, "r1", ["ep-1"])
        assert [e.id for e in await read_episodes(store, "r1")] == ["ep-2"]


# --------------------------------------------------------------------------- #
# 蒸馏解析
# --------------------------------------------------------------------------- #


def test_parse_distilled_handles_both_kinds_and_fullwidth():
    text = "fact: 充电桩 = (3,2)\npref：语言 ＝ 中文\n无关行\npref: 步速 = 缓慢"
    assert parse_distilled(text) == [
        (KIND_FACTS, "充电桩", "(3,2)"),
        (KIND_PREFS, "语言", "中文"),
        (KIND_PREFS, "步速", "缓慢"),
    ]


# --------------------------------------------------------------------------- #
# 验收①：episodic 经验蒸馏进 facts/prefs
# --------------------------------------------------------------------------- #


async def test_ac_distill_writes_facts_and_prefs():
    distiller = make_model(
        responses=[AIMessage("fact: 充电桩位置 = (3,2)\npref: 播报语言 = 中文")]
    )
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        # 攒几条经历（含重复主题）。
        for i in range(3):
            await record_episode(
                store, "r1", Episode(id=f"ep-{i}", ts=float(i), intent="去充电")
            )

        result = await reflect_and_distill(distiller, store, "r1", prune=True)

        assert isinstance(result, DistillResult)
        assert result.episodes_seen == 3
        # 蒸馏结果写进了 facts / prefs。
        fact = await store.aget(ns("r1", KIND_FACTS), "充电桩位置")
        pref = await store.aget(ns("r1", KIND_PREFS), "播报语言")
        assert fact.value == {"value": "(3,2)"}
        assert pref.value == {"value": "中文"}
        # prune=True：消化后 episodic 清空。
        assert await read_episodes(store, "r1") == []


async def test_distill_skips_when_below_min_episodes():
    distiller = make_model(responses=[AIMessage("fact: x = y")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await record_episode(store, "r1", Episode(id="ep-1", intent="a"))
        result = await reflect_and_distill(distiller, store, "r1", min_episodes=5)
    assert result.written == [] and result.episodes_seen == 1


async def test_distill_does_not_prune_on_empty_result():
    """模型畸形输出/降级 → 解析为空时不得 prune，避免无谓丢失经验（codex review P1）。"""
    distiller = make_model(responses=[AIMessage("（无可解析行，模型畸形输出）")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await record_episode(store, "r1", Episode(id="ep-1", ts=1.0, intent="a"))
        result = await reflect_and_distill(distiller, store, "r1", prune=True)
        assert result.written == []
        assert [e.id for e in await read_episodes(store, "r1")] == ["ep-1"]  # 仍在


async def test_distill_bounds_batch_to_max_episodes():
    """单次只处理最近 max_episodes 条，prune 也只删处理过的批次（codex review P2）。"""
    distiller = make_model(responses=[AIMessage("fact: k = v")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        for i in range(5):
            await record_episode(
                store, "r1", Episode(id=f"ep-{i}", ts=float(i), intent="x")
            )
        result = await reflect_and_distill(
            distiller, store, "r1", max_episodes=2, prune=True
        )
        assert result.episodes_seen == 2  # 只看最近 2 条（ep-3, ep-4）
        remaining = {e.id for e in await read_episodes(store, "r1")}
        assert remaining == {"ep-0", "ep-1", "ep-2"}  # 仅处理过的批次被清


# --------------------------------------------------------------------------- #
# 验收②：蒸馏出的偏好在后续回合被注入并影响决策
# --------------------------------------------------------------------------- #


async def test_ac_distilled_pref_injected_into_later_turn():
    robot_id = "robot-1"
    distiller = make_model(responses=[AIMessage("pref: 播报语言 = 只讲中文")])
    turn_model = make_model(responses=[AIMessage("好的")])
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await record_episode(store, robot_id, Episode(id="ep-1", intent="对话"))
        await reflect_and_distill(distiller, store, robot_id)

        # 蒸馏后开一个新回合：pre_model_hook 应把该偏好注入 LLM 输入。
        agent = build_robot_agent(model=turn_model, store=store, robot_id=robot_id)
        await agent.ainvoke({"messages": [HumanMessage("在吗")]})

    sys_texts = [
        m.content for m in turn_model.received[0] if isinstance(m, SystemMessage)
    ]
    assert any("只讲中文" in t for t in sys_texts), sys_texts


# --------------------------------------------------------------------------- #
# driver 衔接：make_reflect_hook 自动记录 + 周期蒸馏
# --------------------------------------------------------------------------- #


async def test_reflect_hook_records_and_periodically_distills():
    robot_id = "robot-1"
    effectors = build_effectors("mock")
    turn_model = make_model(responses=[AIMessage("回合1"), AIMessage("回合2")])
    distiller = make_model(responses=[AIMessage("pref: 步速 = 缓慢")])

    inbox = PriorityInbox()
    await inbox.put(user_message("任务一"))
    await inbox.put(user_message("任务二"))

    async with (
        AsyncSqliteSaver.from_conn_string(":memory:") as saver,
        AsyncSqliteStore.from_conn_string(":memory:") as store,
    ):
        agent = build_robot_agent(
            model=turn_model,
            effectors=effectors,
            store=store,
            checkpointer=saver,
            robot_id=robot_id,
        )
        driver = Driver(agent, inbox, idle_tick=0.01)
        # 每 2 回合蒸馏一次；蒸馏后 prune，便于断言。
        driver.on_turn = make_reflect_hook(
            store, robot_id, distill_model=distiller, reflect_every=2, prune=True
        )

        await driver.run_once()  # 回合1：记录，未到蒸馏点
        eps_after_1 = await read_episodes(store, robot_id)
        await driver.run_once()  # 回合2：记录 + 触发蒸馏（prune 清空 episodic）
        # 在 store 关闭前读取断言素材。
        pref = await store.aget(ns(robot_id, KIND_PREFS), "步速")
        episodic_after = await read_episodes(store, robot_id)

    assert len(eps_after_1) == 1  # 第一回合已被记录
    assert pref.value == {"value": "缓慢"}  # 第二回合触发蒸馏写入 pref
    assert episodic_after == []  # prune 生效，已消化的经历清空


async def test_reflect_hook_skips_interrupted_and_counts_completed():
    """暂停回合不记录；蒸馏按**已完成**回合计数（暂停回合不计入），codex review P2。"""
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        distiller = make_model(responses=[AIMessage("pref: 步速 = 缓慢")])
        hook = make_reflect_hook(store, "r1", distill_model=distiller, reflect_every=2)

        def _turn(index, interrupted):
            t = _Turn(
                index,
                user_message("x"),
                {"messages": [HumanMessage("x"), AIMessage("ok")]},
                "user_msg",
            )
            t.interrupted = interrupted
            return t

        await hook(_turn(1, interrupted=True))  # 暂停：不记录、不计数
        assert await read_episodes(store, "r1") == []

        await hook(_turn(2, interrupted=False))  # 完成回合 1
        # completed=1（非 index=2）→ 此时不应触发蒸馏。
        assert await store.aget(ns("r1", KIND_PREFS), "步速") is None

        await hook(_turn(3, interrupted=False))  # 完成回合 2 → completed=2 触发蒸馏
        pref = await store.aget(ns("r1", KIND_PREFS), "步速")
        recorded = await read_episodes(store, "r1")

    assert pref.value == {"value": "缓慢"}
    assert len(recorded) == 2  # 仅两个完成回合被记录，暂停回合被跳过
