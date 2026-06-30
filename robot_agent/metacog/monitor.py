"""元认知监控（设计 §8.5，实现计划 §P8）：把检测挂进 `pre_model_hook`。

在调 LLM 前检查「是否陷入循环 / 是否超步数预算」，越界则二选一：

- **escalate**（默认）：`interrupt` 暂停上报，等人工/规则 `Command(resume=...)` 决定是否继续
  （对应设计「不确定即上报求助」）。resume 后本回合放行一次。
- **warn**：注入一条元认知告警 `SystemMessage`，促使 LLM 收敛（换策略或直接结束）。

`MetacogPolicy.metrics` 累计循环/预算越界次数，供运维导出（指标导出，§8.5）。
本 hook 是现有 memory `pre_model_hook` 的**装饰器**：先检测，再委托内层做记忆注入。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.types import interrupt

from robot_agent import prompts
from robot_agent.hooks import insert_after_leading_system_messages
from robot_agent.metacog.detect import detect_loop, steps_used

ESCALATE = "escalate"
WARN = "warn"


@dataclass
class MetacogPolicy:
    """自我监控策略（设计 §8.5）。`max_repeats`/`max_steps` 为 None 则不查该项。"""

    max_repeats: int | None = 3  # 连续相同工具调用达此数 → 循环
    max_steps: int | None = None  # 本回合工具调用步数上限
    on_breach: str = ESCALATE  # 越界处理：escalate（interrupt 上报）| warn（注入告警）
    metrics: dict[str, int] = field(
        default_factory=lambda: {"loops": 0, "budget_breaches": 0, "escalations": 0}
    )
    # 已计数的越界事件指纹：escalate 的 interrupt 会令节点 resume 时重放本 hook，
    # 据此去重，避免同一事件被重复计入 metrics。
    _counted: set = field(default_factory=set, repr=False, compare=False)


def make_monitor_hook(
    inner_hook: Callable[[dict], Awaitable[dict]], policy: MetacogPolicy
) -> Callable[[dict], Awaitable[dict]]:
    """用元认知检测装饰一个 `pre_model_hook`（如 memory 注入 hook）。"""

    async def hook(state: dict) -> dict:
        messages: list[BaseMessage] = list(state.get("messages") or [])
        loop = detect_loop(messages, policy.max_repeats) if policy.max_repeats else None
        used = steps_used(messages)
        over_budget = policy.max_steps is not None and used >= policy.max_steps

        if loop or over_budget:
            reason = "检测到重复决策循环" if loop else "本回合步数预算耗尽"
            # 指纹去重：interrupt 在 resume 时重放本 hook → 同指纹只计一次。
            # 用「本回合 human 计数」作判别键：回合内多步 / resume 重放保持不变（同一事件不重复计，
            # 不含步数 used——否则同一持续循环随步数变化被重复计入）；新回合（新增 human 锚点）则
            # 区分开，使跨回合的不同越界各计一次（codex review：避免全局指纹把所有循环坍缩为一次）。
            turn_key = sum(1 for m in messages if getattr(m, "type", None) == "human")
            loop_sig = tuple(loop["signature"]) if loop else None
            incident = (turn_key, loop_sig, over_budget)
            first_time = incident not in policy._counted
            if first_time:
                policy._counted.add(incident)
                if loop:
                    policy.metrics["loops"] += 1
                if over_budget:
                    policy.metrics["budget_breaches"] += 1
            report = {
                "type": "metacog_escalation",
                "reason": reason,
                "loop": loop,
                "steps": used,
            }
            if policy.on_breach == ESCALATE:
                if first_time:
                    policy.metrics["escalations"] += 1
                # 上报暂停；resume 后 hook 从头重放，interrupt 直接返回 resume 值，
                # 落到下方正常注入 → 本回合放行一次（避免再次自我阻断而死锁）。
                interrupt(report)
            else:  # WARN：注入告警，促使 LLM 收敛
                result = await inner_hook(state)
                warn = SystemMessage(prompts.render("metacog_warn", reason=reason))
                msgs = result.get("llm_input_messages", messages)
                result["llm_input_messages"] = insert_after_leading_system_messages(
                    list(msgs), warn
                )
                return result

        return await inner_hook(state)

    return hook
