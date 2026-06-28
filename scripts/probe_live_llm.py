#!/usr/bin/env python3
"""真实 LLM 兼容性探针：对话、历史、工具调用与跨线程长期记忆。

本脚本会访问 `.env` 或命令行指定的远程模型并产生实际费用，刻意不放入
`make test` 的离线回归集合。默认限制单次模型输出、请求超时、用例超时和图递归步数。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_agent import (  # noqa: E402 - 支持从 scripts/ 直接执行
    build_effectors,
    build_robot_agent,
    make_model,
    prompts,
)
from robot_agent.llm import (  # noqa: E402
    load_llm_config_from_env,
    merge_llm_config,
    resolve_model_name,
)
from robot_agent.memory import KIND_FACTS, ns  # noqa: E402
from robot_agent.tools import build_robot_tools  # noqa: E402

ALL_CHECKS = frozenset({"chat", "history", "agent-tool", "forced-tool", "memory"})


@dataclass
class CheckResult:
    """一个兼容性检查的结构化结果。"""

    name: str
    passed: bool
    duration_seconds: float
    details: dict[str, Any]


class UsageTracker:
    """按消息 id 去重汇总 provider 返回的 token usage。"""

    def __init__(self) -> None:
        self._seen: set[Any] = set()
        self.response_count = 0
        self.totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def add(self, messages: list[BaseMessage]) -> None:
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            identity: Any = message.id
            if identity is None:
                identity = (
                    str(message.content),
                    repr(message.tool_calls),
                    repr(message.usage_metadata),
                )
            if identity in self._seen:
                continue
            self._seen.add(identity)
            self.response_count += 1
            usage = message.usage_metadata or {}
            for key in self.totals:
                self.totals[key] += int(usage.get(key, 0) or 0)


def _message_text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def _tool_calls(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            calls.append({"name": call.get("name"), "args": call.get("args")})
    return calls


def _short(value: Any, limit: int = 600) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}…"


def _parse_checks(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return set(ALL_CHECKS)
    checks = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = checks - ALL_CHECKS
    if not checks or unknown:
        valid = ", ".join(sorted(ALL_CHECKS))
        raise ValueError(f"未知 checks={sorted(unknown)}；可选：all, {valid}")
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用真实 `.env` LLM 检查普通对话、线程历史、OpenAI-compatible "
            "tool calling 和 SQLite 长期记忆。"
        )
    )
    parser.add_argument(
        "--profile", default="fast", choices=("fast", "smart", "vision")
    )
    parser.add_argument("--provider", help="覆盖 `.env` 的 LLM_PROVIDER")
    parser.add_argument("--model", help="覆盖所选 profile 的模型名")
    parser.add_argument("--base-url", help="覆盖 `.env` 的 LLM_BASE_URL")
    parser.add_argument(
        "--checks",
        default="all",
        help="逗号分隔：chat,history,agent-tool,forced-tool,memory；默认 all",
    )
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="客户端请求重试次数；默认 0，避免兼容性探针放大费用和等待时间",
    )
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--case-timeout", type=float, default=120.0)
    parser.add_argument("--recursion-limit", type=int, default=6)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="保留 SQLite 文件的目录；默认使用并自动删除 `/tmp` 临时目录",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="把结构化报告写入指定 JSON 文件",
    )
    return parser


class LiveProbe:
    """在同一模型和临时 SQLite 上执行一组有界的真实兼容性检查。"""

    def __init__(self, args: argparse.Namespace, checks: set[str]) -> None:
        self.args = args
        self.checks = checks
        self.results: list[CheckResult] = []
        self.usage = UsageTracker()
        self.suffix = uuid.uuid4().hex[:8]
        self.chat_marker = f"CHAT-{self.suffix}"
        self.memory_key = f"probe_operator_{self.suffix}"
        self.memory_value = f"MEMORY-{uuid.uuid4().hex[:10]}"
        self.robot_id = f"live-probe-{self.suffix}"

    def record(
        self,
        name: str,
        passed: bool,
        started: float,
        **details: Any,
    ) -> None:
        result = CheckResult(
            name=name,
            passed=passed,
            duration_seconds=round(time.monotonic() - started, 3),
            details=details,
        )
        self.results.append(result)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name} ({result.duration_seconds:.3f}s)", flush=True)
        for key, value in details.items():
            print(f"  {key}: {_short(value)}", flush=True)

    async def invoke_agent(
        self,
        agent: Any,
        content: str,
        thread_id: str,
    ) -> dict[str, Any]:
        invocation = agent.ainvoke(
            {"messages": [HumanMessage(content=content)]},
            {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": self.args.recursion_limit,
            },
        )
        result = await asyncio.wait_for(invocation, timeout=self.args.case_timeout)
        self.usage.add(result["messages"])
        return result

    async def check_chat_history_and_agent_tool(
        self, agent: Any, effectors: Any
    ) -> None:
        requested = self.checks & {"chat", "history", "agent-tool"}
        if not requested:
            return

        thread_id = f"history-{self.suffix}"
        started = time.monotonic()
        try:
            first = await self.invoke_agent(
                agent,
                prompts.render("live_probe_chat", marker=self.chat_marker),
                thread_id,
            )
            first_reply = _message_text(first["messages"][-1])
            first_calls = _tool_calls(first["messages"])
            if "chat" in self.checks:
                self.record(
                    "chat",
                    self.chat_marker in first_reply,
                    started,
                    reply=first_reply,
                    tool_calls=first_calls,
                )
        except Exception as exc:
            for name in sorted(requested):
                self.record(name, False, started, error=f"{type(exc).__name__}: {exc}")
            return

        if not ({"history", "agent-tool"} & self.checks):
            return

        started = time.monotonic()
        before_log = len(effectors["speaker"].log)
        try:
            second = await self.invoke_agent(
                agent,
                prompts.render("live_probe_history_tool"),
                thread_id,
            )
            second_reply = _message_text(second["messages"][-1])
            calls = _tool_calls(second["messages"])
            new_speaker_log = effectors["speaker"].log[before_log:]
            evidence = f"{second_reply} {calls} {new_speaker_log}"
            if "history" in self.checks:
                self.record(
                    "history",
                    self.chat_marker in evidence,
                    started,
                    reply=second_reply,
                    expected_marker=self.chat_marker,
                )
            if "agent-tool" in self.checks:
                called = any(call["name"] == "speak" for call in calls)
                executed = bool(new_speaker_log) and self.chat_marker in str(
                    new_speaker_log
                )
                self.record(
                    "agent-tool",
                    called and executed,
                    started,
                    tool_calls=calls,
                    speaker_log=new_speaker_log,
                )
        except Exception as exc:
            for name in sorted({"history", "agent-tool"} & self.checks):
                self.record(name, False, started, error=f"{type(exc).__name__}: {exc}")

    async def check_forced_tool(self, model: Any, effectors: Any) -> None:
        if "forced-tool" not in self.checks:
            return
        started = time.monotonic()
        try:
            speak = next(
                tool for tool in build_robot_tools(effectors) if tool.name == "speak"
            )
            bound = model.bind_tools([speak], tool_choice="speak")
            invocation = bound.ainvoke(
                [HumanMessage(content=prompts.render("live_probe_forced_tool"))]
            )
            response = await asyncio.wait_for(
                invocation, timeout=self.args.case_timeout
            )
            self.usage.add([response])
            calls = _tool_calls([response])
            passed = any(call["name"] == "speak" for call in calls)
            self.record(
                "forced-tool",
                passed,
                started,
                content=_message_text(response),
                tool_calls=calls,
                finish_reason=response.response_metadata.get("finish_reason"),
            )
        except Exception as exc:
            self.record(
                "forced-tool", False, started, error=f"{type(exc).__name__}: {exc}"
            )

    async def check_memory(self, agent: Any, store: AsyncSqliteStore) -> None:
        if "memory" not in self.checks:
            return

        started = time.monotonic()
        stored: Any = None
        try:
            write_result = await self.invoke_agent(
                agent,
                prompts.render(
                    "live_probe_memory_write",
                    key=self.memory_key,
                    value=self.memory_value,
                ),
                f"memory-write-{self.suffix}",
            )
            calls = _tool_calls(write_result["messages"])
            stored = await store.aget(ns(self.robot_id, KIND_FACTS), self.memory_key)
            stored_value = None if stored is None else stored.value
            called = any(call["name"] == "remember_fact" for call in calls)
            persisted = (
                isinstance(stored_value, dict)
                and stored_value.get("value") == self.memory_value
            )
            self.record(
                "memory-write",
                called and persisted,
                started,
                tool_calls=calls,
                stored_value=stored_value,
            )
        except Exception as exc:
            self.record(
                "memory-write", False, started, error=f"{type(exc).__name__}: {exc}"
            )

        started = time.monotonic()
        try:
            recall_result = await self.invoke_agent(
                agent,
                prompts.render("live_probe_memory_recall", key=self.memory_key),
                f"memory-recall-{self.suffix}",
            )
            reply = _message_text(recall_result["messages"][-1])
            self.record(
                "memory-recall",
                stored is not None and self.memory_value in reply,
                started,
                reply=reply,
                expected_value=self.memory_value,
            )
        except Exception as exc:
            self.record(
                "memory-recall", False, started, error=f"{type(exc).__name__}: {exc}"
            )

    async def run(self, work_dir: Path) -> None:
        effectors = build_effectors("mock")
        model = make_model(
            self.args.profile,
            provider=self.args.provider,
            model=self.args.model,
            base_url=self.args.base_url,
            max_tokens=self.args.max_tokens,
            max_retries=self.args.max_retries,
            temperature=0,
            timeout=self.args.request_timeout,
        )
        checkpoint_path = str(work_dir / "checkpoints.db")
        memory_path = str(work_dir / "memory.db")
        async with (
            AsyncSqliteSaver.from_conn_string(checkpoint_path) as checkpointer,
            AsyncSqliteStore.from_conn_string(memory_path) as store,
        ):
            agent = build_robot_agent(
                model=model,
                effectors=effectors,
                checkpointer=checkpointer,
                store=store,
                robot_id=self.robot_id,
                context_policy=None,
            )
            await self.check_chat_history_and_agent_tool(agent, effectors)
            await self.check_forced_tool(model, effectors)
            await self.check_memory(agent, store)


def _config_summary(args: argparse.Namespace) -> dict[str, Any]:
    config = merge_llm_config(
        load_llm_config_from_env(provider=args.provider),
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )
    return {
        "provider": config.provider,
        "profile": args.profile,
        "model": resolve_model_name(args.profile, config),
        "base_url": config.base_url or "<provider-default>",
        "api_key_configured": bool(config.api_key),
    }


async def _run(args: argparse.Namespace, checks: set[str]) -> tuple[LiveProbe, float]:
    probe = LiveProbe(args, checks)
    started = time.monotonic()
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        await probe.run(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(
            prefix="robot-agent-live-probe-", dir="/tmp"
        ) as tmp:
            await probe.run(Path(tmp))
    return probe, time.monotonic() - started


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        checks = _parse_checks(args.checks)
    except ValueError as exc:
        parser.error(str(exc))

    if args.max_tokens <= 0 or args.request_timeout <= 0 or args.case_timeout <= 0:
        parser.error("max-tokens、request-timeout、case-timeout 必须大于 0")
    if args.max_retries < 0:
        parser.error("max-retries 不能小于 0")
    if args.recursion_limit < 2:
        parser.error("recursion-limit 必须至少为 2")

    config = _config_summary(args)
    print("真实 LLM 兼容性探针（会访问远程 API 并产生费用）", flush=True)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    print(f"checks: {sorted(checks)}", flush=True)

    probe, duration = asyncio.run(_run(args, checks))
    passed = all(result.passed for result in probe.results)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "duration_seconds": round(duration, 3),
        "config": config,
        "limits": {
            "max_tokens_per_model_call": args.max_tokens,
            "max_retries": args.max_retries,
            "request_timeout_seconds": args.request_timeout,
            "case_timeout_seconds": args.case_timeout,
            "recursion_limit": args.recursion_limit,
        },
        "requested_checks": sorted(checks),
        "observed_model_responses": probe.usage.response_count,
        "token_usage": probe.usage.totals,
        "results": [asdict(result) for result in probe.results],
    }

    print("\nSUMMARY")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.json_report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
