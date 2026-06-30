"""Web 控制台入口：`python -m robot_agent.frontends.web`。

装配一个常驻 agent（默认 mock 模型 + 内存存储，离线即可点亮），起本地网页通道，
常驻到 Ctrl-C。嵌入式落盘可用 `--sqlite <path>` 切换到 SQLite 短期/长期记忆。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib

from robot_agent.frontends.service import build_default_service
from robot_agent.frontends.web.server import WebChannel


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m robot_agent.frontends.web",
        description="机器人 Agent 本地 Web 调试控制台（对话/历史/记忆/工具）。",
    )
    p.add_argument("--host", default="127.0.0.1", help="绑定地址（默认仅本地）")
    p.add_argument("--port", type=int, default=8080, help="端口（默认 8080）")
    p.add_argument("--robot-id", default="robot-1", help="机器人/数字个体 ID")
    p.add_argument(
        "--idle-prompt",
        default=None,
        help="空闲自发回合提示（不填则待机最省电）",
    )
    p.add_argument(
        "--sqlite",
        default=None,
        help="SQLite 文件路径（落盘短期/长期记忆；不填用内存存储）",
    )
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> None:
    checkpointer = store = None
    stack = contextlib.AsyncExitStack()
    async with stack:
        if args.sqlite:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            from langgraph.store.sqlite.aio import AsyncSqliteStore

            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(args.sqlite)
            )
            store = await stack.enter_async_context(
                AsyncSqliteStore.from_conn_string(args.sqlite)
            )

        service = build_default_service(
            checkpointer=checkpointer,
            store=store,
            robot_id=args.robot_id,
            idle_prompt=args.idle_prompt,
        )
        await service.start()
        channel = WebChannel(host=args.host, port=args.port)
        await channel.start(service)
        host, port = channel.address
        print(f"机器人 Agent Web 控制台已启动 → http://{host}:{port}")
        print("按 Ctrl-C 停止。")
        try:
            await asyncio.Event().wait()  # 常驻直到被取消
        finally:
            await channel.stop()
            await service.stop()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
