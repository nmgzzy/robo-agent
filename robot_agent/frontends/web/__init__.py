"""Web 通道：纯 stdlib（http.server + SSE）的本地调试控制台，零第三方依赖。

`python -m robot_agent.frontends.web` 起一个常驻 agent + 本地网页，可对话 / 看历史 /
看记忆 / 看可调用工具。实现见 server.py（`WebChannel`），入口见 __main__.py。
"""

from __future__ import annotations

from robot_agent.frontends.web.server import WebChannel

__all__ = ["WebChannel"]
