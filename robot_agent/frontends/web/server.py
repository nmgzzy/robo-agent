"""Web 通道：stdlib `http.server` + SSE 的极简前端，零第三方依赖（嵌入式友好）。

实现 `Channel` 协议（channel.py）。driver 跑在主 asyncio loop；HTTP 请求由
`ThreadingHTTPServer` 的工作线程处理，线程内用 `run_coroutine_threadsafe` 把 async 调用
桥回 driver 的 loop——这是 stdlib 同步 server 与 async 闭环之间的标准做法，避免引入
aiohttp/uvicorn 等重依赖。

路由（默认仅绑 127.0.0.1，本地调试足够；多机访问/鉴权留作后续）：
    GET  /                     单页前端（static/index.html）
    POST /api/chat             {text, thread_id?} 投递用户消息
    GET  /api/stream?thread_id 服务器推送事件（SSE）：增量回合广播
    GET  /api/history?thread_id 会话历史（短期记忆）
    GET  /api/memory           长期记忆（facts/episodic/prefs）
    GET  /api/tools            可调用工具
    GET  /api/health           健康度快照
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from robot_agent.frontends.service import DEFAULT_USER_THREAD, AgentService

_STATIC = Path(__file__).parent / "static"


class WebChannel:
    """把 `AgentService` 暴露为本地 Web 控制台的通道（实现 Channel 协议）。"""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    async def start(self, service: AgentService) -> None:
        """绑定门面并在后台线程起 HTTP 服务（driver 的 loop 由 service 暴露）。"""
        loop = service.loop or asyncio.get_running_loop()
        handler = _make_handler(service, loop)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="web-channel", daemon=True
        )
        self._thread.start()

    async def stop(self) -> None:
        """关闭 HTTP 服务并回收线程。"""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def address(self) -> tuple[str, int]:
        """实际绑定地址（port=0 时取系统分配的真实端口，便于测试）。"""
        if self._httpd is not None:
            return self._httpd.server_address[:2]
        return (self.host, self.port)


def _make_handler(service: AgentService, loop: asyncio.AbstractEventLoop):
    """构造绑定了 service / loop 的请求处理器类。"""

    def call(coro) -> Any:
        """把一个 async 调用桥回 driver 所在 loop，同步等待结果。"""
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # 静默默认 stderr 噪声
            pass

        # —— 响应辅助 ————————————————————————————————————————
        def _send_json(self, obj: Any, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.is_file():
                self._send_json({"error": "not found"}, status=404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # —— 路由 ————————————————————————————————————————————
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path in ("/", "/index.html"):
                self._send_file(_STATIC / "index.html", "text/html; charset=utf-8")
            elif path == "/api/stream":
                self._stream(query.get("thread_id", [DEFAULT_USER_THREAD])[0])
            elif path == "/api/history":
                thread_id = query.get("thread_id", [DEFAULT_USER_THREAD])[0]
                self._send_json(call(service.history(thread_id)))
            elif path == "/api/memory":
                self._send_json(call(service.memory()))
            elif path == "/api/tools":
                self._send_json(service.tools())
            elif path == "/api/health":
                self._send_json(call(service.health()))
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/chat":
                self._send_json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._send_json({"error": "bad json"}, status=400)
                return
            text = (data.get("text") or "").strip()
            if not text:
                self._send_json({"error": "empty text"}, status=400)
                return
            thread_id = data.get("thread_id") or DEFAULT_USER_THREAD
            call(service.submit_user_text(text, thread_id=thread_id))
            self._send_json({"ok": True})

        # —— SSE：把回合广播逐条推给浏览器 ————————————————————
        def _stream(self, thread_id: str) -> None:
            q = service.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                # 首帧注释立即冲刷，确认连接已建立。
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    fut = asyncio.run_coroutine_threadsafe(q.get(), loop)
                    try:
                        event = fut.result(timeout=15.0)
                    except (TimeoutError, asyncio.TimeoutError):
                        # 取消挂起的 q.get()，否则它会留在 loop 上吞掉下一条广播
                        # （回复丢失），且每次心跳/断连都堆积一个等待者。
                        fut.cancel()
                        self.wfile.write(b": ping\n\n")  # 心跳保活
                        self.wfile.flush()
                        continue
                    if thread_id and event.get("thread_id") not in (thread_id, None):
                        continue
                    payload = json.dumps(event, ensure_ascii=False, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                service.unsubscribe(q)

    return Handler
