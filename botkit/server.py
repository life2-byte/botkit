"""A small HTTP API so any web/mobile UI can talk to your bot (standard library only).

    POST /chat    {"message": "...", "session": "user42"}  ->  {"reply": "...", "files": [{"name", "url"}]}
    GET  /files/<session>/<path>                           ->  download a file the bot created
    GET  /health

Each `session` is an independent user: own history, own workspace folder. Set `token=` and send
`Authorization: Bearer <token>` (file downloads also accept `?token=`). Listening on anything other than
localhost without a token is refused.
"""
from __future__ import annotations

import dataclasses
import hmac
import json
import mimetypes
import os
import re
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from .errors import BotError, LLMError

_SESSION = re.compile(r"[A-Za-z0-9_-]{1,64}")
_MAX_BODY = 1_000_000
_MAX_MESSAGE = 20_000
_MAX_DOWNLOAD = 50_000_000
_LOCAL = ("127.0.0.1", "localhost", "::1")


def _snapshot(root: str) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(base, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            out[os.path.relpath(full, root).replace(os.sep, "/")] = (st.st_mtime_ns, st.st_size)
            if len(out) >= 5000:
                return out
    return out


class _Sessions:
    def __init__(self, agent: Any, cap: int) -> None:
        self.base, self.cap = agent, cap
        self._agents: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, sid: str) -> Any:
        with self._lock:
            agent = self._agents.get(sid)
            if agent is None:
                agent = self._agents[sid] = self.base.fork(sid)
                while len(self._agents) > self.cap:
                    self._agents.popitem(last=False)
            else:
                self._agents.move_to_end(sid)
            return agent


def _make_handler(sessions: _Sessions, token: str | None, cors_origins: list[str] | None) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "botkit"

        def log_message(self, *args: Any) -> None:  # keep the console quiet
            pass

        # -- helpers
        def _cors(self) -> None:
            origin = self.headers.get("Origin")
            if cors_origins and origin and ("*" in cors_origins or origin in cors_origins):
                self.send_header("Access-Control-Allow-Origin", "*" if "*" in cors_origins else origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, code: int, obj: Any) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _authed(self, query: dict[str, list[str]]) -> bool:
            if token is None:
                return True
            supplied = self.headers.get("Authorization", "")
            supplied = supplied[7:] if supplied.lower().startswith("bearer ") else (query.get("token") or [""])[0]
            return hmac.compare_digest(supplied.encode(), token.encode())

        # -- routes
        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/health":
                return self._json(200, {"ok": True})
            if not self._authed(query):
                return self._json(401, {"error": "unauthorized"})
            m = re.fullmatch(r"/files/([^/]+)/(.+)", url.path)
            if not m or not _SESSION.fullmatch(m.group(1)):
                return self._json(404, {"error": "not found"})
            policy = dataclasses.replace(sessions.base.policy, workdir=sessions.base.policy.workdir / m.group(1))
            try:
                target = policy.resolve(unquote(m.group(2)))
            except PermissionError:
                return self._json(403, {"error": "forbidden"})
            if not target.is_file() or target.stat().st_size > _MAX_DOWNLOAD:
                return self._json(404, {"error": "not found"})
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="{quote(target.name)}"')
            self.send_header("X-Content-Type-Options", "nosniff")
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            if url.path != "/chat":
                return self._json(404, {"error": "not found"})
            if not self._authed(parse_qs(url.query)):
                return self._json(401, {"error": "unauthorized"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "bad Content-Length"})
            if length <= 0 or length > _MAX_BODY:
                return self._json(413 if length > _MAX_BODY else 400, {"error": "body missing or too large"})
            try:
                payload = json.loads(self.rfile.read(length))
                message, sid = payload["message"], payload.get("session", "default")
                if not isinstance(message, str) or not message.strip() or len(message) > _MAX_MESSAGE:
                    raise ValueError("message")
                if not isinstance(sid, str) or not _SESSION.fullmatch(sid):
                    raise ValueError("session")
            except (ValueError, KeyError, TypeError):
                return self._json(400, {"error": 'expected JSON {"message": "...", "session": "id"}'})
            agent = sessions.get(sid)
            root = str(agent.policy.workdir)
            before = _snapshot(root)
            try:
                reply = agent.chat(message)
            except LLMError as e:
                return self._json(502, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - never leak internals to the client
                print(f"botkit server error: {type(e).__name__}: {e}", file=sys.stderr)
                return self._json(500, {"error": "internal error"})
            after = _snapshot(root)
            changed = [n for n, sig in after.items() if before.get(n) != sig][:50]
            files = [{"name": n, "url": f"/files/{sid}/{quote(n)}"} for n in changed]
            self._json(200, {"reply": reply, "files": files})

    return Handler


def serve(
    agent: Any,
    host: str = "127.0.0.1",
    port: int = 8000,
    token: str | None = None,
    cors_origins: list[str] | None = None,
    max_sessions: int = 200,
    block: bool = True,
) -> ThreadingHTTPServer:
    """Start the HTTP API. With block=False the server runs in a background thread and is returned
    (call .shutdown() to stop it)."""
    if token is None and host not in _LOCAL:
        raise BotError("Refusing to listen on a non-local address without token=... (anyone could use your bot).")
    httpd = ThreadingHTTPServer((host, port), _make_handler(_Sessions(agent, max_sessions), token, cors_origins))
    httpd.daemon_threads = True
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    print(f"botkit listening on http://{host}:{httpd.server_address[1]}  (POST /chat)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        agent.close()
    return httpd
