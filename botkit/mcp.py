"""MCP bridge: connect to MCP servers (stdio / HTTP / SSE) and expose their tools as botkit Tools.

Works with both the 1.x and 2.x `mcp` SDKs. The async SDK runs on one shared background event loop
thread, so the rest of botkit (and your code) stays plain synchronous.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import shlex
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from .tool import Tool

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="botkit-mcp", daemon=True).start()
            _loop = loop
        return _loop


def _leaf(err: BaseException) -> BaseException:
    """Unwrap (nested) exception groups to the first real error."""
    while getattr(err, "exceptions", None):
        err = err.exceptions[0]  # type: ignore[attr-defined]
    return err


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:64]


class MCP:
    """One MCP server.

        MCP("npx", "-y", "@modelcontextprotocol/server-filesystem", ".")   # local process (stdio)
        MCP("https://example.com/mcp", headers={"Authorization": "Bearer ..."})  # remote (HTTP)
    """

    def __init__(
        self,
        command: str,
        *args: str,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        name: str | None = None,
        timeout: float = 30.0,
        call_timeout: float = 60.0,
        prefix: bool = True,
        debug: bool = False,
    ) -> None:
        self.url = command if command.startswith(("http://", "https://")) else None
        self.command = None if self.url else command
        self.args = list(args)
        self.env = env
        self.headers = headers
        self.name = name
        self.timeout = timeout
        self.call_timeout = call_timeout
        self.prefix = prefix
        self.debug = debug
        self._session: Any = None
        self._raw_tools: list[Any] = []
        self._runner: concurrent.futures.Future | None = None
        self._ready: concurrent.futures.Future | None = None
        self._stop: asyncio.Event | None = None
        self._start_lock = threading.RLock()

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> "MCP":
        """Build from one entry of the standard `mcpServers` config (same format Claude Desktop uses)."""
        target = cfg.get("url") or cfg.get("command")
        if not target:
            raise ValueError(f"MCP server '{name}' needs a 'command' or a 'url'")
        return cls(target, *cfg.get("args", []), env=cfg.get("env"), headers=cfg.get("headers"), name=name)

    # ---- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        with self._start_lock:  # several agents may share one server and start it at the same moment
            self._start()

    def _start(self) -> None:
        if self._session is not None:
            return
        try:
            import mcp  # noqa: F401
        except ImportError:
            raise ImportError("MCP support is not installed. Run: pip install 'botkit[mcp]'") from None
        self._ready = concurrent.futures.Future()
        self._runner = asyncio.run_coroutine_threadsafe(self._main(), _get_loop())
        try:
            self._ready.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            self._runner.cancel()
            raise TimeoutError(f"MCP server '{self.name}' did not start within {self.timeout:g}s") from None

    def close(self) -> None:
        if self._runner is None:
            return
        if self._stop is not None:
            _get_loop().call_soon_threadsafe(self._stop.set)
        try:
            self._runner.result(timeout=10)
        except (Exception, asyncio.CancelledError):
            pass
        self._runner, self._session, self._stop = None, None, None

    def __enter__(self) -> "MCP":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- connection (runs on the background loop, in ONE task so anyio scopes stay valid) ----
    async def _main(self) -> None:
        assert self._ready is not None
        try:
            async with AsyncExitStack() as stack:
                from mcp import ClientSession

                read, write = await self._connect(stack)
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                raw = await self._list_all(session)
                self._raw_tools, self._session = raw, session
                self._stop = asyncio.Event()
                self._ready.set_result(None)
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001
            if not self._ready.done():
                self._ready.set_exception(_leaf(e))
            if isinstance(e, asyncio.CancelledError):
                raise
        finally:
            self._session = None

    async def _connect(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        if self.url is None:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
            errlog = None if self.debug else stack.enter_context(open(os.devnull, "w"))
            cm = stdio_client(params, errlog=errlog) if errlog else stdio_client(params)
            streams = await stack.enter_async_context(cm)
        elif self.url.rstrip("/").endswith("/sse"):
            from mcp.client.sse import sse_client

            streams = await stack.enter_async_context(sse_client(self.url, headers=self.headers or None))
        else:
            streams = await stack.enter_async_context(self._http_transport(stack))
        return streams[0], streams[1]

    def _http_transport(self, stack: AsyncExitStack) -> Any:
        try:  # mcp 2.x
            from mcp.client.streamable_http import streamable_http_client as connect

            kwargs: dict[str, Any] = {}
            if self.headers:
                try:
                    import httpx2 as httpx_mod
                except ImportError:
                    import httpx as httpx_mod
                client = httpx_mod.AsyncClient(headers=self.headers, timeout=self.call_timeout)
                stack.push_async_callback(client.aclose)
                kwargs["http_client"] = client
            return connect(self.url, **kwargs)
        except ImportError:  # mcp 1.x
            from mcp.client.streamable_http import streamablehttp_client

            return streamablehttp_client(self.url, headers=self.headers or None)

    @staticmethod
    async def _list_all(session: Any) -> list[Any]:
        res = await session.list_tools()
        tools = list(res.tools)
        cursor = getattr(res, "nextCursor", None) or getattr(res, "next_cursor", None)
        for _ in range(20):  # follow pagination, bounded
            if not cursor:
                break
            try:
                from mcp_types import PaginatedRequestParams  # mcp 2.x

                res = await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
            except ImportError:  # mcp 1.x
                res = await session.list_tools(cursor=cursor)
            tools.extend(res.tools)
            cursor = getattr(res, "nextCursor", None) or getattr(res, "next_cursor", None)
        return tools

    # ---- using the server ------------------------------------------------------
    def tools(self) -> list[Tool]:
        self.start()
        result = []
        for raw in self._raw_tools:
            schema = getattr(raw, "inputSchema", None) or getattr(raw, "input_schema", None) or {}
            schema = dict(schema) if schema else {"type": "object", "properties": {}}
            full = f"{self.name}_{raw.name}" if self.prefix and self.name else raw.name
            result.append(
                Tool(
                    name=_safe_name(full),
                    description=(getattr(raw, "description", None) or raw.name)[:1000],
                    parameters=schema,
                    fn=self._caller(raw.name),
                )
            )
        return result

    def _caller(self, tool_name: str) -> Any:
        return lambda **kwargs: self.call(tool_name, kwargs)

    def call(self, tool_name: str, arguments: dict) -> str:
        if self._session is None:  # server died or never started: try once to (re)connect
            self.start()
        future = asyncio.run_coroutine_threadsafe(self._session.call_tool(tool_name, arguments), _get_loop())
        try:
            res = future.result(timeout=self.call_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"MCP tool '{tool_name}' timed out after {self.call_timeout:g}s") from None
        parts = []
        for block in getattr(res, "content", None) or []:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else f"[{getattr(block, 'type', 'content')}]")
        text = "\n".join(parts)
        if not text:
            structured = getattr(res, "structuredContent", None) or getattr(res, "structured_content", None)
            text = json.dumps(structured, default=str) if structured else "OK"
        if getattr(res, "isError", False) or getattr(res, "is_error", False):
            raise RuntimeError(text)
        return text


def normalize(spec: Any) -> list[MCP]:
    """Accept the many ways a user may describe MCP servers and return a list of MCP objects.

    - an MCP object, or a list of them
    - a command string: "npx -y @modelcontextprotocol/server-filesystem ."
    - a URL string
    - a path to a JSON file, or a dict, in the standard {"mcpServers": {name: {command, args, env}}} format
    """
    if spec is None or spec is False:
        return []
    if isinstance(spec, MCP):
        return [spec]
    if isinstance(spec, (str, os.PathLike)):
        text = str(spec)
        if text.endswith(".json") and Path(text).expanduser().is_file():
            return normalize(json.loads(Path(text).expanduser().read_text(encoding="utf-8")))
        if text.startswith(("http://", "https://")):
            return [MCP(text)]
        return [MCP(*shlex.split(text))]
    if isinstance(spec, dict):
        servers = spec.get("mcpServers", spec)
        return [MCP.from_config(name, cfg) for name, cfg in servers.items()]
    out: list[MCP] = []
    for item in spec:
        out.extend(normalize(item))
    return out
