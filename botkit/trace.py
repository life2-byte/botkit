"""Observability: every model call, tool call, error and graph step is recorded as an event.

    bot = Agent(model, tools=True, trace=True)
    bot.chat("...")
    print(bot.trace.summary())      # counts, timings, tokens
    bot.trace.print()               # readable timeline
    bot.trace.to_json()             # for logs / dashboards

Events are also sent to Python's `logging` module (logger name "botkit"), so a normal logging setup works:
    import logging; logging.basicConfig(level=logging.INFO)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("botkit")


@dataclass
class Event:
    kind: str
    t: float  # seconds since the trace started
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        who = self.data.get("agent") or self.data.get("node") or ""
        extra = {k: v for k, v in self.data.items() if k not in ("agent", "node", "result", "args")}
        brief = ", ".join(f"{k}={str(v)[:60]}" for k, v in extra.items())
        return f"{self.t:7.2f}s  {self.kind:<12} {who:<14} {brief}".rstrip()


class Trace:
    """Thread-safe event log shared by an agent, or by every agent of a graph."""

    def __init__(self, max_events: int = 5000) -> None:
        self.events: deque[Event] = deque(maxlen=max_events)
        self.max_events = max_events
        self._t0 = time.monotonic()
        self._lock = threading.Lock()

    def add(self, kind: str, **data: Any) -> None:
        with self._lock:
            self.events.append(Event(kind, time.monotonic() - self._t0, data))  # oldest events drop when full

    def clear(self) -> None:
        with self._lock:
            self.events.clear()
            self._t0 = time.monotonic()

    def kinds(self, kind: str | None = None) -> list[Event]:
        return [e for e in self.events if kind is None or e.kind == kind]

    def flow(self) -> list[str]:
        """Names of the graph nodes in the order they started: ['planner', 'worker', 'reviewer', 'worker', ...]"""
        return [e.data["node"] for e in self.events if e.kind == "node_start"]

    def summary(self) -> dict[str, Any]:
        llm = self.kinds("llm_call")
        tools = self.kinds("tool_result")
        return {
            "events": len(self.events),
            "llm_calls": len(llm),
            "llm_seconds": round(sum(e.data.get("seconds", 0) for e in llm), 3),
            "prompt_tokens": sum(e.data.get("prompt_tokens", 0) or 0 for e in llm),
            "completion_tokens": sum(e.data.get("completion_tokens", 0) or 0 for e in llm),
            "tool_calls": len(self.kinds("tool_call")),
            "tool_errors": sum(1 for e in tools if str(e.data.get("result", "")).startswith("Error")),
            "errors": len(self.kinds("error")),
            "nodes_run": len(self.kinds("node_start")),
            "total_seconds": round(self.events[-1].t, 3) if self.events else 0.0,
        }

    def print(self, file: Any = None) -> None:
        for e in self.events:
            print(e, file=file)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(
            [{"kind": e.kind, "t": round(e.t, 3), **{k: _clip(v) for k, v in e.data.items()}} for e in self.events],
            indent=indent,
            default=str,
            ensure_ascii=False,
        )


def _clip(v: Any, n: int = 500) -> Any:
    return v[:n] + "..." if isinstance(v, str) and len(v) > n else v
