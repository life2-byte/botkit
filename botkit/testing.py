"""Test helpers: try agents and whole workflows without calling a real model.

    from botkit import Agent
    from botkit.testing import FakeLLM, say, call, assert_flow

    bot = Agent(llm=FakeLLM(call("calculate", {"expression": "2+2"}), say("4")), tools=True)
    assert bot.chat("2+2?") == "4"

    # a workflow: give each agent its own scripted model
    g = chain(Agent(name="planner", llm=FakeLLM(say("plan"))), Agent(name="worker", llm=FakeLLM(say("done"))))
    assert g.chat("go") == "done"
    assert_flow(g, ["planner", "worker"])
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

from .errors import LLMError
from .trace import Trace


def say(text: str) -> dict:
    """A scripted assistant reply with plain text."""
    return {"role": "assistant", "content": text}


def call(name: str, args: dict | str | None = None, id: str = "c1") -> dict:
    """A scripted assistant reply that asks for one tool call."""
    raw = args if isinstance(args, str) else json.dumps(args or {})
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": id, "type": "function", "function": {"name": name, "arguments": raw}}]}


class FakeLLM:
    """A model that plays back scripted replies (and records what it was sent), safe to use from threads.

        FakeLLM(say("hi"), call("add", {"a": 1, "b": 2}), say("3"))      # in order
        FakeLLM(fn=lambda messages, tools: say("always this"))            # or decide per request
    Exceptions in the script are raised (e.g. LLMError("down")). `llm.seen` holds (messages, tools) of every call;
    `llm.prompts` the last user text of every call.
    """

    def __init__(self, *replies: Any, fn: Callable[[list[dict], Any], Any] | None = None) -> None:
        self.replies, self.fn, self.seen = list(replies), fn, []
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    @property
    def prompts(self) -> list[str]:
        return [next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "") for msgs, _ in self.seen]

    def chat(self, messages: list[dict], tools: Any = None) -> dict:
        with self._lock:
            self.seen.append((json.loads(json.dumps(messages)), tools))
            self.usage["calls"] += 1
            if self.fn is not None:
                reply = self.fn(messages, tools)
            elif self.replies:
                reply = self.replies.pop(0)
            else:
                raise LLMError("FakeLLM ran out of scripted replies")
        if isinstance(reply, Exception):
            raise reply
        return say(reply) if isinstance(reply, str) else reply

    def stream(self, messages: list[dict], tools: Any = None, on_token: Callable[[str], None] | None = None) -> dict:
        reply = self.chat(messages, tools)
        if on_token and reply.get("content"):
            for word in reply["content"].split(" "):
                on_token(word + " ")
        return reply


def _flow_of(obj: Any) -> list[str]:
    trace = obj if isinstance(obj, Trace) else getattr(obj, "trace", None)
    if trace is None:
        raise AssertionError("no trace to check: create the agent/graph with trace=True")
    return trace.flow()


def assert_flow(graph_or_trace: Any, expected: list[str]) -> None:
    """Assert the nodes ran in exactly this order, e.g. ['planner', 'worker', 'reviewer', 'worker', 'reviewer']."""
    got = _flow_of(graph_or_trace)
    assert got == expected, f"workflow ran {got}, expected {expected}"


def assert_tools_called(agent_or_trace: Any, expected: list[str]) -> None:
    """Assert which tools were called (in order)."""
    trace = agent_or_trace if isinstance(agent_or_trace, Trace) else agent_or_trace.trace
    got = [e.data["name"] for e in trace.kinds("tool_call")]
    assert got == expected, f"tools called {got}, expected {expected}"
