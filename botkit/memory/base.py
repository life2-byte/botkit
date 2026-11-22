"""The storage contract. Implement this class to keep botkit's memory in ANY database."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Fact:
    id: int
    text: str


def trim_history(messages: list[dict], limit: int | None) -> list[dict]:
    """Keep roughly the last `limit` messages, but never start mid tool-call (must begin with a user message)."""
    if limit is None or len(messages) <= limit:
        return list(messages)
    kept = messages[-limit:]
    while kept and kept[0].get("role") != "user":
        kept = kept[1:]
    return kept


def estimate_tokens(messages: list[dict]) -> int:
    """Cheap token estimate (about 1 token per 3.5 characters, no tokenizer needed)."""
    chars = 0
    for m in messages:
        chars += len(str(m.get("content") or "")) + 12
        for tc in m.get("tool_calls") or []:
            chars += len(str(tc.get("function", {}).get("arguments", ""))) + len(str(tc.get("function", {}).get("name", "")))
    return int(chars / 3.5)


def compact_history(messages: list[dict], keep_chars: int = 1200) -> list[dict]:
    """Shorten big tool results in OLD turns (the model already used them). Returns copies; the stored history is untouched."""
    out = []
    for m in messages:
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, str) and len(c) > keep_chars:
            m = {**m, "content": c[:keep_chars] + f"\n...[old tool output shortened, {len(c) - keep_chars} characters]"}
        out.append(m)
    return out


def trim_to_tokens(messages: list[dict], budget: int | None) -> list[dict]:
    """Drop the oldest turns until the estimate fits `budget` tokens; always starts on a user message."""
    if not budget or estimate_tokens(messages) <= budget:
        return list(messages)
    kept = list(messages)
    while kept and estimate_tokens(kept) > budget:
        kept = kept[1:]
        while kept and kept[0].get("role") != "user":
            kept = kept[1:]
    return kept


class Storage(ABC):
    """Where memory lives. botkit ships SQLiteStorage and SupabaseStorage; write your own for Postgres,
    MongoDB, Redis, a file, anything. Seven small methods, all scoped by `session` (one session = one user).

    Messages are plain dicts (OpenAI chat format); store them as JSON and return them unchanged, oldest first.
    """

    @abstractmethod
    def add_messages(self, session: str, messages: list[dict]) -> None:
        """Append messages to a session's history, keeping their order."""

    @abstractmethod
    def get_messages(self, session: str, limit: int) -> list[dict]:
        """Return the last `limit` messages of a session, oldest first."""

    @abstractmethod
    def clear_messages(self, session: str) -> None:
        """Delete a session's history (facts are kept)."""

    @abstractmethod
    def add_fact(self, session: str, text: str) -> int:
        """Save a long-term fact and return its id."""

    @abstractmethod
    def list_facts(self, session: str, limit: int = 30) -> list[Fact]:
        """Return the newest `limit` facts of a session, oldest first."""

    @abstractmethod
    def delete_fact(self, session: str, fact_id: int) -> bool:
        """Delete one fact. Return False if it did not exist."""

    def close(self) -> None:
        """Release connections (optional)."""
