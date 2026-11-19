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
