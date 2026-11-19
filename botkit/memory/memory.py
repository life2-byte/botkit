"""Memory = conversation history + long-term facts, on top of any Storage backend."""
from __future__ import annotations

import re
from pathlib import Path

from ..tool import Tool, tool
from .base import Fact, Storage, trim_history
from .sqlite import SQLiteStorage


class Memory:
    """
        Memory("bot.db")                                  # SQLite file
        Memory(SupabaseStorage(url, key))                 # Supabase
        Memory(MyStorage(), session="user42")             # your own backend

    - history: the conversation so far (survives restarts)
    - facts:   durable things worth remembering ("user's name is Ahmed") that the model saves/looks up via tools
    One storage can serve many sessions (one per user): `memory.for_session("user42")`.
    """

    def __init__(self, storage: Storage | str | Path | None = None, session: str = "default", max_messages: int = 40) -> None:
        if storage is None:
            storage = SQLiteStorage(":memory:")
        elif isinstance(storage, (str, Path)):
            storage = SQLiteStorage(storage)
        elif not isinstance(storage, Storage):
            raise TypeError("memory storage must be a path or a botkit Storage object")
        self.storage = storage
        self.session = session
        self.max_messages = max_messages

    def for_session(self, session: str) -> "Memory":
        """Same storage, another conversation/user."""
        return Memory(self.storage, session=session, max_messages=self.max_messages)

    # ---- conversation history ------------------------------------------------
    def load(self) -> list[dict]:
        msgs = self.storage.get_messages(self.session, self.max_messages * 3)
        return trim_history(msgs, self.max_messages)

    def extend(self, messages: list[dict]) -> None:
        if messages:
            self.storage.add_messages(self.session, messages)

    def clear_history(self) -> None:
        self.storage.clear_messages(self.session)

    # ---- long-term facts -----------------------------------------------------
    def add_fact(self, text: str) -> int:
        text = text.strip()
        for f in self.storage.list_facts(self.session, 1000):
            if f.text.lower() == text.lower():
                return f.id
        return self.storage.add_fact(self.session, text)

    def list_facts(self, limit: int = 30) -> list[Fact]:
        return self.storage.list_facts(self.session, limit)

    def search_facts(self, query: str, limit: int = 5) -> list[Fact]:
        words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for f in self.list_facts(limit=1000):
            score = len(words & set(re.findall(r"\w+", f.text.lower())))
            if score:
                scored.append((score, f))
        scored.sort(key=lambda x: (-x[0], -x[1].id))
        return [f for _, f in scored[:limit]]

    def delete_fact(self, fact_id: int) -> bool:
        return self.storage.delete_fact(self.session, fact_id)

    def close(self) -> None:
        self.storage.close()

    # ---- tools exposed to the model -----------------------------------------
    def tools(self) -> list[Tool]:
        memory = self

        @tool
        def remember(fact: str) -> str:
            """Save a durable fact or preference about the user for future conversations.

            Args:
                fact: A short, self-contained statement, e.g. "User's name is Ahmed".
            """
            return f"Saved (id {memory.add_fact(fact)})."

        @tool
        def recall(query: str) -> str:
            """Search saved facts about the user.

            Args:
                query: Keywords to look for.
            """
            found = memory.search_facts(query)
            return "\n".join(f"[{f.id}] {f.text}" for f in found) or "Nothing saved matches that."

        @tool
        def forget(fact_id: int) -> str:
            """Delete a saved fact by its id (see `recall`).

            Args:
                fact_id: The id of the fact to delete.
            """
            return "Deleted." if memory.delete_fact(fact_id) else "No fact with that id."

        return [remember, recall, forget]
