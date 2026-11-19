"""SQLite storage: the zero-setup default. One file, many sessions, plus room for your own tables."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .base import Fact, Storage

_PREFIX = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,40}")


class SQLiteStorage(Storage):
    """
        SQLiteStorage("bot.db")                       # file, created if missing
        SQLiteStorage(":memory:")                     # RAM only

    Add your own tables to the same database:

        db = SQLiteStorage("bot.db", extra_schema="CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, item TEXT)")
        db.execute("INSERT INTO orders(item) VALUES(?)", ("tea",))
        db.query("SELECT * FROM orders")              # -> [{"id": 1, "item": "tea"}]

    botkit's own tables are named `<prefix>messages` and `<prefix>facts` (default prefix "botkit_").
    """

    def __init__(self, path: str | Path = ":memory:", *, prefix: str = "botkit_", extra_schema: str | None = None) -> None:
        if not _PREFIX.fullmatch(prefix):
            raise ValueError("prefix may only contain letters, digits and underscores")
        if str(path) != ":memory:":
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            path = p
        self.path = path
        self._m, self._f = f"{prefix}messages", f"{prefix}facts"
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {self._m}(
                id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, data TEXT NOT NULL, ts REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_{self._m}_session ON {self._m}(session, id);
            CREATE TABLE IF NOT EXISTS {self._f}(
                id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, text TEXT NOT NULL, ts REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_{self._f}_session ON {self._f}(session, id);
            """
        )
        if extra_schema:
            self.db.executescript(extra_schema)
        self.db.commit()

    # ---- your own tables -----------------------------------------------------
    def execute(self, sql: str, params: tuple | list = ()) -> int:
        """Run INSERT/UPDATE/DELETE/DDL with ? parameters. Returns the number of affected rows."""
        with self._lock:
            cur = self.db.execute(sql, params)
            self.db.commit()
            return cur.rowcount

    def query(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        """Run a SELECT with ? parameters. Returns a list of dicts."""
        with self._lock:
            return [dict(r) for r in self.db.execute(sql, params).fetchall()]

    # ---- Storage contract ------------------------------------------------------
    def add_messages(self, session: str, messages: list[dict]) -> None:
        now = time.time()
        with self._lock:
            self.db.executemany(
                f"INSERT INTO {self._m}(session, data, ts) VALUES(?,?,?)",
                [(session, json.dumps(m, ensure_ascii=False, default=str), now) for m in messages],
            )
            self.db.commit()

    def get_messages(self, session: str, limit: int) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                f"SELECT data FROM {self._m} WHERE session=? ORDER BY id DESC LIMIT ?", (session, limit)
            ).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    def clear_messages(self, session: str) -> None:
        with self._lock:
            self.db.execute(f"DELETE FROM {self._m} WHERE session=?", (session,))
            self.db.commit()

    def add_fact(self, session: str, text: str) -> int:
        with self._lock:
            cur = self.db.execute(
                f"INSERT INTO {self._f}(session, text, ts) VALUES(?,?,?)", (session, text, time.time())
            )
            self.db.commit()
            return int(cur.lastrowid)

    def list_facts(self, session: str, limit: int = 30) -> list[Fact]:
        with self._lock:
            rows = self.db.execute(
                f"SELECT id, text FROM {self._f} WHERE session=? ORDER BY id DESC LIMIT ?", (session, limit)
            ).fetchall()
        return [Fact(r[0], r[1]) for r in reversed(rows)]

    def delete_fact(self, session: str, fact_id: int) -> bool:
        with self._lock:
            cur = self.db.execute(f"DELETE FROM {self._f} WHERE id=? AND session=?", (fact_id, session))
            self.db.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self.db.close()
