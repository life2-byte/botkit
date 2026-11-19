"""Supabase storage over its REST API (PostgREST). No extra packages needed.

1. In Supabase: SQL Editor -> run the output of `python -m botkit.memory.supabase`.
2. Use the project URL and the **service_role** key (server side only, never in a browser/app).

    storage = SupabaseStorage()                       # reads SUPABASE_URL and SUPABASE_KEY
    storage = SupabaseStorage("https://xyz.supabase.co", "service-role-key")
    bot = Agent(memory=storage)

Your own tables work the same way:   storage.table("orders").insert({"item": "tea"})
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..errors import BotError
from .base import Fact, Storage

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")
_ORDER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.(asc|desc))?")


def schema_sql(prefix: str = "botkit_") -> str:
    """SQL that creates botkit's two tables. Run it once in the Supabase SQL editor."""
    if not _IDENT.fullmatch(prefix):
        raise ValueError("prefix may only contain letters, digits and underscores")
    m, f = f"{prefix}messages", f"{prefix}facts"
    return f"""create table if not exists {m} (
  id bigint generated always as identity primary key,
  session text not null,
  data jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists {m}_session_idx on {m} (session, id);
create table if not exists {f} (
  id bigint generated always as identity primary key,
  session text not null,
  text text not null,
  created_at timestamptz not null default now()
);
create index if not exists {f}_session_idx on {f} (session, id);
-- Row level security on, no policies: only the service_role key can read/write these tables.
alter table {m} enable row level security;
alter table {f} enable row level security;
"""


def _filter_params(filters: dict[str, Any] | None) -> list[tuple[str, str]]:
    """{"user": "ahmed", "age": ("gt", 18)} -> [("user", "eq.ahmed"), ("age", "gt.18")]"""
    out: list[tuple[str, str]] = []
    for col, cond in (filters or {}).items():
        if not _IDENT.fullmatch(str(col)):
            raise ValueError(f"invalid column name {col!r}")
        if isinstance(cond, tuple):
            op, value = cond
            if op not in ("eq", "neq", "gt", "gte", "lt", "lte", "like", "ilike"):
                raise ValueError(f"unsupported operator {op!r}")
        else:
            op, value = "eq", cond
        out.append((col, "is.null" if value is None else f"{op}.{value}"))
    return out


class SupabaseTable:
    """Tiny helper for your own tables: select / insert / update / delete."""

    def __init__(self, storage: "SupabaseStorage", name: str) -> None:
        if not _IDENT.fullmatch(name):
            raise ValueError(f"invalid table name {name!r}")
        self._s, self.name = storage, name

    def select(self, filters: dict | None = None, columns: str = "*", order: str | None = None,
               limit: int | None = None) -> list[dict]:
        params = [("select", columns), *_filter_params(filters)]
        if order:
            if not _ORDER.fullmatch(order):
                raise ValueError("order must look like 'id' or 'id.desc'")
            params.append(("order", order))
        if limit is not None:
            params.append(("limit", str(int(limit))))
        return self._s._request("GET", self.name, params) or []

    def insert(self, rows: dict | list[dict], returning: bool = True) -> list[dict]:
        return self._s._request("POST", self.name, None, rows, returning) or []

    def update(self, values: dict, filters: dict) -> list[dict]:
        if not filters:
            raise ValueError("update() needs filters (refusing to change every row)")
        return self._s._request("PATCH", self.name, _filter_params(filters), values, True) or []

    def delete(self, filters: dict) -> list[dict]:
        if not filters:
            raise ValueError("delete() needs filters (refusing to delete every row)")
        return self._s._request("DELETE", self.name, _filter_params(filters), None, True) or []


class SupabaseStorage(Storage):
    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        prefix: str = "botkit_",
        timeout: float = 15.0,
    ) -> None:
        url = url or os.environ.get("SUPABASE_URL")
        key = key or os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise BotError("Supabase needs a URL and a key: pass them or set SUPABASE_URL and SUPABASE_KEY.")
        if not url.startswith(("http://", "https://")):
            raise BotError("Supabase URL must start with https://")
        url = url.rstrip("/")
        self.base = url if url.endswith("/rest/v1") else url + "/rest/v1"
        self.key, self.timeout = key, timeout
        if not _IDENT.fullmatch(prefix):
            raise ValueError("prefix may only contain letters, digits and underscores")
        self._m, self._f = f"{prefix}messages", f"{prefix}facts"
        self.prefix = prefix

    @staticmethod
    def schema_sql(prefix: str = "botkit_") -> str:
        return schema_sql(prefix)

    def table(self, name: str) -> SupabaseTable:
        return SupabaseTable(self, name)

    # ---- HTTP ------------------------------------------------------------------
    def _request(self, method: str, table: str, params: list[tuple[str, str]] | None = None,
                 body: Any = None, returning: bool = False) -> Any:
        query = urllib.parse.urlencode(params or [], quote_via=urllib.parse.quote)
        url = f"{self.base}/{table}" + (f"?{query}" if query else "")
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if returning:
            headers["Prefer"] = "return=representation"
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")[:400]
            if e.code == 404 and ("PGRST205" in text or "schema cache" in text):
                raise BotError(
                    f"Supabase table '{table}' not found. Create botkit's tables first: run "
                    "SupabaseStorage.schema_sql() in the Supabase SQL editor."
                ) from None
            raise BotError(f"Supabase error {e.code}: {text}") from None
        except urllib.error.URLError as e:
            raise BotError(f"Could not reach Supabase: {e.reason}") from None
        return json.loads(raw) if raw.strip() else None

    # ---- Storage contract --------------------------------------------------------
    def add_messages(self, session: str, messages: list[dict]) -> None:
        if messages:
            self.table(self._m).insert([{"session": session, "data": m} for m in messages], returning=False)

    def get_messages(self, session: str, limit: int) -> list[dict]:
        rows = self.table(self._m).select({"session": session}, "data", order="id.desc", limit=limit)
        return [r["data"] for r in reversed(rows)]

    def clear_messages(self, session: str) -> None:
        self.table(self._m).delete({"session": session})

    def add_fact(self, session: str, text: str) -> int:
        rows = self.table(self._f).insert({"session": session, "text": text})
        return int(rows[0]["id"])

    def list_facts(self, session: str, limit: int = 30) -> list[Fact]:
        rows = self.table(self._f).select({"session": session}, "id,text", order="id.desc", limit=limit)
        return [Fact(int(r["id"]), r["text"]) for r in reversed(rows)]

    def delete_fact(self, session: str, fact_id: int) -> bool:
        return bool(self.table(self._f).delete({"id": int(fact_id), "session": session}))


if __name__ == "__main__":  # python -m botkit.memory.supabase
    print(schema_sql())
