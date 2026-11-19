"""Turn any Python function into an LLM tool."""
from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import types as _types
import typing
from dataclasses import dataclass
from typing import Any, Callable

_PRIMS = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _schema_for(tp: Any) -> dict:
    """JSON-schema fragment for a Python type hint."""
    if tp is inspect.Parameter.empty or tp is Any:
        return {"type": "string"}
    if tp in _PRIMS:
        return {"type": _PRIMS[tp]}
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if tp in (list, tuple, set, frozenset) or origin in (list, tuple, set, frozenset):
        schema: dict = {"type": "array"}
        if args and args[0] is not Ellipsis:
            schema["items"] = _schema_for(args[0])
        return schema
    if tp is dict or origin is dict:
        return {"type": "object"}
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin is typing.Union or origin is getattr(_types, "UnionType", object()):
        opts = [a for a in args if a is not type(None)]
        if len(opts) == 1:
            return _schema_for(opts[0])
        return {"anyOf": [_schema_for(a) for a in opts]}
    return {"type": "string"}


def _parse_doc(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into (description, {param: description}) - Google style `Args:`."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    parts = re.split(r"\n\s*(?:Args|Arguments|Parameters):\s*\n", doc, maxsplit=1)
    desc = parts[0].strip()
    params: dict[str, str] = {}
    if len(parts) > 1:
        for m in re.finditer(r"^\s*(\w+)\s*(?:\([^)]*\))?:\s*(.+)$", parts[1], re.M):
            params[m.group(1)] = m.group(2).strip()
    return desc, params


def to_text(value: Any) -> str:
    """Convert any tool return value to the string the model will read."""
    if value is None:
        return "OK"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _run_async(awaitable: Any) -> Any:
    """Run a coroutine to completion from sync code (works even inside a running loop)."""
    box: dict[str, Any] = {}

    async def _wrap() -> Any:
        return await awaitable

    def target() -> None:
        try:
            box["value"] = asyncio.run(_wrap())
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


@dataclass
class Tool:
    """A function the model may call. Build one with the @tool decorator."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    sensitive: bool = False  # sensitive tools go through the Agent's `approve` callback, if one is set

    def spec(self) -> dict:
        params = self.parameters or {"type": "object", "properties": {}}
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def _coerce(self, args: dict) -> dict:
        """Models often send "5" for an int or 5 for a string; fix the easy cases."""
        props = (self.parameters or {}).get("properties", {})
        out = dict(args)
        for key, val in args.items():
            kind = props.get(key, {}).get("type")
            try:
                if isinstance(val, str):
                    if kind == "integer":
                        out[key] = int(val)
                    elif kind == "number":
                        out[key] = float(val)
                    elif kind == "boolean" and val.strip().lower() in ("true", "false"):
                        out[key] = val.strip().lower() == "true"
                    elif kind in ("array", "object"):
                        out[key] = json.loads(val)
                elif kind == "string" and isinstance(val, (int, float)) and not isinstance(val, bool):
                    out[key] = str(val)
                elif kind == "integer" and isinstance(val, float) and val.is_integer():
                    out[key] = int(val)
            except ValueError:
                pass
        return out

    def run(self, args: dict) -> Any:
        out = self.fn(**self._coerce(args))
        if inspect.isawaitable(out):
            out = _run_async(out)
        return out


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    sensitive: bool = False,
):
    """Decorator: `@tool` on a function -> a Tool (schema built from type hints + docstring).

        @tool
        def weather(city: str) -> str:
            '''Get the weather for a city.
            Args:
                city: City name, e.g. "Lahore".
            '''
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        if isinstance(f, Tool):
            return f
        sig = inspect.signature(f)
        try:
            hints = typing.get_type_hints(f)
        except Exception:
            hints = {}
        doc_desc, doc_params = _parse_doc(inspect.getdoc(f))
        props: dict[str, dict] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            schema = dict(_schema_for(hints.get(pname, p.annotation)))
            if pname in doc_params:
                schema["description"] = doc_params[pname]
            props[pname] = schema
            if p.default is p.empty:
                required.append(pname)
        params: dict[str, Any] = {"type": "object", "properties": props}
        if required:
            params["required"] = required
        tool_name = name or getattr(f, "__name__", "tool")
        return Tool(tool_name, description or doc_desc or tool_name, params, f, sensitive)

    return wrap(fn) if fn is not None else wrap


def as_tool(obj: Any) -> Tool:
    """Accept a Tool or a plain callable."""
    if isinstance(obj, Tool):
        return obj
    if callable(obj):
        return tool(obj)
    raise TypeError(f"Cannot use {obj!r} as a tool (expected a function or Tool)")


def clip(text: str, limit: int) -> str:
    """Shorten long text and say how much was cut."""
    return text if len(text) <= limit else text[:limit] + f"\n...[truncated {len(text) - limit} characters]"
