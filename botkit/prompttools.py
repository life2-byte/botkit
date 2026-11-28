"""Prompt tools: tool calling for models or proxies that ignore the API `tools` field.

Some OpenAI-compatible endpoints (for example a proxy in front of a chat website's web token) accept `tools` but never
return `tool_calls`; the model then answers in its own sandbox. With `tool_mode="prompt"` the tools are described in
the system prompt as text, the model answers with a fenced ```tool block, and botkit turns that block back into a normal
tool call. Tools still run in YOUR workspace. This is opt-in: the default `tool_mode="native"` never touches this file.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

TOOL_MODES = ("native", "prompt")

_BLOCK = re.compile(r"```tool[ \t]*\r?\n(.*?)```", re.S | re.I)
_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S | re.I)

INSTRUCTIONS = """\
You can use tools. To call a tool, write a block exactly like this and nothing else inside it:

```tool
{"name": "tool_name", "arguments": {"arg": "value"}}
```

Rules for tools:
- Use ONLY the tools listed below. They run in the user's own workspace. You have no file system, sandbox or tools of your own here: never create, save or edit a file yourself, and never say a file exists unless a tool result confirms it.
- When the user asks for something a tool can do (create, read, edit, search, calculate...), your ENTIRE reply must be the tool block(s), with no other text.
- The block must be valid JSON. Use the exact tool name and the exact argument names from the list. Several blocks in one reply call several tools.
- After your tool block(s), stop. The results come back in the next message as "[tool result: name]".
- When no tool is needed, or when the task is finished, answer in plain text without any ```tool block.

Example of a correct reply:
{example}

Available tools:
{tools}
"""

TOOL_PROMPTS = ("system", "user")


def _example(tools: list[dict]) -> str:
    specs = [t.get("function", t) for t in tools]
    f = next((x for x in specs if x.get("name") == "write_file"), specs[0] if specs else {"name": "tool", "parameters": {}})
    params = f.get("parameters") or {}
    props = params.get("properties", {}) or {}
    args = {k: ("hello.txt" if k == "path" else "hi" if k == "content" else f"<{k}>") for k in (params.get("required") or list(props)[:2])}
    return "```tool\n" + json.dumps({"name": f.get("name"), "arguments": args}) + "\n```"


def render_tools(tools: list[dict], template: str | None = None) -> str:
    """The tool instructions. `template` may use {tools} (the list) and {example}; default is INSTRUCTIONS."""
    lines = []
    for t in tools:
        f = t.get("function", t)
        params = f.get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        required = set(params.get("required", []) or [])
        args = []
        for name, p in props.items():
            kind = p.get("type", "any") if isinstance(p, dict) else "any"
            desc = (p.get("description", "") if isinstance(p, dict) else "").strip().replace("\n", " ")
            args.append(f"    - {name} ({kind}{', required' if name in required else ''}){': ' + desc if desc else ''}")
        head = f"* {f.get('name')}: {(f.get('description') or '').strip().splitlines()[0] if f.get('description') else ''}"
        lines.append(head + ("\n" + "\n".join(args) if args else ""))
    return (template or INSTRUCTIONS).replace("{example}", _example(tools) if tools else "").replace("{tools}", "\n".join(lines))


def _call_text(calls: list[dict]) -> str:
    out = []
    for c in calls:
        fn = c.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        out.append("```tool\n" + json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False) + "\n```")
    return "\n".join(out)


def to_plain(messages: list[dict], tools: list[dict] | None, place: str = "system", template: str | None = None) -> list[dict]:
    """Messages with tool calls/results rewritten as plain text, and the tool instructions added to the system prompt."""
    names: dict[str, str] = {}
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                names[c.get("id", "")] = c.get("function", {}).get("name", "tool")
            text = ((m.get("content") or "").strip() + "\n" + _call_text(m["tool_calls"])).strip()
            out.append({"role": "assistant", "content": text})
        elif role == "tool":
            name = names.get(m.get("tool_call_id", ""), "tool")
            out.append({"role": "user", "content": f"[tool result: {name}]\n{m.get('content', '')}"})
        else:
            out.append({k: v for k, v in m.items() if k != "tool_calls"})
    merged: list[dict] = []  # some endpoints dislike two user messages in a row
    for m in out:
        if merged and m["role"] == "user" and merged[-1]["role"] == "user" and isinstance(m.get("content"), str) \
                and isinstance(merged[-1].get("content"), str):
            merged[-1] = {"role": "user", "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(m)
    if tools:
        block = render_tools(tools, template)
        last = next((i for i in range(len(merged) - 1, -1, -1) if merged[i]["role"] == "user"), None)
        if place == "user" and last is not None:  # many chat websites follow the user's own message more than a system prompt
            merged[last] = {"role": "user", "content": f"{block}\n\n---\n{merged[last]['content']}"}
        elif merged and merged[0]["role"] == "system":
            merged[0] = {"role": "system", "content": f"{merged[0]['content']}\n\n{block}"}
        else:
            merged.insert(0, {"role": "system", "content": block})
    return merged


def _norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def _json_or_none(text: str) -> Any:
    text = text.strip()
    fence = re.match(r"^```(?:json)?[ \t]*\r?\n(.*?)```$", text, re.S | re.I)  # a plain ```json fence around the whole reply
    if fence:
        text = fence.group(1).strip()
    if not text.startswith(("{", "[")):
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def looks_like_call(text: str) -> bool:
    return bool(re.search(r"```tool|<tool_call>", text or "", re.I)) or _json_or_none(text or "") is not None


def _shape(data: Any, by_norm: dict[str, str]) -> tuple[str | None, Any]:
    """(tool name as written, arguments) from the shapes models use: {"name","arguments"}, {"tool","args"}, {"write_file": {...}}."""
    if not isinstance(data, dict):
        return None, {}
    if isinstance(data.get("function"), dict):
        data = data["function"]
    for key in ("name", "tool", "tool_name", "function"):
        if isinstance(data.get(key), str):
            args = next((data[k] for k in ("arguments", "args", "parameters", "input") if k in data), {})
            return data[key], args
    if len(data) == 1:  # {"Create-file": {"arguments": {...}}}
        (key, inner), = data.items()
        if isinstance(inner, dict):
            args = next((inner[k] for k in ("arguments", "args", "parameters", "input") if k in inner), inner)
            return key, args
    return None, {}


def parse_reply(text: str, tools: list[dict] | None) -> tuple[str, list[dict], list[str]]:
    """(text without tool blocks, tool_calls in OpenAI shape, problems).

    Reads ```tool blocks, <tool_call> tags, and a reply that is nothing but JSON. Tool names are matched loosely
    (case, - and _ ignored). A call with an unknown tool or missing required arguments is a problem, not a call."""
    specs = {t.get("function", t).get("name"): t.get("function", t) for t in tools or []}
    by_norm = {_norm(n): n for n in specs}
    bodies = [*_BLOCK.findall(text or ""), *_TAG.findall(text or "")]
    whole = None
    if not bodies:
        whole = _json_or_none(text or "")
        if whole is not None:
            bodies = [json.dumps(x) for x in (whole if isinstance(whole, list) else [whole])]
    calls: list[dict] = []
    problems: list[str] = []
    valid = ", ".join(f"{n}({', '.join((f.get('parameters') or {}).get('required', []))})" for n, f in specs.items())
    for body in bodies:
        try:
            data = json.loads(body.strip())
        except ValueError:
            problems.append(f"not valid JSON: {body.strip()[:80]}")
            continue
        raw, args = _shape(data, by_norm)
        name = by_norm.get(_norm(raw))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                problems.append(f"arguments of {raw} are not valid JSON")
                continue
        if name is None:
            problems.append(f"unknown tool {raw!r}; valid tools with their required arguments: {valid}")
            continue
        args = args if isinstance(args, dict) else {}
        missing = [r for r in (specs[name].get("parameters") or {}).get("required", []) if r not in args]
        if missing:
            problems.append(f"{name} needs the arguments {', '.join(missing)} (you sent {', '.join(args) or 'none'})")
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:10]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    clean = "" if whole is not None else _TAG.sub("", _BLOCK.sub("", text or "")).strip()
    return clean, calls, problems


RETRY_NOTE = ("[system] Your last reply contained a tool block that could not be used ({why}). Reply again with a "
              "correct block: ```tool\\n{{\"name\": \"...\", \"arguments\": {{...}}}}\\n``` using only the listed tools, "
              "or answer in plain text.")
