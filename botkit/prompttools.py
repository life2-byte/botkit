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
- Use ONLY the tools listed below. They run in the user's own workspace, not in any sandbox of yours. Do not say a file was created unless a tool result confirms it.
- The block must be valid JSON. You may write several blocks in one reply to call several tools.
- After your tool block(s), stop and wait. The results come back in the next message as "[tool result: name]".
- When no tool is needed, or when the task is finished, answer in plain text without any ```tool block.

Available tools:
"""


def render_tools(tools: list[dict]) -> str:
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
    return INSTRUCTIONS + "\n".join(lines)


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


def to_plain(messages: list[dict], tools: list[dict] | None) -> list[dict]:
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
        block = render_tools(tools)
        if merged and merged[0]["role"] == "system":
            merged[0] = {"role": "system", "content": f"{merged[0]['content']}\n\n{block}"}
        else:
            merged.insert(0, {"role": "system", "content": block})
    return merged


def looks_like_call(text: str) -> bool:
    return bool(re.search(r"```tool|<tool_call>", text or "", re.I))


def parse_reply(text: str, tools: list[dict] | None) -> tuple[str, list[dict], list[str]]:
    """(text without tool blocks, tool_calls in OpenAI shape, problems). Unknown tool names are reported as problems."""
    known = {t.get("function", t).get("name") for t in tools or []}
    calls: list[dict] = []
    problems: list[str] = []
    for body in [*_BLOCK.findall(text or ""), *_TAG.findall(text or "")]:
        body = body.strip()
        try:
            data = json.loads(body)
        except ValueError:
            problems.append(f"not valid JSON: {body[:80]}")
            continue
        if isinstance(data, dict) and "function" in data and isinstance(data["function"], dict):
            data = data["function"]
        name = data.get("name") if isinstance(data, dict) else None
        args = (data.get("arguments", data.get("parameters", {})) if isinstance(data, dict) else {}) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                problems.append(f"arguments of {name} are not valid JSON")
                continue
        if name not in known:
            problems.append(f"unknown tool: {name}")
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:10]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    clean = _TAG.sub("", _BLOCK.sub("", text or "")).strip()
    return clean, calls, problems


RETRY_NOTE = ("[system] Your last reply contained a tool block that could not be used ({why}). Reply again with a "
              "correct block: ```tool\\n{{\"name\": \"...\", \"arguments\": {{...}}}}\\n``` using only the listed tools, "
              "or answer in plain text.")
