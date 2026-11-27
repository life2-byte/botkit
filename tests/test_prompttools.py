"""tool_mode="prompt": tools in the prompt, calls read from text. Default mode must be untouched."""
import json
from types import SimpleNamespace as NS

import pytest

from botkit import Agent
from botkit.errors import LLMError
from botkit.llm import FallbackLLM, LLM
from botkit.prompttools import parse_reply, render_tools, to_plain

TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Create a file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]


def fake_llm(texts, **kw):
    llm = LLM("openai:x", api_key="k", **kw)
    llm.sent = []
    queue = list(texts)

    def create(**kwargs):
        llm.sent.append(kwargs)
        return NS(choices=[NS(message=NS(content=queue.pop(0), tool_calls=None))], usage=None)

    llm._client = NS(chat=NS(completions=NS(create=create)))
    return llm


def block(name, **args):
    return "```tool\n" + json.dumps({"name": name, "arguments": args}) + "\n```"


def test_default_is_native_and_sends_tools_field():
    llm = fake_llm(["hi"])
    assert llm.tool_mode == "native"
    llm.chat([{"role": "user", "content": "x"}], TOOLS)
    assert llm.sent[0]["tools"] == TOOLS


def test_bad_mode_rejected():
    with pytest.raises(LLMError):
        LLM("openai:x", api_key="k", tool_mode="nope")


def test_prompt_mode_does_not_send_tools_field_but_describes_them():
    llm = fake_llm(["plain answer"], tool_mode="prompt")
    out = llm.chat([{"role": "system", "content": "SYS"}, {"role": "user", "content": "x"}], TOOLS)
    sent = llm.sent[0]
    assert "tools" not in sent and "tool_choice" not in sent
    assert sent["messages"][0]["content"].startswith("SYS") and "write_file" in sent["messages"][0]["content"]
    assert out == {"role": "assistant", "content": "plain answer"}


def test_parse_block_into_tool_call():
    text = "Creating it.\n" + block("write_file", path="a.txt", content="hi")
    content, calls, problems = parse_reply(text, TOOLS)
    assert content == "Creating it." and not problems
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt", "content": "hi"}


def test_parse_tag_style_and_several_blocks():
    text = block("write_file", path="a", content="1") + "\n" + block("write_file", path="b", content="2") \
        + '\n<tool_call>{"name": "write_file", "arguments": {"path": "c", "content": "3"}}</tool_call>'
    _, calls, problems = parse_reply(text, TOOLS)
    assert [json.loads(c["function"]["arguments"])["path"] for c in calls] == ["a", "b", "c"] and not problems


def test_unknown_tool_and_bad_json_are_problems_not_calls():
    _, calls, problems = parse_reply(block("rm_rf", path="/") + "\n```tool\n{oops\n```", TOOLS)
    assert calls == [] and len(problems) == 2


def test_to_plain_rewrites_tool_messages():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {
                "name": "write_file", "arguments": json.dumps({"path": "a", "content": "b"})}}]},
            {"role": "tool", "tool_call_id": "1", "content": "Wrote a"}]
    plain = to_plain(msgs, TOOLS)
    assert all("tool_calls" not in m and m["role"] != "tool" for m in plain)
    assert "```tool" in plain[2]["content"] and plain[3] == {"role": "user", "content": "[tool result: write_file]\nWrote a"}
    assert "Available tools" in plain[0]["content"]


def test_bad_block_gets_one_correction_round():
    llm = fake_llm(["```tool\n{oops\n```", block("write_file", path="a", content="b")], tool_mode="prompt")
    out = llm.chat([{"role": "user", "content": "x"}], TOOLS)
    assert out["tool_calls"][0]["function"]["name"] == "write_file" and len(llm.sent) == 2
    assert "could not be used" in llm.sent[1]["messages"][-1]["content"]


def test_never_loops_forever_on_bad_blocks():
    llm = fake_llm(["```tool\n{a\n```", "```tool\n{b\n```"], tool_mode="prompt")
    out = llm.chat([{"role": "user", "content": "x"}], TOOLS)
    assert "tool_calls" not in out and len(llm.sent) == 2


def test_stream_in_prompt_mode_hides_tool_block():
    llm = fake_llm(["Making it.\n" + block("write_file", path="a", content="b")], tool_mode="prompt")
    seen = []
    out = llm.stream([{"role": "user", "content": "x"}], TOOLS, seen.append)
    assert "".join(seen) == "Making it." and out["tool_calls"]


def test_agent_runs_tool_in_own_workspace_end_to_end(tmp_path):
    agent = Agent("openai:x", api_key="k", tools=["write_file"], workdir=tmp_path, tool_mode="prompt")
    llm = fake_llm(["On it.\n" + block("write_file", path="hello.txt", content="hi"), "Created hello.txt."], tool_mode="prompt")
    agent.llm = llm
    assert agent.chat("make hello.txt") == "Created hello.txt."
    assert (tmp_path / "hello.txt").read_text() == "hi"
    assert all("tools" not in s for s in llm.sent)
    assert "[tool result: write_file]" in llm.sent[1]["messages"][-1]["content"]


def test_agent_flag_reaches_llm_and_fallback_dict():
    a = Agent("openai:x", api_key="k", tool_mode="prompt")
    assert a.llm.tool_mode == "prompt"
    assert Agent("openai:x", api_key="k").llm.tool_mode == "native"
    f = FallbackLLM({"model": "openai:x", "api_key": "k"}, [{"model": "openai:y", "api_key": "k", "tool_mode": "prompt"}])
    assert f._get(0).tool_mode == "native" and f._get(1).tool_mode == "prompt"


def test_render_tools_lists_args():
    text = render_tools(TOOLS)
    assert "write_file" in text and "path (string, required)" in text
