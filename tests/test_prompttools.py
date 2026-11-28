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


def test_bare_json_without_fence_and_loose_names():
    text = json.dumps({"Write-File": {"arguments": {"path": "a.txt", "content": "1"}}})
    clean, calls, problems = parse_reply(text, TOOLS)
    assert clean == "" and not problems and calls[0]["function"]["name"] == "write_file"
    for shape in ({"tool": "write_file", "args": {"path": "a", "content": "b"}},
                  {"name": "WRITE_FILE", "parameters": {"path": "a", "content": "b"}}):
        assert parse_reply("```json\n" + json.dumps(shape) + "\n```", TOOLS)[1][0]["function"]["name"] == "write_file"


def test_wrong_or_missing_arguments_are_problems():
    wrong = json.dumps({"Create-file": {"arguments": {"context": "<html>"}}})  # a real DeepSeek reply
    _, calls, problems = parse_reply(wrong, TOOLS)
    assert calls == [] and "valid tools" in problems[0]
    _, calls, problems = parse_reply(json.dumps({"name": "write_file", "arguments": {"context": "x"}}), TOOLS)
    assert calls == [] and "needs the arguments path, content" in problems[0]


def test_wrong_json_reply_gets_a_correction_round_then_works():
    wrong = json.dumps({"Create-file": {"arguments": {"context": "<html>"}}})
    llm = fake_llm([wrong, block("write_file", path="a.html", content="<html>")], tool_mode="prompt")
    out = llm.chat([{"role": "user", "content": "x"}], TOOLS)
    assert out["tool_calls"][0]["function"]["name"] == "write_file" and len(llm.sent) == 2
    assert "write_file(path, content)" in llm.sent[1]["messages"][-1]["content"]


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


def test_tool_prompt_user_puts_instructions_in_latest_user_message_only():
    llm = fake_llm(["ok"], tool_mode="prompt", tool_prompt="user")
    llm.chat([{"role": "system", "content": "SYS"}, {"role": "user", "content": "first"},
              {"role": "assistant", "content": "a"}, {"role": "user", "content": "make a file"}], TOOLS)
    m = llm.sent[0]["messages"]
    assert m[0] == {"role": "system", "content": "SYS"} and "first" == m[1]["content"]
    assert "write_file" in m[-1]["content"] and m[-1]["content"].endswith("make a file")


def test_custom_template_and_example_in_default_text():
    plain = to_plain([{"role": "user", "content": "x"}], TOOLS, "system", "MY RULES\n{tools}\nEG {example}")
    assert plain[0]["content"].startswith("MY RULES") and "* write_file" in plain[0]["content"] and "EG ```tool" in plain[0]["content"]
    default = render_tools(TOOLS)
    assert '"name": "write_file", "arguments": {"path": "hello.txt", "content": "hi"}' in default
    assert "{tools}" not in default and "{example}" not in default


def test_bad_tool_prompt_rejected():
    with pytest.raises(LLMError):
        LLM("openai:x", api_key="k", tool_mode="prompt", tool_prompt="nope")


def test_bare_prompt_is_only_system_rules_security(tmp_path):
    a = Agent("openai:x", api_key="k", tools=True, skills=True, system="", rules=["R1"], bare_prompt=True, workdir=tmp_path)
    text = a._system_prompt()
    assert text.startswith("Rules from the app owner") and "R1" in text and "Security rules" in text
    assert "Current date" not in text and "Your tools" not in text and "skills" not in text
    b = Agent("openai:x", api_key="k", system="", workdir=tmp_path)  # empty system without bare: no blank start
    assert b._system_prompt().startswith("Current date")


def test_agent_passes_tool_prompt_options():
    a = Agent("openai:x", api_key="k", tool_mode="prompt", tool_prompt="user", tool_prompt_text="T {tools}")
    assert a.llm.tool_prompt == "user" and a.llm.tool_prompt_text == "T {tools}"
