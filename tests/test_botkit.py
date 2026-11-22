import json
import sys
import time
from pathlib import Path

try:  # newer openai releases use httpx2
    import httpx2 as httpx
except ImportError:
    import httpx
import openai
import pytest

from botkit import MCP, Agent, LLMError, Memory, Skill, tool
from botkit.llm import LLM, parse_model
from botkit.memory import trim_history

HERE = Path(__file__).parent


# ---------------------------------------------------------------- helpers
def call(name, args, id="c1"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": id, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}]}


def say(text):
    return {"role": "assistant", "content": text}


class FakeLLM:
    """Plays back scripted replies and records what it was sent."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None):
        self.seen.append((json.loads(json.dumps(messages)), tools))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first number
        b: second number
    """
    return a + b


@tool
def explode() -> str:
    """Always raises."""
    raise ValueError("bad thing")


# ---------------------------------------------------------------- tools
def test_tool_schema_from_hints_and_docstring():
    spec = add.spec()["function"]
    assert spec["name"] == "add" and spec["description"] == "Add two numbers."
    assert spec["parameters"]["properties"]["a"] == {"type": "integer", "description": "first number"}
    assert spec["parameters"]["required"] == ["a", "b"]


def test_tool_schema_complex_types():
    from typing import Literal, Optional

    @tool
    def f(tags: list[str], mode: Literal["a", "b"], n: Optional[int] = None, flag: bool = False): ...

    p = f.spec()["function"]["parameters"]
    assert p["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert p["properties"]["mode"] == {"enum": ["a", "b"]}
    assert p["properties"]["n"] == {"type": "integer"}
    assert p["required"] == ["tags", "mode"]


def test_async_tool_and_arg_coercion():
    @tool
    async def double(x: int) -> int:
        """Double."""
        return x * 2

    assert double.run({"x": "21"}) == 42


# ---------------------------------------------------------------- agent loop
def test_plain_chat():
    llm = FakeLLM(say("hi there"))
    bot = Agent(llm=llm)
    assert bot.chat("hello") == "hi there"
    sent, tools = llm.seen[0]
    assert sent[0]["role"] == "system" and sent[-1] == {"role": "user", "content": "hello"} and tools is None


def test_tool_loop():
    llm = FakeLLM(call("add", {"a": 2, "b": 3}), say("The answer is 5"))
    bot = Agent(llm=llm, tools=[add])
    assert bot.chat("2+3?") == "The answer is 5"
    assert llm.seen[1][0][-1] == {"role": "tool", "tool_call_id": "c1", "content": "5"}


def test_plain_function_works_as_tool():
    def shout(text: str) -> str:
        """Uppercase text."""
        return text.upper()

    llm = FakeLLM(call("shout", {"text": "hey"}), say("done"))
    bot = Agent(llm=llm, tools=[shout])
    bot.chat("x")
    assert llm.seen[1][0][-1]["content"] == "HEY"


def test_tool_exception_is_fed_back_not_raised():
    llm = FakeLLM(call("explode", {}), say("sorry, that failed"))
    bot = Agent(llm=llm, tools=[explode])
    assert bot.chat("go") == "sorry, that failed"
    assert llm.seen[1][0][-1]["content"] == "Error: ValueError: bad thing"


def test_unknown_tool_and_bad_json_args():
    llm = FakeLLM(call("nope", {}), call("add", "{not json", id="c2"), say("ok"))
    bot = Agent(llm=llm, tools=[add])
    bot.chat("x")
    assert "unknown tool 'nope'" in llm.seen[1][0][-1]["content"]
    assert "invalid arguments for 'add'" in llm.seen[2][0][-1]["content"]


def test_missing_argument_reported_to_model():
    llm = FakeLLM(call("add", {"a": 1}), say("ok"))
    Agent(llm=llm, tools=[add]).chat("x")
    assert llm.seen[1][0][-1]["content"].startswith("Error: TypeError")


def test_tool_timeout():
    @tool
    def slow() -> str:
        """Slow."""
        time.sleep(2)
        return "late"

    llm = FakeLLM(call("slow", {}), say("ok"))
    Agent(llm=llm, tools=[slow], tool_timeout=0.2).chat("x")
    assert "timed out" in llm.seen[1][0][-1]["content"]


def test_repeated_identical_call_is_blocked():
    llm = FakeLLM(*[call("add", {"a": 1, "b": 1}, id=f"c{i}") for i in range(4)], say("fine"))
    Agent(llm=llm, tools=[add]).chat("x")
    assert "already called" in llm.seen[3][0][-1]["content"]


def test_long_tool_output_truncated():
    @tool
    def big() -> str:
        """Big."""
        return "x" * 500

    llm = FakeLLM(call("big", {}), say("ok"))
    Agent(llm=llm, tools=[big], max_tool_output=100).chat("x")
    assert "truncated 400 characters" in llm.seen[1][0][-1]["content"]


def test_step_limit_gives_final_answer():
    llm = FakeLLM(call("add", {"a": 1, "b": 1}, id="a"), call("add", {"a": 2, "b": 2}, id="b"), say("best effort"))
    bot = Agent(llm=llm, tools=[add], max_steps=2)
    assert bot.chat("x") == "best effort"
    assert llm.seen[2][1] is None  # final call has no tools


def test_llm_failure_rolls_back_so_retry_works():
    llm = FakeLLM(call("add", {"a": 1, "b": 2}), LLMError("down"), say("recovered"))
    bot = Agent(llm=llm, tools=[add])
    with pytest.raises(LLMError):
        bot.chat("first")
    assert bot.messages == []
    assert bot.chat("again") == "recovered"
    assert [m["role"] for m in bot.messages] == ["user", "assistant"]


def test_events_callback():
    events = []
    llm = FakeLLM(call("add", {"a": 1, "b": 2}), say("3"))
    Agent(llm=llm, tools=[add], on_event=lambda k, d: events.append((k, d.get("name")))).chat("x")
    assert [e for e in events if e[0] != "llm_call"] == [("tool_call", "add"), ("tool_result", "add")]
    assert [e[0] for e in events].count("llm_call") == 2


# ---------------------------------------------------------------- memory
def test_history_persists_across_instances(tmp_path):
    db = tmp_path / "m.db"
    Agent(llm=FakeLLM(say("nice to meet you")), memory=db).chat("I am Ahmed")
    llm2 = FakeLLM(say("Ahmed!"))
    bot2 = Agent(llm=llm2, memory=db)
    bot2.chat("who am I?")
    roles = [m["role"] for m in llm2.seen[0][0]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm2.seen[0][0][1]["content"] == "I am Ahmed"


def test_sessions_are_isolated_and_facts_injected(tmp_path):
    db = tmp_path / "m.db"
    a = Agent(llm=FakeLLM(call("remember", {"fact": "Likes chai"}), say("noted")), memory=Memory(db, session="a"))
    a.chat("remember I like chai")
    llm_b = FakeLLM(say("hi"))
    Agent(llm=llm_b, memory=Memory(db, session="b")).chat("hello")
    assert "chai" not in llm_b.seen[0][0][0]["content"] and len(llm_b.seen[0][0]) == 2
    llm_a = FakeLLM(say("hi"))
    Agent(llm=llm_a, memory=Memory(db, session="a")).chat("hello")
    assert "Likes chai" in llm_a.seen[0][0][0]["content"]


def test_recall_forget_and_dedupe():
    mem = Memory()
    first = mem.add_fact("Lives in Lahore")
    assert mem.add_fact("lives in lahore") == first
    remember, recall, forget = mem.tools()
    assert "Lahore" in recall.run({"query": "where lahore"})
    assert recall.run({"query": "zzz"}) == "Nothing saved matches that."
    assert forget.run({"fact_id": first}) == "Deleted."
    assert mem.list_facts() == []


def test_reset_clears_history(tmp_path):
    bot = Agent(llm=FakeLLM(say("a")), memory=tmp_path / "m.db")
    bot.chat("x")
    bot.reset()
    assert bot.messages == [] and Memory(tmp_path / "m.db").load() == []


def test_trim_history_never_starts_mid_toolcall():
    msgs = [{"role": "user"}, call("a", {}), {"role": "tool"}, say("x"), {"role": "user"}, say("y")]
    assert trim_history(msgs, 4)[0]["role"] == "user"
    assert trim_history(msgs, 4) == msgs[4:]
    assert trim_history(msgs, 99) == msgs


def test_current_turn_never_trimmed():
    llm = FakeLLM(*[call("add", {"a": i, "b": 1}, id=f"c{i}") for i in range(5)], say("done"))
    bot = Agent(llm=llm, tools=[add], max_history=2, max_steps=10)
    bot.chat("go")
    assert llm.seen[-1][0][1] == {"role": "user", "content": "go"}


# ---------------------------------------------------------------- skills
def test_skills_from_folder(tmp_path):
    (tmp_path / "pirate").mkdir()
    (tmp_path / "pirate" / "SKILL.md").write_text("---\nname: pirate\ndescription: Talk like a pirate\n---\nSay arr a lot.\n")
    (tmp_path / "tone.md").write_text("# Be formal\nNo slang.")
    llm = FakeLLM(call("load_skill", {"name": "pirate"}), say("Arr!"))
    bot = Agent(llm=llm, skills=tmp_path)
    assert bot.chat("hi") == "Arr!"
    system = llm.seen[0][0][0]["content"]
    assert "pirate: Talk like a pirate" in system and "tone: Be formal" in system
    assert llm.seen[1][0][-1]["content"] == "Say arr a lot."


def test_inline_skill_and_missing_path():
    bot = Agent(llm=FakeLLM(say("x")), skills=[Skill("s", "d", "body")])
    assert "s" in bot.skills.skills
    with pytest.raises(FileNotFoundError):
        Agent(llm=FakeLLM(), skills="/no/such/dir")


# ---------------------------------------------------------------- MCP (real stdio server)
def demo_server(**kw):
    return MCP(sys.executable, str(HERE / "demo_mcp_server.py"), **kw)


def test_mcp_tools_discovered_and_called():
    llm = FakeLLM(call("demo_multiply", {"a": 6, "b": 7}), say("42"))
    with Agent(llm=llm, mcp=demo_server(name="demo")) as bot:
        assert "demo_multiply" in bot.tool_names
        assert bot.chat("6*7") == "42"
        assert llm.seen[1][0][-1]["content"] == "42"


def test_mcp_tool_error_reaches_model():
    llm = FakeLLM(call("demo_boom", {}), say("it failed"))
    with Agent(llm=llm, mcp=demo_server(name="demo")) as bot:
        bot.chat("x")
        result = llm.seen[1][0][-1]["content"]  # the SDK may mask the server-side message
        assert result.startswith("Error") and "boom" in result


def test_mcp_config_dict_and_command_string():
    cfg = {"mcpServers": {"calc": {"command": sys.executable, "args": [str(HERE / "demo_mcp_server.py")]}}}
    with Agent(llm=FakeLLM(), mcp=cfg) as bot:
        assert "calc_multiply" in bot.tool_names
    with Agent(llm=FakeLLM(), mcp=f"{sys.executable} {HERE / 'demo_mcp_server.py'}") as bot:
        assert "mcp1_multiply" in bot.tool_names


def test_broken_mcp_server_is_skipped_with_warning():
    llm = FakeLLM(say("still works"))
    bad = MCP(sys.executable, "-c", "import sys; sys.exit(1)", name="bad", timeout=10)
    with pytest.warns(RuntimeWarning, match="failed to start"):
        with Agent(llm=llm, mcp=bad, tools=[add]) as bot:
            assert bot.chat("hi") == "still works"
            assert "add" in bot.tool_names and "demo_multiply" not in bot.tool_names


# ---------------------------------------------------------------- LLM wrapper
def test_parse_model():
    assert parse_model("groq:llama-3.3-70b-versatile") == ("groq", "llama-3.3-70b-versatile")
    assert parse_model("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    assert parse_model("gpt-4o") == ("openai", "gpt-4o")
    assert parse_model("llama3.1:8b") == ("openai", "llama3.1:8b")


def test_missing_api_key_message(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        LLM("groq:x")


def _status_error(cls, code):
    req = httpx.Request("POST", "http://x")
    return cls("boom", response=httpx.Response(code, request=req), body=None)


class _Completions:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def create(self, **kw):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def _fake_llm(*outcomes):
    llm = LLM("groq:x", api_key="k")
    llm._sleep = lambda s: None
    comp = _Completions(*outcomes)
    llm._client = type("C", (), {"chat": type("Ch", (), {"completions": comp})()})()
    return llm, comp


def _ok(text="hi", tool_calls=None):
    msg = type("M", (), {"content": text, "tool_calls": tool_calls})()
    return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()


def test_llm_retries_rate_limit_then_succeeds():
    llm, comp = _fake_llm(_status_error(openai.RateLimitError, 429), _status_error(openai.InternalServerError, 503), _ok("yay"))
    assert llm.chat([{"role": "user", "content": "x"}])["content"] == "yay" and comp.calls == 3


def test_llm_gives_up_after_retries():
    llm, comp = _fake_llm(*[_status_error(openai.RateLimitError, 429) for _ in range(5)])
    with pytest.raises(LLMError, match="after 5 attempts"):
        llm.chat([])
    assert comp.calls == 5


def test_llm_auth_error_not_retried_and_bad_request_clear():
    llm, comp = _fake_llm(_status_error(openai.AuthenticationError, 401))
    with pytest.raises(LLMError, match="Authentication"):
        llm.chat([])
    assert comp.calls == 1
    llm, comp = _fake_llm(_status_error(openai.BadRequestError, 400))
    with pytest.raises(LLMError, match="Bad request"):
        llm.chat([])


def test_llm_parses_tool_calls():
    tc = type("TC", (), {"id": "t1", "function": type("F", (), {"name": "add", "arguments": '{"a":1}'})()})()
    llm, _ = _fake_llm(_ok("", [tc]))
    out = llm.chat([])
    assert out["tool_calls"] == [{"id": "t1", "type": "function", "function": {"name": "add", "arguments": '{"a":1}'}}]
