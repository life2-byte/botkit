"""Fallback models, limit detection, and resume (agent loop and graph loop)."""
import json
import pytest

from botkit import Agent, BotError, FallbackLLM, Graph, LLMError, chain, END
from botkit.llm import LLM, _parse_wait
from botkit.testing import FakeLLM, call, fail, say


# ---- limit detection ---------------------------------------------------------------
def test_parse_wait():
    assert _parse_wait("Please try again in 12m38.4s.") == pytest.approx(758.4)
    assert _parse_wait("try again in 1h2m") == 3720
    assert _parse_wait("try again in 350ms") == pytest.approx(0.35)
    assert _parse_wait("nothing here") is None


def _openai_error(cls, message, headers=None, status=429):
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    import openai

    req = httpx.Request("POST", "http://x/v1/chat/completions")
    resp = httpx.Response(status, request=req, headers=headers or {})
    return getattr(openai, cls)(message, response=resp, body=None)


def _llm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    llm = LLM("groq:m")
    llm._sleep = lambda s: None
    return llm


def test_daily_quota_is_not_retried(monkeypatch):
    llm = _llm(monkeypatch)
    calls = []

    def attempt():
        calls.append(1)
        raise _openai_error("RateLimitError", "Rate limit reached ... tokens per day (TPD): Limit 100000. Please try again in 12m38.4s.")

    with pytest.raises(LLMError) as e:
        llm._request(attempt)
    assert len(calls) == 1 and e.value.kind == "quota" and e.value.retry_after == pytest.approx(758.4)


def test_short_rate_limit_is_retried(monkeypatch):
    llm = _llm(monkeypatch)
    calls = []

    def attempt():
        calls.append(1)
        if len(calls) < 3:
            raise _openai_error("RateLimitError", "slow down", headers={"retry-after": "1"})
        return "ok"

    assert llm._request(attempt) == "ok" and len(calls) == 3


def test_long_retry_after_header_is_hard_limit(monkeypatch):
    llm = _llm(monkeypatch)
    with pytest.raises(LLMError) as e:
        llm._request(lambda: (_ for _ in ()).throw(_openai_error("RateLimitError", "busy", headers={"retry-after": "900"})))
    assert e.value.kind == "quota" and e.value.retry_after == 900


def test_payment_required_is_quota(monkeypatch):
    llm = _llm(monkeypatch)
    with pytest.raises(LLMError) as e:
        llm._request(lambda: (_ for _ in ()).throw(_openai_error("APIStatusError", "pay up", status=402)))
    assert e.value.kind == "quota"


# ---- fallback ----------------------------------------------------------------------
def test_fallback_switches_and_remembers():
    a, b = FakeLLM(fail("quota"), fail("quota")), FakeLLM(say("from b"), say("again b"))
    switches = []
    f = FallbackLLM(a, [b], on_switch=lambda x, y, err: switches.append((x, y)))
    assert f.chat([{"role": "user", "content": "hi"}])["content"] == "from b"
    assert len(switches) == 1
    # a is cooling down: the next call goes straight to b, a is not asked again
    assert f.chat([{"role": "user", "content": "hi"}])["content"] == "again b"
    assert len(a.seen) == 1


def test_fallback_tries_primary_again_after_cooldown():
    a, b = FakeLLM(fail("rate_limit"), say("a is back")), FakeLLM(say("b"))
    f = FallbackLLM(a, [b], cooldown=0.0)
    assert f.chat([]) ["content"] == "b"
    assert f.chat([])["content"] == "a is back"


def test_all_models_failing_raises_exhausted():
    f = FallbackLLM(FakeLLM(fail("quota", "a down", retry_after=300)), [FakeLLM(fail("server", "b down"))])
    with pytest.raises(LLMError) as e:
        f.chat([])
    assert e.value.kind == "exhausted" and "a down" in str(e.value) and "b down" in str(e.value) and e.value.retry_after == 300


def test_bad_request_does_not_start_a_cooldown():
    a = FakeLLM(fail("bad_request"), say("a ok"))
    f = FallbackLLM(a, [FakeLLM(say("b"))])
    assert f.chat([])["content"] == "b"
    assert f.chat([])["content"] == "a ok"


def test_missing_key_fallback_is_skipped(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    f = FallbackLLM(FakeLLM(fail("quota")), ["deepseek:deepseek-chat", FakeLLM(say("third"))])
    assert f.chat([])["content"] == "third"


def test_stream_falls_back_when_nothing_was_sent_yet():
    f = FallbackLLM(FakeLLM(fail("quota")), [FakeLLM(say("hello world"))])
    got = []
    assert f.stream([], None, got.append)["content"] == "hello world"
    assert "".join(got).strip() == "hello world"


def test_agent_fallback_flag_and_trace():
    a, b = FakeLLM(fail("quota")), FakeLLM(say("saved by b"))
    bot = Agent(llm=a, fallback=[b], trace=True)
    assert bot.chat("hi") == "saved by b"
    assert [e.kind for e in bot.trace.events if e.kind == "fallback"] == ["fallback"]


def test_usage_adds_up_across_models():
    f = FallbackLLM(FakeLLM(fail("quota")), [FakeLLM(say("x"))])
    f.chat([])
    assert f.usage["calls"] == 2


# ---- agent resume --------------------------------------------------------------------
def test_agent_keeps_work_and_resumes():
    llm = FakeLLM(call("calculate", {"expression": "6*7"}), fail("quota"), say("The answer is 42"))
    bot = Agent(llm=llm, tools=["calculate"])
    with pytest.raises(LLMError):
        bot.chat("what is 6*7?")
    assert bot.interrupted
    assert [m["role"] for m in bot.messages] == ["user", "assistant", "tool"]  # the tool work is kept
    assert bot.resume() == "The answer is 42"
    assert not bot.interrupted
    # the model saw the finished tool result when it was asked again, and the user text was not repeated
    last_call = llm.seen[-1][0]
    assert any(m["role"] == "tool" and "42" in m["content"] for m in last_call)
    assert [m["content"] for m in last_call if m["role"] == "user"] == ["what is 6*7?"]


def test_resume_with_nothing_to_resume():
    bot = Agent(llm=FakeLLM(say("hi")))
    bot.chat("hello")
    with pytest.raises(BotError):
        bot.resume()


def test_agent_resume_survives_restart(tmp_path):
    db = str(tmp_path / "m.db")
    bot = Agent(llm=FakeLLM(call("calculate", {"expression": "2+2"}), fail("quota")), tools=["calculate"], memory=db)
    with pytest.raises(LLMError):
        bot.chat("2+2?")
    new = Agent(llm=FakeLLM(say("four")), tools=["calculate"], memory=db)   # a new process, same database
    assert new.interrupted
    assert new.resume() == "four"
    again = Agent(llm=FakeLLM(say("x")), tools=["calculate"], memory=db)
    assert not again.interrupted and again.messages[-1]["content"] == "four"


def test_non_llm_failure_still_rolls_back():
    class Boom(FakeLLM):
        def chat(self, messages, tools=None):
            raise RuntimeError("bug")

    bot = Agent(llm=Boom())
    with pytest.raises(RuntimeError):
        bot.chat("hi")
    assert bot.messages == []


def test_fallback_then_resume_when_everything_is_down():
    a, b = FakeLLM(fail("quota"), say("a back")), FakeLLM(fail("quota"))
    bot = Agent(llm=a, fallback=[b])
    with pytest.raises(LLMError) as e:
        bot.chat("task")
    assert e.value.kind == "exhausted"
    bot.llm._until.clear()          # pretend the limit is over
    assert bot.resume() == "a back"


# ---- graph resume --------------------------------------------------------------------
def _team(reviewer_llm, worker_llm=None, **kw):
    planner = Agent(name="planner", llm=FakeLLM(say("plan")))
    worker = Agent(name="worker", llm=worker_llm or FakeLLM(say("work")))
    reviewer = Agent(name="reviewer", llm=reviewer_llm)
    return chain(planner, worker, reviewer, **kw), planner, worker, reviewer


def test_graph_resumes_from_the_failed_node_only():
    g, planner, worker, reviewer = _team(FakeLLM(fail("quota"), say("final answer")))
    with pytest.raises(Exception):
        g.chat("do it")
    assert g.interrupted
    assert g.resume() == "final answer"
    assert len(planner.llm.seen) == 1 and len(worker.llm.seen) == 1     # not paid for twice
    assert not g.interrupted
    assert g.messages[-1]["content"] == "final answer"                  # the turn is recorded once
    assert g.trace.flow() == ["planner", "worker", "reviewer", "reviewer"]


def test_graph_checkpoint_file_survives_restart(tmp_path):
    ck = tmp_path / "run.json"
    g, *_ = _team(FakeLLM(fail("quota")), checkpoint=ck)
    with pytest.raises(Exception):
        g.chat("do it")
    data = json.loads(ck.read_text())
    assert data["frontier"] == ["reviewer"] and data["state"]["outputs"]["worker"] == "work"
    g2, planner2, worker2, _ = _team(FakeLLM(say("done after restart")), checkpoint=ck)   # new process
    assert g2.interrupted
    assert g2.resume() == "done after restart"
    assert not ck.exists()
    assert len(planner2.llm.seen) == 0 and len(worker2.llm.seen) == 0


def test_graph_parallel_level_keeps_finished_branches():
    fast = Agent(name="fast", llm=FakeLLM(say("fast result")))
    flaky = Agent(name="flaky", llm=FakeLLM(fail("quota"), say("flaky result")))
    g = Graph()
    g.add(fast); g.add(flaky)
    g.add(lambda s: s.outputs["fast"] + "+" + s.outputs["flaky"], name="join")
    g.start = ["fast", "flaky"]
    g.edge(fast, "join"); g.edge(flaky, "join"); g.edge("join", END)
    with pytest.raises(Exception):
        g.chat("go")
    assert g.resume() == "fast result+flaky result"
    assert len(fast.llm.seen) == 1                                      # the finished branch was not re-run


def test_resume_nothing_and_new_run_replaces_old():
    g, *_ = _team(FakeLLM(fail("quota")))
    with pytest.raises(BotError):
        g.resume()
    with pytest.raises(Exception):
        g.chat("one")
    assert g.interrupted
    g.reset()
    assert not g.interrupted


def test_graph_node_failure_with_fallback_never_pauses():
    reviewer = Agent(name="reviewer", llm=FakeLLM(fail("quota")), fallback=[FakeLLM(say("b reviewed"))])
    g = chain(Agent(name="planner", llm=FakeLLM(say("p"))), reviewer)
    assert g.chat("x") == "b reviewed" and not g.interrupted


def test_botkit_flags(monkeypatch):
    from botkit import BotKit

    monkeypatch.setenv("GROQ_API_KEY", "k")
    team = BotKit("groq:m", multi_agent=True, checkpoint="run.json")
    assert str(team.checkpoint) == "run.json"
    bot = BotKit("groq:m", fallback=["groq:other"])
    assert isinstance(bot.llm, FallbackLLM) and bot.llm.model == "groq:m"


def test_missing_key_on_primary_does_not_block_fallback(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    bot = Agent("groq:some-model", fallback=[FakeLLM(say("fallback works"))])
    assert bot.chat("hi") == "fallback works"


def test_checkpoint_is_written_after_every_level(tmp_path):
    ck = tmp_path / "run.json"
    seen = {}

    def second(state):
        seen["during"] = json.loads(ck.read_text())     # written after the first node finished, before the run ended
        return "two"

    g = Graph(checkpoint=ck)
    g.add(lambda s: "one", name="a"); g.add(second, name="b")
    g.edge("a", "b"); g.edge("b", END)
    assert g.chat("go") == "two"
    assert seen["during"]["frontier"] == ["b"] and seen["during"]["state"]["outputs"] == {"a": "one"}
    assert not ck.exists()                                               # cleaned up when the run finished


def test_crash_mid_run_resumes_in_a_new_process(tmp_path):
    ck = tmp_path / "run.json"
    calls = []

    def build(crash):
        def b(state):
            calls.append("b")
            if crash:
                raise KeyboardInterrupt()    # simulates the process being stopped
            return "two"

        g = Graph(checkpoint=ck)
        g.add(lambda s: calls.append("a") or "one", name="a"); g.add(b, name="b")
        g.edge("a", "b"); g.edge("b", END)
        return g

    with pytest.raises(BaseException):
        build(True).chat("go")
    g2 = build(False)
    assert g2.interrupted and g2.resume() == "two"
    assert calls.count("a") == 1


def test_fork_with_own_trace():
    from botkit import Trace

    g, *_ = _team(FakeLLM(say("ok")))
    t = Trace()
    child = g.fork("job1", trace=t)
    child.chat("go")
    assert t.flow() == ["planner", "worker", "reviewer"] and g.trace.flow() == []
