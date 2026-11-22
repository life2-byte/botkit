import asyncio
import threading
import time
import warnings

import pytest

from botkit import (END, Agent, BotKit, Graph, GraphError, LLMError, Memory, SQLiteStorage, chain, parallel, tool)
from botkit.llm import PROVIDERS, parse_model, register_provider
from botkit.memory import compact_history, estimate_tokens, trim_to_tokens
from botkit.testing import FakeLLM, assert_flow, assert_tools_called, call, say


def ag(name, *replies, **kw):
    return Agent(name=name, llm=FakeLLM(*replies), **kw)


# ---------------------------------------------------------------- the three modes, one engine
def test_single_is_a_plain_agent():
    bot = BotKit(llm=FakeLLM(say("hi")))
    assert isinstance(bot, Agent) and bot.chat("x") == "hi"


def test_sequential_passes_outputs_forward():
    planner, worker = ag("planner", say("1. do it")), ag("worker", say("done"))
    g = chain(planner, worker, trace=True)
    assert g.chat("build a thing") == "done"
    assert_flow(g, ["planner", "worker"])
    seen = worker.llm.prompts[0]
    assert "build a thing" in seen and "Output from planner" in seen and "1. do it" in seen


def test_pipe_operator_builds_a_chain():
    a, b, c = ag("a", say("A")), ag("b", say("B")), ag("c", say("C"))
    g = a | b | c
    assert isinstance(g, Graph) and g.chat("go") == "C"
    assert_flow(g, ["a", "b", "c"])


def test_parallel_runs_at_the_same_time_and_merges():
    barrier = threading.Barrier(3, timeout=5)  # all three must be inside the model call together

    def slow(name):
        def fn(messages, tools):
            barrier.wait()
            return say(f"{name}-out")
        return Agent(name=name, llm=FakeLLM(fn=fn))

    g = parallel([slow("a"), slow("b"), slow("c")], merge=ag("merger", say("merged")))
    assert g.chat("task") == "merged"
    merge_prompt = g.nodes["merger"].target.llm.prompts[0]
    assert all(f"{n}-out" in merge_prompt for n in "abc")


def test_parallel_without_merger_joins_outputs_and_survives_a_failed_branch():
    g = parallel([ag("a", say("A")), ag("b", LLMError("down"))])
    out = g.chat("x")
    assert "## a\nA" in out and "Error" in out


def test_parallel_agents_sharing_a_folder_get_their_own(tmp_path):
    a, b = Agent(name="a", llm=FakeLLM(say("1")), workdir=tmp_path), Agent(name="b", llm=FakeLLM(say("2")), workdir=tmp_path)
    g = parallel([a, b])
    g.chat("x")
    assert a.policy.workdir == tmp_path.resolve() / "a" and b.policy.workdir == tmp_path.resolve() / "b"


# ---------------------------------------------------------------- the correction loop
def test_review_loop_sends_work_back_until_approved():
    worker = ag("worker", say("draft 1"), say("draft 2"))
    reviewer = ag("reviewer", say("CHANGES NEEDED: add detail"), say("APPROVED"))
    g = chain(worker, reviewer, review=True)
    assert g.chat("write") == "draft 2"                      # the answer is the worker's result, not the verdict
    assert_flow(g, ["worker", "reviewer", "worker", "reviewer"])
    assert "add detail" in worker.llm.prompts[1]             # the worker saw the feedback


def test_review_loop_is_bounded():
    worker = Agent(name="worker", llm=FakeLLM(fn=lambda m, t: say(f"v{len(m)}x{time.monotonic()}")))
    reviewer = ag("reviewer", *[say("CHANGES NEEDED: no") for _ in range(10)])
    g = chain(worker, reviewer, review=True, max_rounds=2)
    out = g.chat("write")
    assert out.startswith("v") and g.trace.flow() == ["worker", "reviewer", "worker", "reviewer"]


# ---------------------------------------------------------------- custom graphs
def test_custom_graph_branches_on_state():
    router = lambda s: "code" if "bug" in s.task else "text"
    g = Graph()
    g.add(router, name="router")
    g.add(ag("coder", say("fixed")), name="coder")
    g.add(ag("writer", say("written")), name="writer")
    g.edge("router", "coder", when=lambda s: s.last == "code")
    g.edge("router", "writer")
    assert g.invoke("there is a bug") == "fixed"
    assert g.invoke("hello") == "written"


def test_function_nodes_share_data_through_state():
    g = Graph()
    g.add(lambda s: {"n": 41}, name="a")
    g.add(lambda s: str(s.shared["n"] + 1), name="b")
    g.edge("a", "b")
    assert g.invoke("x") == "42"


def test_loops_are_stopped_by_step_limit():
    g = Graph(max_steps=6, max_visits=100)
    counter = {"n": 0}

    def tick(s):
        counter["n"] += 1
        return f"tick {counter['n']}"

    g.add(tick, name="loop")
    g.edge("loop", "loop")
    assert g.invoke("x").startswith("tick")
    assert counter["n"] == 6 and g.trace.kinds("step_limit")


def test_repeated_identical_output_ends_a_loop():
    g = Graph(max_steps=50, max_visits=50)
    g.add(lambda s: "same", name="n")
    g.edge("n", "n")
    assert g.invoke("x") == "same"
    assert g.trace.kinds("no_progress") and len(g.trace.flow()) == 2


def test_subgraph_as_a_node():
    inner = chain(ag("x", say("X")), ag("y", say("Y")))
    outer = Graph()
    outer.add(inner, name="inner")
    outer.add(ag("z", say("Z")), name="z")
    outer.edge("inner", "z")
    assert outer.invoke("t") == "Z"


def test_bad_graphs_are_rejected():
    g = Graph()
    with pytest.raises(GraphError, match="no nodes"):
        g.invoke("x")
    g.add(lambda s: "a", name="a")
    g.add(lambda s: "b", name="b")
    with pytest.raises(GraphError, match="never run"):
        g.invoke("x")
    with pytest.raises(GraphError, match="unknown node"):
        g.edge("a", "nope")
    with pytest.raises(GraphError):
        g.add(42)


def test_node_failure_names_the_node():
    g = chain(ag("a", say("A")), ag("b", LLMError("model down")))
    with pytest.raises(GraphError, match="node 'b' failed"):
        g.chat("x")


# ---------------------------------------------------------------- BotKit flags pick the mode
def fake_role_llm():
    return FakeLLM(fn=lambda m, t: say(("APPROVED" if "Reviewer" in m[0]["content"] else "ok") + ":" + m[0]["content"][:14]))


def test_multi_agent_flag_builds_planner_worker_reviewer():
    g = BotKit(multi_agent=True, llm=fake_role_llm())
    assert isinstance(g, Graph) and list(g.nodes) == ["planner", "worker", "reviewer"]
    assert g.chat("hi").startswith("ok:")
    assert g.trace.flow() == ["planner", "worker", "reviewer"]


def test_review_flag_builds_worker_and_reviewer():
    g = BotKit(review=True, llm=fake_role_llm())
    assert list(g.nodes) == ["worker", "reviewer"]


def test_parallel_mode_with_roles_and_models_per_agent():
    g = BotKit(mode="parallel", agents=["researcher", {"role": "critic", "name": "c2"}], llm=fake_role_llm())
    assert list(g.nodes) == ["researcher", "c2", "synthesizer"]
    assert g.chat("q")


def test_different_models_per_agent():
    g = BotKit("groq:m1", api_key="k", mode="sequential",
               agents=["planner", {"role": "worker", "model": "deepseek:deepseek-chat", "api_key": "k2"}, "claude:claude-x"],
               )
    models = [(n.target.llm.provider, n.target.llm.model) for n in g.nodes.values()]
    assert models == [("groq", "m1"), ("deepseek", "deepseek-chat"), ("claude", "claude-x")]


def test_mode_rules():
    with pytest.raises(GraphError, match="parallel"):
        BotKit(mode="parallel", llm=FakeLLM())
    with pytest.raises(GraphError, match="custom"):
        BotKit(mode="custom", llm=FakeLLM())
    g = chain(ag("a", say("A")))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert BotKit(mode="parallel", graph=g) is g
    assert any("graph= was given" in str(x.message) for x in w)


def test_planner_and_reviewer_do_not_get_the_apps_tools():
    g = BotKit(multi_agent=True, tools=["calculate"], llm=FakeLLM())
    assert g.nodes["worker"].target.tool_names == ["calculate"]
    assert g.nodes["planner"].target.tool_names == [] and g.nodes["reviewer"].target.tool_names == []


# ---------------------------------------------------------------- memory / users
def test_graph_memory_and_sessions(tmp_path):
    store = SQLiteStorage(tmp_path / "g.db")

    def build():
        return chain(Agent(name="a", llm=FakeLLM(fn=lambda m, t: say("ans"))), memory=Memory(store, session="u1"))

    g = build()
    g.chat("I am Ahmed")
    g2 = build()                                     # a restart: history comes back
    g2.chat("who am I?")
    assert "I am Ahmed" in g2.nodes["a"].target.llm.prompts[0]
    other = g2.fork("u2")
    other.chat("hello")
    assert "I am Ahmed" not in other.nodes["a"].target.llm.prompts[-1]
    assert other.nodes["a"].target is not g2.nodes["a"].target


# ---------------------------------------------------------------- streaming, async, tracing, loop optimizations
def test_streaming_agent_and_graph():
    bot = Agent(llm=FakeLLM(say("hello big world")))
    assert "".join(bot.stream("x")) == "hello big world "
    g = chain(ag("a", say("A")), ag("b", say("final answer here")))
    assert "".join(g.stream("x")).strip() == "final answer here"


def test_stream_raises_model_errors():
    with pytest.raises(LLMError):
        list(Agent(llm=FakeLLM(LLMError("down"))).stream("x"))


def test_async_api():
    async def go():
        bot = Agent(llm=FakeLLM(say("one"), say("two three")))
        assert await bot.achat("a") == "one"
        return "".join([p async for p in bot.astream("b")])

    assert asyncio.run(go()).strip() == "two three"


def test_trace_records_models_tools_and_summary():
    @tool
    def add(a: int, b: int) -> int:
        """Add.

        Args:
            a: x
            b: y
        """
        return a + b

    bot = Agent(llm=FakeLLM(call("add", {"a": 1, "b": 2}), say("3")), tools=[add], trace=True)
    bot.chat("x")
    s = bot.trace.summary()
    assert s["llm_calls"] == 2 and s["tool_calls"] == 1 and s["errors"] == 0
    assert_tools_called(bot, ["add"])
    assert '"tool_call"' in bot.trace.to_json()


def test_tools_in_one_reply_run_at_the_same_time():
    barrier = threading.Barrier(2, timeout=5)

    @tool
    def meet(who: str) -> str:
        """Meet.

        Args:
            who: name
        """
        barrier.wait()      # only passes if both calls are running together
        return f"met {who}"

    two = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "meet", "arguments": '{"who": "a"}'}},
        {"id": "2", "type": "function", "function": {"name": "meet", "arguments": '{"who": "b"}'}}]}
    bot = Agent(llm=FakeLLM(two, say("ok")), tools=[meet])
    assert bot.chat("x") == "ok"
    results = [m["content"] for m in bot.messages if m["role"] == "tool"]
    assert results == ["met a", "met b"]                         # order is kept


def test_tools_info_and_discovery():
    bot = Agent(llm=FakeLLM(), tools=["calculate"], memory=None)
    info = {t["name"]: t["source"] for t in bot.tools_info()}
    assert info == {"calculate": "bundled"}


def test_context_is_compacted_and_trimmed_by_tokens():
    big = {"role": "tool", "tool_call_id": "1", "content": "x" * 5000}
    assert len(compact_history([big])[0]["content"]) < 1400 and len(big["content"]) == 5000   # copy, not edited
    msgs = [{"role": "user", "content": "q" * 700}, {"role": "assistant", "content": "a" * 700}] * 10
    kept = trim_to_tokens(msgs, 1000)
    assert estimate_tokens(kept) <= 1000 and kept[0]["role"] == "user" and len(kept) < len(msgs)
    assert trim_to_tokens(msgs, None) == msgs


# ---------------------------------------------------------------- providers
def test_providers():
    assert parse_model("deepseek:deepseek-chat") == ("deepseek", "deepseek-chat")
    assert parse_model("claude:claude-sonnet-4-5") == ("claude", "claude-sonnet-4-5")
    assert parse_model("gpt-4o") == ("openai", "gpt-4o")
    register_provider("mycorp", "https://llm.corp/v1", "CORP_KEY")
    assert parse_model("mycorp:big") == ("mycorp", "big") and PROVIDERS["mycorp"][1] == "CORP_KEY"
    bot = Agent("mycorp:big", api_key="k")
    assert str(bot.llm._client.base_url).startswith("https://llm.corp/v1")


def test_llm_stream_assembles_text_and_tool_calls():
    from types import SimpleNamespace as NS

    from botkit.llm import LLM

    def chunk(content=None, tcs=None):
        return NS(choices=[NS(delta=NS(content=content, tool_calls=tcs))])

    def tc(i, id=None, name=None, args=None):
        return NS(index=i, id=id, function=NS(name=name, arguments=args))

    chunks = [chunk("Hel"), chunk("lo"), chunk(tcs=[tc(0, "c1", "add", '{"a":')]), chunk(tcs=[tc(0, None, None, ' 1}')]),
              chunk(tcs=[tc(1, "c2", "sub", "{}")])]
    llm = LLM("groq:m", api_key="k")
    llm._client = NS(chat=NS(completions=NS(create=lambda **kw: iter(chunks))))
    tokens = []
    msg = llm.stream([{"role": "user", "content": "x"}], None, tokens.append)
    assert tokens == ["Hel", "lo"] and msg["content"] == "Hello"
    assert [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in msg["tool_calls"]] == [
        ("c1", "add", '{"a": 1}'), ("c2", "sub", "{}")]


def test_real_tools_inside_a_forked_workflow_write_to_the_users_folder(tmp_path):
    planner = Agent(name="planner", llm=FakeLLM(say("1. write the file")), workdir=tmp_path)
    worker = Agent(name="worker", llm=FakeLLM(call("write_file", {"path": "note.txt", "content": "hi"}), say("saved")),
                   tools=["files"], workdir=tmp_path)
    g = chain(planner, worker)
    user = g.fork("ahmed")
    user.nodes["worker"].target.llm = FakeLLM(call("write_file", {"path": "note.txt", "content": "hi"}), say("saved"))
    assert user.chat("save a note") == "saved"
    assert (tmp_path / "ahmed" / "note.txt").read_text() == "hi"
    assert user.policy.workdir == tmp_path.resolve() / "ahmed"
