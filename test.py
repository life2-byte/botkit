"""botkit 0.4.1 check. Run:  python test_041.py
Part A-C need no API key. Part D-E use the real model (set GROQ_API_KEY)."""
import json
import os
import shutil
import sys

from botkit import Agent, BotKit, __version__
from botkit.testing import FakeLLM

MODEL = "groq:openai/gpt-oss-120b"
WS = "ws_test"
results = []


def check(name, ok, note=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {note}")


def tools(g, node):
    return set(g.nodes[node].target.tool_names)


shutil.rmtree(WS, ignore_errors=True)

# ---------------------------------------------------------------- A) version + tool inventory
print("A) version and tools")
check("version is 0.4.1", __version__ == "0.4.1", f"(got {__version__})")
everything = Agent(llm=FakeLLM(), tools=True, shell="local", workdir=WS)
check("17 bundled tools", len(everything.tool_names) == 17, f"(got {len(everything.tool_names)})")
cats = {t["category"] for t in everything.tools_info()}
check("tools have categories", cats == {"inspect", "edit", "documents", "web", "compute", "run"}, str(sorted(cats)))

# ---------------------------------------------------------------- B) who gets which tools
print("\nB) role-scoped tools")
team = BotKit(multi_agent=True, tools=True, shell="local", workdir=WS, llm=FakeLLM())
for name in team.nodes:
    print(f"   {name:9}", sorted(tools(team, name)))
check("planner can look but not edit/run", "read_file" in tools(team, "planner") and not tools(team, "planner") & {"write_file", "edit_file", "run_command"})
check("worker has everything", {"write_file", "create_pdf", "run_command", "run_python"} <= tools(team, "worker"))
check("reviewer can read AND run", {"read_file", "tree", "run_command", "run_python"} <= tools(team, "reviewer"))
check("reviewer cannot edit", not tools(team, "reviewer") & {"write_file", "edit_file", "create_docx", "create_pdf"})
noshell = BotKit(multi_agent=True, tools=True, workdir=WS, llm=FakeLLM())
check("no shell given -> nobody can run", not any(tools(noshell, n) & {"run_command", "run_python"} for n in noshell.nodes))
everyone = BotKit(multi_agent=True, tools=True, shell="local", role_tools=False, workdir=WS, llm=FakeLLM())
check("role_tools=False -> everyone has all", "write_file" in tools(everyone, "reviewer") and "write_file" in tools(everyone, "planner"))
plain = BotKit(MODEL, api_key="x", mode="sequential", tools=["inspect", "edit"], workdir=WS, agents=["groq:other-model"])
check("a plain new model inherits the app's tools", "write_file" in tools(plain, "agent1"))

# ---------------------------------------------------------------- C) the new tools, called directly (no model)
print("\nC) new tools")
os.makedirs(f"{WS}/pkg", exist_ok=True)
with open(f"{WS}/notes.txt", "w") as f:
    f.write("\n".join(f"line {i}" for i in range(1, 101)))


def run(name, **args):
    return everything._exec({"id": "x", "function": {"name": name, "arguments": json.dumps(args)}}, {})


check("read_file slice", run("read_file", path="notes.txt", start=10, lines=2) == "[lines 10-11 of 100]\n10: line 10\n11: line 11")
check("tree", "pkg/" in run("tree") and "notes.txt" in run("tree"))
check("file_info", "100 lines" in run("file_info", path="notes.txt"))
check("current_time", "UTC" in run("current_time", timezone="UTC"))
check("run_python", "5050" in run("run_python", code="print(sum(range(101)))"))
check("run_command", "hello" in run("run_command", command="echo hello"))
r = run("create_pdf", path="r.pdf", title="T", content="# Hi\n- a\n- b")
check("create_pdf", r == "Created r.pdf" and open(f"{WS}/r.pdf", "rb").read(4) == b"%PDF", r)
check("files stay confined", run("read_file", path="../../etc/passwd").startswith("Error"))

# ---------------------------------------------------------------- D/E) real model
if not os.environ.get("GROQ_API_KEY"):
    print("\nD/E) skipped: set GROQ_API_KEY to test with the real model")
else:
    print("\nD) real team: worker writes + runs code, reviewer verifies (takes ~1 minute)")
    ws_d = f"{WS}/d"
    t = BotKit(MODEL, multi_agent=True, tools=True, shell="local", workdir=ws_d)
    out = t.chat("Write fib.py that prints the first 10 Fibonacci numbers, run it to check, and tell me the output.")
    print("   answer:", out[:200].replace("\n", " "))
    s = t.trace.summary()
    reviewer_tools = [e.data["name"] for e in t.trace.kinds("tool_call") if e.data["agent"] == "reviewer"]
    print("   flow:", t.trace.flow(), "| reviewer used:", reviewer_tools)
    check("fib.py was created", os.path.exists(f"{ws_d}/fib.py"))
    check("answer contains the sequence", "34" in out)
    check("no errors in the trace", s["errors"] == 0 and s["tool_errors"] == 0, str(s))
    check("reviewer used tools to verify", bool(reviewer_tools), "(model's choice; if FAIL, run D once more)")
    check("reviewer never edited", not set(reviewer_tools) & {"write_file", "edit_file", "create_pdf"})

    print("\nE) bait: worker is told to write a WRONG answer first; does the reviewer catch it by opening the file?")
    ws_e = f"{WS}/e"
    bait = BotKit(
        MODEL, mode="sequential", tools=True, workdir=ws_e, review=True,
        agents=[
            {"role": "worker", "system": "You are the Worker. FIRST attempt: write the number 41 into answer.txt and say 'done'. "
             "If you get CHANGES NEEDED feedback, fix answer.txt with the correct value and say 'fixed'."},
            "reviewer",
        ],
    )
    out = bait.chat("Put the result of 6*7 into answer.txt")
    flow = bait.trace.flow()
    final = open(f"{ws_e}/answer.txt").read().strip() if os.path.exists(f"{ws_e}/answer.txt") else "(missing)"
    rv = [e.data["name"] for e in bait.trace.kinds("tool_call") if e.data["agent"] == "reviewer"]
    print("   flow:", flow, "| reviewer used:", rv, "| answer.txt:", final)
    check("reviewer caught the wrong value (loop ran)", flow[:4] == ["worker", "reviewer", "worker", "reviewer"])
    check("file ends with 42", "42" in final)
    check("reviewer opened the real file", "read_file" in rv)

print("\n" + "=" * 50)
failed = [n for n, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
for n in failed:
    print("  FAILED:", n)
sys.exit(1 if failed else 0)