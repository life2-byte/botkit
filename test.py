"""Quick check of 0.4.2: fallback + resume. Needs GROQ_API_KEY in the environment (or in .env)."""
import os
from pathlib import Path

env = Path(".env")
if env.exists() and "GROQ_API_KEY" not in os.environ:
    for line in env.read_text().splitlines():
        if line.startswith("GROQ_API_KEY="):
            os.environ["GROQ_API_KEY"] = line.split("=", 1)[1].strip().strip('"\'')

import botkit
from botkit import Agent, BotKit, LLMError
from botkit.testing import FakeLLM, call, fail, say

GOOD = "groq:openai/gpt-oss-120b"
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), name, extra)


check("version", botkit.__version__ == "0.4.2", botkit.__version__)

# 1) real fallback: first model does not exist -> falls back to a real one
bot = Agent("groq:this-model-does-not-exist", fallback=[GOOD], trace=True)
answer = bot.chat("Reply with exactly: pong")
sw = [e for e in bot.trace.events if e.kind == "fallback"]
check("fallback to real model", "pong" in answer.lower() and len(sw) == 1, answer[:40])

# 2) quota simulated, 2 models, work continues on the second
a, b = FakeLLM(fail("quota", retry_after=600)), FakeLLM(say("b took over"))
check("fake fallback", Agent(llm=a, fallback=[b]).chat("x") == "b took over")

# 3) everything down -> work kept -> resume later (agent loop)
llm = FakeLLM(call("calculate", {"expression": "6*7"}), fail("quota"), say("42"))
bot = Agent(llm=llm, tools=["calculate"])
try:
    bot.chat("6*7?")
    check("agent paused", False)
except LLMError as e:
    check("agent paused", bot.interrupted and e.kind == "quota", e.kind)
check("agent resume", bot.resume() == "42" and not bot.interrupted)

# 4) graph resume with real model for the first steps, limit on the last node
from botkit import Agent as A, chain
planner = A(GOOD, name="planner", system="Make a 2-line plan.")
worker = A(GOOD, name="worker", system="Do the plan briefly.")
reviewer = A(llm=FakeLLM(fail("quota"), say("APPROVED")), name="reviewer")
team = chain(planner, worker, reviewer, checkpoint="ck_test.json")
try:
    team.chat("Name two fruits.")
    check("graph paused", False)
except Exception as e:
    check("graph paused", team.interrupted and Path("ck_test.json").exists(), type(e).__name__)
calls_before = planner.llm.usage["calls"] + worker.llm.usage["calls"]
res = team.resume()
calls_after = planner.llm.usage["calls"] + worker.llm.usage["calls"]
check("graph resume (only reviewer re-ran)", calls_after == calls_before and not Path("ck_test.json").exists(), res[:40])

print("\nALL OK" if ok else "\nSOME FAILED")