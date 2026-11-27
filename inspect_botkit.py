import json, tempfile
from botkit import Agent
from botkit.testing import FakeLLM, say

work = tempfile.mkdtemp(prefix="SECRETPATH_")
llm = FakeLLM(say("ok"))
bot = Agent(llm=llm, tools=True, skills=True, rules=["Reply in Urdu", "Use write_file to create files"], workdir=work)
bot.chat("hello")

messages, tools = llm.seen[-1][0], llm.seen[-1][1]
system = messages[0]["content"]

print("=" * 70, "\nSYSTEM PROMPT sent to the model\n" + "=" * 70)
print(system)
print("=" * 70, "\nTOOLS sent in the `tools` field:", len(tools or []), "\n" + "=" * 70)
for t in tools or []:
    f = t["function"]
    print(f"- {f['name']}({', '.join(f['parameters'].get('properties', {}))})  {f['description'].splitlines()[0][:70]}")
print("=" * 70)
everything = json.dumps(messages) + json.dumps(tools)
print("Rules in system prompt      :", "Reply in Urdu" in system)
print("Skills listed in prompt     :", "skill" in system.lower())
print("Real workspace path leaked  :", "SECRETPATH_" in everything, "   (False = good)")
