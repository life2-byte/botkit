"""Multi-agent examples. Each block is independent; run one at a time.  pip install "techtide-botkit[all]"
Set the keys of the providers you use (GROQ_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, ...)."""
from botkit import END, Agent, BotKit, Graph, chain, parallel

MODEL = "groq:openai/gpt-oss-120b"

# 1) The simplest team: planner -> worker -> reviewer (the reviewer sends work back until it approves)
team = BotKit(MODEL, multi_agent=True, tools=True)
print(team.chat("Make a 5-slide deck outline about solar energy and save it as outline.pptx"))
team.trace.print()                      # who ran, how long, which tools

# 2) Your own agents, each on a different model
planner = Agent("groq:openai/gpt-oss-120b", name="planner", system="Write a short numbered plan.")
worker = Agent("deepseek:deepseek-chat", name="worker", tools=True)
reviewer = Agent("claude:claude-sonnet-4-5", name="reviewer")
flow = chain(planner, worker, reviewer, review=True)            # == planner | worker | reviewer + correction loop
print(flow.chat("Write a short report on electric scooters"))

# 3) Parallel: three researchers at the same time, then one writer merges
research = BotKit(MODEL, mode="parallel", agents=["researcher", "researcher", "critic"], merge="synthesizer", tools=["web"])
print(research.chat("Pros and cons of remote work"))

# 4) Custom graph with a branch
coder = Agent(MODEL, name="coder", tools=["files"])
writer = Agent(MODEL, name="writer")
g = Graph()
g.add(lambda s: "code" if "bug" in s.task.lower() else "text", name="router")
g.add(coder)
g.add(writer)
g.edge("router", coder, when=lambda s: s.last == "code")
g.edge("router", writer)                                         # default branch
g.edge(coder, END)
print(BotKit(graph=g).chat("There is a bug in my loop"))
