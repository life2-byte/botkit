"""botkit - an agent loop, plus optional tools, skills, memory, MCP and multi-agent workflows.

    from botkit import BotKit

    bot = BotKit("groq:openai/gpt-oss-120b")                      # one agent loop; API key from the environment
    print(bot.chat("hello"))

    bot = BotKit(model, tools=True, skills=True, memory="bot.db") # switch features on
    team = BotKit(model, multi_agent=True)                        # planner -> worker -> reviewer
    team = BotKit(model, mode="parallel", agents=["researcher", "researcher", "critic"])

`Agent`, `Graph`, `chain` and `parallel` are there when you want full control.
"""
from .agent import Agent
from .errors import BotError, GraphError, LLMError
from .graph import END, Graph, State, chain, parallel
from .kit import BotKit
from .llm import LLM, FallbackLLM, make_llm, register_provider
from .mcp import MCP
from .memory import Fact, Memory, SQLiteStorage, Storage, SupabaseStorage
from .sandbox import DockerSandbox, LocalSandbox, Sandbox
from .security import Policy
from .skill import Skill
from .tool import Tool, tool
from .trace import Trace

__all__ = [
    "BotKit", "Agent", "Graph", "State", "END", "chain", "parallel", "tool", "Tool", "Skill", "MCP", "Memory",
    "Storage", "SQLiteStorage", "SupabaseStorage", "Fact", "Policy", "Sandbox", "LocalSandbox", "DockerSandbox",
    "LLM", "FallbackLLM", "make_llm", "register_provider", "Trace", "BotError", "LLMError", "GraphError",
]
__version__ = "0.4.4"
