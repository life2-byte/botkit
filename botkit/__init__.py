"""botkit - a chatbot/agent loop with optional tools, skills, memory and MCP.

    from botkit import Agent

    bot = Agent("groq:openai/gpt-oss-120b")     # just the loop; API key from the environment
    print(bot.chat("hello"))

    bot = Agent("groq:openai/gpt-oss-120b", tools=True, skills=True, memory="bot.db")   # switch features on
"""
from .agent import Agent
from .errors import BotError, LLMError
from .llm import LLM
from .mcp import MCP
from .memory import Fact, Memory, SQLiteStorage, Storage, SupabaseStorage
from .sandbox import DockerSandbox, LocalSandbox, Sandbox
from .security import Policy
from .skill import Skill
from .tool import Tool, tool

__all__ = [
    "Agent", "tool", "Tool", "Skill", "MCP", "Memory", "Storage", "SQLiteStorage", "SupabaseStorage", "Fact",
    "Policy", "Sandbox", "LocalSandbox", "DockerSandbox", "LLM", "BotError", "LLMError",
]
__version__ = "0.3.0"
