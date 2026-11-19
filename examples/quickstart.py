"""The smallest bot: just the loop.  Run: GROQ_API_KEY=... python examples/quickstart.py"""
from botkit import Agent

bot = Agent("groq:openai/gpt-oss-120b")
bot.run()          # terminal chat, or: print(bot.chat("hello"))
