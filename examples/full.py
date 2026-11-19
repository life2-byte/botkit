"""Everything switched on. Remove any flag you don't want.  Run: GROQ_API_KEY=... python examples/full.py"""
from pathlib import Path

from botkit import Agent, tool


@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: The order number.
    """
    return f"Order {order_id}: shipped"


bot = Agent(
    "groq:openai/gpt-oss-120b",
    tools=[True, order_status],                      # all bundled tools + your own function
    skills=[True, Path(__file__).parent / "my_skills"],   # bundled skills + your own folder
    memory="full.db",                                # SQLite file (or SupabaseStorage(), or your own Storage)
    # mcp="mcp.json",                                # MCP servers (pip install 'botkit[mcp]')
    # shell="docker",                                # lets the bot run commands, in an isolated container
    rules=["Reply in the same language as the user."],
)
bot.run()
