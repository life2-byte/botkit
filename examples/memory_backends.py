"""Choosing where memory lives, and adding your own tables. Needs GROQ_API_KEY (and Supabase keys for part 2)."""
from botkit import Agent, SQLiteStorage, SupabaseStorage, tool

MODEL = "groq:openai/gpt-oss-120b"

# 1) SQLite (default). Your own tables live in the same file:
db = SQLiteStorage(
    "bot.db",
    extra_schema="CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, item TEXT, qty INTEGER)",
)


@tool
def add_order(item: str, qty: int) -> str:
    """Save a customer order.

    Args:
        item: What was ordered.
        qty: How many.
    """
    db.execute("INSERT INTO orders(item, qty) VALUES(?, ?)", (item, qty))
    return "Order saved."


@tool
def list_orders() -> str:
    """Show all saved orders."""
    return str(db.query("SELECT * FROM orders"))


bot = Agent(MODEL, memory=db, tools=[add_order, list_orders])

# 2) Supabase instead of SQLite:
#    - run `python -m botkit.memory.supabase` and paste the SQL into the Supabase SQL editor (once)
#    - set SUPABASE_URL and SUPABASE_KEY (the service_role key), then:
#
#    store = SupabaseStorage()
#    bot = Agent(MODEL, memory=store)
#    store.table("orders").insert({"item": "tea", "qty": 2})       # your own tables work too
#    store.table("orders").select({"item": "tea"}, order="id.desc", limit=5)

if __name__ == "__main__":
    bot.run()
