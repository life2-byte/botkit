# backend.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

# Aapka botkit import aur bot setup (same as your example)
from botkit import Agent, tool

@tool
def order_status(order_id: str) -> str:
    """Look up an order."""
    return f"Order {order_id}: shipped"

bot = Agent(
    "groq:openai/gpt-oss-120b",
    tools=[True, order_status],
    skills=[True, Path(__file__).parent / "my_skills"],
    memory="full.db",
    rules=["Reply in the same language as the user."],
    # api_key="...."
)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

# NOTE: yeh function botkit ke actual API pe depend karta hai.
# Agar botkit mein direct single-turn method nahi hai,
# to aapko bot.run() ke code ko dekhkar ek loop-free version banana hoga.
async def get_bot_response(user_message: str) -> str:
    # Placeholder: botkit ke documentation ke mutabiq adjust karein.
    # Common possibilities:
    #   response = bot.chat(user_message)
    #   response = bot.agent.run(user_message)
    #   response = await bot.process(user_message)
    #
    # Filhal, hum maan rahe hain botkit mein aisa method hai:
    response = bot.chat(user_message)  # <-- YEH LINE BOTKIT PE DEPEND KARTI HAI
    return str(response)

@app.post("/chat")
async def chat(req: ChatRequest):
    reply = await get_bot_response(req.message)
    return {"reply": reply}

@app.get("/")
async def index():
    html = Path(__file__).parent / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))