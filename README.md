<div align="center">

# 🤖 botkit

**A chatbot / AI agent loop in a few lines of Python.**
Tools, skills, memory and MCP are optional. Switch on only what you need.

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Memory](https://img.shields.io/badge/memory-SQLite%20%7C%20Supabase%20%7C%20custom-orange)
![Models](https://img.shields.io/badge/models-Groq%20%7C%20OpenAI%20%7C%20OpenRouter%20%7C%20Gemini%20%7C%20Ollama-purple)

</div>

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [🎛️ Flags](#️-flags)
- [🛠️ Bundled Tools](#️-bundled-tools)
- [🧠 Bundled Skills](#-bundled-skills)
- [💾 Memory](#-memory)
- [🔌 MCP](#-mcp)
- [🔒 Rules & Security](#-rules--security)
- [🌐 HTTP Server (optional)](#-http-server-optional)
- [📁 Project Structure](#-project-structure)
- [🗺️ Roadmap](#️-roadmap)
- [🧪 Tests](#-tests)
- [📄 License](#-license)

---

## ✨ Features

- **Just the loop by default.** Model call, tool-call handling, retries and error recovery. Nothing else is switched on until you ask.
- **12 bundled tools.** Files, Word / Excel / PowerPoint, web fetch and search, calculator, optional shell.
- **10 bundled skills.** Research, planning, coding, debugging, writing and more. The model loads them only when needed.
- **Memory on any database.** SQLite out of the box, Supabase, or your own backend by implementing one small class. Add your own tables too.
- **MCP support.** Connect any MCP server (stdio, HTTP, SSE).
- **You choose where commands run.** No shell, local shell, or a locked-down Docker container.
- **Security enforced in code.** Workspace confinement, secret redaction, blocked internal URLs, and rules that can never loosen the security policy.
- **Many providers.** Groq, OpenAI, OpenRouter, Gemini, Ollama, or any OpenAI-compatible server.

## 📦 Installation

```bash
git clone https://github.com/YOUR-USERNAME/botkit.git
cd botkit
pip install -e .            # the loop only (needs just `openai`)
pip install -e ".[docs]"    # + Word / Excel / PowerPoint tools
pip install -e ".[mcp]"     # + MCP servers
pip install -e ".[all]"     # everything
```

Or straight from GitHub, for example in a `requirements.txt`:

```
botkit[all] @ git+https://github.com/YOUR-USERNAME/botkit.git
```

Set your API key (or pass `api_key=` in code):

```bash
export GROQ_API_KEY="gsk_..."      # or OPENAI_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY (Ollama needs none)
```

Models are written as `"provider:model"`: `groq:`, `openai:`, `openrouter:`, `gemini:`, `ollama:` (no prefix means OpenAI). Any other OpenAI-compatible server works with `base_url=` and `api_key=`.

## 🚀 Quick Start

The smallest bot, just the loop:

```python
from botkit import Agent

bot = Agent("groq:openai/gpt-oss-120b")
print(bot.chat("hello"))
bot.run()                      # or a terminal chat
```

Everything switched on:

```python
from botkit import Agent

bot = Agent(
    "groq:openai/gpt-oss-120b",
    tools=True,                # bundled tools
    skills=True,               # bundled skills
    memory="bot.db",           # SQLite memory
)
bot.run()
```

> Without `memory=`, the bot still remembers the conversation while the program runs. Memory adds saving across restarts plus long-term facts.

## 🎛️ Flags

| Flag | Default | What it does |
|---|---|---|
| `tools=` | off | `True` all bundled tools · `["files", "web"]` groups · `["read_file"]` single tools · your own functions · any mix |
| `skills=` | off | `True` bundled skills · `"research"` one by name · `"my_skills/"` your own folder · any mix |
| `memory=` | off | `"bot.db"` SQLite · `SupabaseStorage()` · your own `Storage` |
| `mcp=` | off | command string, URL, `MCP(...)`, list, or an `mcpServers` JSON file |
| `shell=` | off | `"local"` or `"docker"`. Turns on `run_command`. |
| `rules=` | none | a string, a list, or a text file added to the prompt |
| `workdir=` | `workspace/` | the only folder the file tools and shell can touch |
| `session=` | `"default"` | one conversation per user inside the same memory |

## 🛠️ Bundled Tools

| Group | Tools |
|---|---|
| 📂 `files` | `list_files` · `read_file` (also reads .docx / .xlsx / .pptx) · `write_file` · `edit_file` · `search_files` |
| 📄 `docs` | `create_docx` · `create_xlsx` · `create_pptx` |
| 🌍 `web` | `fetch_url` · `web_search` (DuckDuckGo; it scrapes a free page and may be rate-limited) |
| 💻 `shell` | `run_command` (only exists with `shell=`) |
| 🧮 `util` | `calculate` (safe arithmetic, no `eval`) |

Files the bot creates are saved in the workspace folder (`workspace/` where you run the script). Change it with `workdir="D:/my_files"`.

**Your own tool.** Put `@tool` on a function. Type hints and the docstring become the schema:

```python
from botkit import Agent, tool

@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: The order number.
    """
    return f"Order {order_id}: shipped"

bot = Agent("groq:openai/gpt-oss-120b", tools=["files", "calculate", order_status])
```

## 🧠 Bundled Skills

`research` · `planning` · `verify-work` · `writing-documents` · `presentations` · `spreadsheets` · `data-analysis` · `coding` · `debugging` · `summarizing`

Only each skill's name and one-line description go into the prompt. The model calls `load_skill` to read the full instructions when a task matches, so many skills cost almost no context.

**Your own skills** are folders with a `SKILL.md` (frontmatter `name` and `description`, then the instructions):

```python
Agent(model, skills=[True, "my_skills/"])    # bundled + yours
```

## 💾 Memory

Memory stores the conversation (survives restarts) and long-term facts (`remember` / `recall` / `forget` tools). It runs on a `Storage` backend, so the database is your choice.

```python
Agent(model, memory="bot.db")                    # SQLite (zero setup)
Agent(model, memory=SupabaseStorage())           # Supabase
Agent(model, memory=MyStorage())                 # anything you write
Agent(model, memory="bot.db", session="user42")  # one session per user
```

### Supabase

Run this once in the Supabase SQL editor (or print it with `python -m botkit.memory.supabase`):

```sql
create table if not exists botkit_messages (
  id bigint generated always as identity primary key,
  session text not null, data jsonb not null, created_at timestamptz not null default now());
create index if not exists botkit_messages_session_idx on botkit_messages (session, id);
create table if not exists botkit_facts (
  id bigint generated always as identity primary key,
  session text not null, text text not null, created_at timestamptz not null default now());
create index if not exists botkit_facts_session_idx on botkit_facts (session, id);
alter table botkit_messages enable row level security;
alter table botkit_facts enable row level security;
```

Then connect (use the **service_role** key, server side only, never in a browser or app):

```python
import os
from botkit import Agent, SupabaseStorage

store = SupabaseStorage(url="https://YOUR-PROJECT.supabase.co", key=os.environ["SUPABASE_KEY"])
bot = Agent("groq:openai/gpt-oss-120b", memory=store, session="ahmed")
```

### Your own tables

```python
from botkit import SQLiteStorage, SupabaseStorage

db = SQLiteStorage("bot.db", extra_schema="CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, item TEXT)")
db.execute("INSERT INTO orders(item) VALUES(?)", ("tea",))
db.query("SELECT * FROM orders")

store = SupabaseStorage()
store.table("orders").insert({"item": "tea"})
store.table("orders").select({"item": "tea"}, order="id.desc", limit=5)
```

Wrap these calls in `@tool` functions to let the bot use them. See `examples/memory_backends.py`.

### Any other database

Subclass `Storage` and implement six methods: `add_messages`, `get_messages`, `clear_messages`, `add_fact`, `list_facts`, `delete_fact`. A complete template is in `examples/custom_storage.py`.

## 🔌 MCP

```python
Agent(model, mcp="npx -y @modelcontextprotocol/server-filesystem .")     # local server
Agent(model, mcp="https://example.com/mcp")                              # remote server
Agent(model, mcp="mcp.json")                                             # Claude-Desktop style config
```

Requires `pip install -e ".[mcp]"`. A server that fails to start is skipped with a warning instead of crashing the bot.

## 🔒 Rules & Security

```python
Agent(model, rules=["Reply in Urdu", "Never discuss pricing"])     # or rules="rules.txt"
Agent(model, shell="docker", deny=[r"git push", r"pip install"])   # extra blocked commands
Agent(model, shell="local", approve=lambda name, args: input(f"Run {args['command']}? ") == "y")
```

Your rules go into the prompt, but the security rules always come last and cannot be loosened. The real enforcement is in code, not in the prompt:

- 📁 File tools and the shell are confined to the workspace folder (`..`, absolute paths and symlinks are refused).
- 💻 There is **no shell** unless you choose one in code: `"local"` (workspace only, API keys hidden from commands, dangerous commands blocked; a speed bump, not a jail) or `"docker"` (throwaway container: no network, read-only root, memory and CPU limits). Use Docker for anyone you do not fully trust.
- 🔑 API keys and tokens are redacted from anything a tool returns to the model.
- 🌐 `fetch_url` and `web_search` refuse private and internal addresses, including after redirects.
- 🧱 Tool output and web pages are marked as untrusted data (prompt-injection defence).
- ♻️ Tool errors, bad JSON, timeouts and repeated calls are fed back to the model so it can recover. API errors (429, 5xx) are retried with backoff.

## 🌐 HTTP Server (optional)

Expose the bot to a website or mobile app:

```python
bot.serve(host="0.0.0.0", port=8000, token="change-me", cors_origins=["https://myapp.com"])
```

```
POST /chat   {"message": "make a budget sheet", "session": "user42"}
GET  /files/<session>/<path>      download a file the bot created
GET  /health
```

Each `session` is a separate user with its own history, facts and workspace folder. Listening on a public address without a `token` is refused.

## 📁 Project Structure

```
botkit/
├── pyproject.toml
├── README.md
├── botkit/
│   ├── agent.py          # the loop (Agent)
│   ├── llm.py            # model client, retries
│   ├── tool.py           # @tool, Tool
│   ├── skill.py          # skill loader
│   ├── mcp.py            # MCP bridge
│   ├── security.py       # policy: workspace, commands, secrets
│   ├── sandbox.py        # local and Docker shell
│   ├── server.py         # optional HTTP API
│   ├── memory/           # Storage base, SQLite, Supabase, Memory
│   ├── tools/            # files.py docs.py web.py shell.py util.py
│   └── skills/           # 10 folders, each with a SKILL.md
├── examples/
└── tests/
```

## 🗺️ Roadmap

- [ ] Streaming replies
- [ ] Async API (`await bot.achat(...)`)
- [ ] Token-based context trimming (history is trimmed by message count today)
- [ ] `create_pdf` tool
- [ ] File storage backends (S3, Supabase Storage) like the memory backends
- [ ] Sub-agents
- [ ] Image input

## 🧪 Tests

```bash
pip install -e ".[dev]"
pytest
```

## 📄 License

MIT. See the `LICENSE` file.

---

<div align="center">

Built by **Ahmed** · If this helped you, give it a ⭐

</div>
