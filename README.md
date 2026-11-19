# botkit

A chatbot/agent loop as a library. By default you get **only the loop** (model call, tool-call handling, retries,
error handling). Everything else is a flag you switch on and wire where you like.

```python
from botkit import Agent

bot = Agent("groq:openai/gpt-oss-120b")          # key from GROQ_API_KEY
print(bot.chat("hello"))                          # or bot.run() for a terminal chat

bot = Agent("groq:openai/gpt-oss-120b", tools=True, skills=True, memory="bot.db")   # now with everything
```

## Install

```bash
pip install -e .                 # the loop only (needs just `openai`)
pip install -e ".[docs]"         # + Word/Excel/PowerPoint tools
pip install -e ".[mcp]"          # + MCP servers
pip install -e ".[all]"          # everything
export GROQ_API_KEY=...          # or OPENAI_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY (Ollama needs none)
```

Models are `"provider:model"`: `groq:`, `openai:`, `openrouter:`, `gemini:`, `ollama:` (no prefix = OpenAI).
Other OpenAI-compatible servers: `base_url=` and `api_key=`.

## The flags

| Flag | Default | What it switches on |
|---|---|---|
| `tools=` | none | `True` = all bundled tools · `["files", "web"]` groups · `["read_file"]` single tools · your own functions · any mix |
| `skills=` | none | `True` = bundled skills · `"research"` one by name · `"my_skills/"` your folder · `Skill(...)` · any mix |
| `memory=` | none (history in RAM only) | `"bot.db"` SQLite · `SupabaseStorage()` · your own `Storage` · `Memory(...)` |
| `mcp=` | none | command string, URL, `MCP(...)`, list, or an `mcpServers` JSON file |
| `shell=` | none | `"local"` or `"docker"` (or a dict / `Sandbox`). Turns on `run_command`. |
| `rules=` | none | string, list, or a text file; added to the prompt |

## Bundled tools (12): `botkit/tools/`

| Group | Tools |
|---|---|
| `files` | `list_files` `read_file` (also reads .docx/.xlsx/.pptx) `write_file` `edit_file` `search_files` |
| `docs` | `create_docx` `create_xlsx` `create_pptx` |
| `web` | `fetch_url` `web_search` (DuckDuckGo; scraping a free page, may be rate-limited) |
| `shell` | `run_command` (only with `shell=`) |
| `util` | `calculate` (safe arithmetic) |

Your own tool: `@tool` on a function. Type hints and the docstring become the schema.

```python
from botkit import Agent, tool

@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: The order number.
    """
    return "shipped"

bot = Agent(tools=["files", "calculate", order_status])
```

## Bundled skills (10): `botkit/skills/`

`research` · `planning` · `verify-work` · `writing-documents` · `presentations` · `spreadsheets` ·
`data-analysis` · `coding` · `debugging` · `summarizing`

Only each skill's name and one-line description go into the prompt; the model loads the full text with
`load_skill` when a task matches. Your own skills are folders with a `SKILL.md`
(frontmatter `name` and `description`, then the instructions): `Agent(skills=[True, "my_skills/"])`.

## Memory: choose the database

Memory is history (survives restarts) plus long-term facts (`remember` / `recall` / `forget` tools).
It sits on a `Storage` backend, so the database is your choice:

```python
Agent(memory="bot.db")                              # SQLite (zero setup)
Agent(memory=SupabaseStorage())                     # Supabase
Agent(memory=MyStorage())                           # anything you write (see below)
Agent(memory="bot.db", session="user42")            # one session per user, same database
```

**Supabase.** Run once in the Supabase SQL editor (`python -m botkit.memory.supabase` prints it), then set
`SUPABASE_URL` and `SUPABASE_KEY` (the `service_role` key, server side only):

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

**Your own tables** (same connection):

```python
db = SQLiteStorage("bot.db", extra_schema="CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, item TEXT)")
db.execute("INSERT INTO orders(item) VALUES(?)", ("tea",));  db.query("SELECT * FROM orders")

store = SupabaseStorage()
store.table("orders").insert({"item": "tea"});  store.table("orders").select({"item": "tea"}, order="id.desc")
```

Wrap those calls in `@tool` functions to let the bot use them (`examples/memory_backends.py`).

**Any other database:** subclass `Storage` and implement 6 methods (`add_messages`, `get_messages`,
`clear_messages`, `add_fact`, `list_facts`, `delete_fact`). `examples/custom_storage.py` is a complete template.

## Rules and security

`rules=["Reply in Urdu"]` adds your rules to the prompt; the security rules always come last and cannot be loosened.
Enforcement is in code, not in the prompt:

- File tools and the shell are confined to the workspace folder (`..`, absolute paths and symlinks are refused).
- The shell does not exist unless you choose an environment: `"local"` (workspace only, API keys hidden, dangerous
  commands blocked; a speed bump, not a jail) or `"docker"` (throwaway container: no network, read-only root,
  memory/CPU limits). `deny=[r"git push"]` blocks more commands, `approve=callback` asks you before each command.
- API keys and tokens are redacted from anything a tool returns to the model.
- `fetch_url` and `web_search` refuse private/internal addresses, also after redirects.
- Tool output and web pages are marked as untrusted data (prompt-injection defence).

## Layout

```
botkit/
  pyproject.toml  README.md
  botkit/
    agent.py        the loop (Agent)
    llm.py          model client, retries
    tool.py         @tool, Tool
    skill.py        skill loader
    mcp.py          MCP bridge
    security.py  sandbox.py  errors.py
    server.py       optional HTTP API: Agent(...).serve()
    memory/         Storage base, SQLite, Supabase, Memory
    tools/          files.py docs.py web.py shell.py util.py
    skills/         10 folders, each with a SKILL.md
  examples/  tests/
```

## Not included (yet)

Streaming replies, async API, token-based context trimming (history is trimmed by message count),
sub-agents, image input, scheduled tasks.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
