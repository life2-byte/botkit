# botkit

An agent loop as a library. By default you get **only the loop** (model call, tool-call handling, retries,
error handling). Everything else is a flag you switch on: tools, skills, memory, MCP, a shell, and
**multi-agent workflows** (sequential, parallel, custom graphs). Simple outside, complicated inside.

```python
from botkit import BotKit

bot = BotKit("groq:openai/gpt-oss-120b")                         # key from GROQ_API_KEY
print(bot.chat("hello"))                                          # or bot.run() for a terminal chat

bot = BotKit(model, tools=True, skills=True, memory="bot.db")     # single agent with features
team = BotKit(model, multi_agent=True)                            # planner -> worker -> reviewer
team = BotKit(model, mode="parallel", agents=["researcher", "researcher", "critic"])
```

`BotKit(...)` returns an `Agent` (single mode) or a `Graph` (multi-agent modes). Both have the same API:
`chat`, `stream`, `achat`, `astream`, `run`, `serve`, `fork`, `close`. The older `Agent(...)` still works unchanged.

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

## Which loop do I get? (you decide, with flags)

| You write | You get |
|---|---|
| `BotKit(model)` | one agent loop (default) |
| `BotKit(model, multi_agent=True)` | graph: planner -> worker -> reviewer (reviewer sends work back until approved) |
| `BotKit(model, review=True)` | graph: worker + reviewer correction loop |
| `BotKit(model, mode="sequential", agents=[...])` | graph: agents one after another |
| `BotKit(model, mode="parallel", agents=[...], merge="synthesizer")` | graph: agents at the same time, results merged |
| `BotKit(graph=g)` | your own graph (branching, loops) |

`agents=[...]` takes `Agent` objects, role names (`"planner"`, `"worker"`, `"reviewer"`, `"researcher"`, `"writer"`,
`"critic"`, `"synthesizer"`), `"provider:model"` strings, or dicts like `{"role": "worker", "model": "deepseek:deepseek-chat", "tools": True}`.
If you pass both `graph=` and `mode=`, the graph wins (with a warning). Nothing graph-related is used unless you ask for it.

## Multi-agent

```python
from botkit import Agent, BotKit, Graph, chain, parallel, END

planner  = Agent("openai:gpt-4o-mini",              name="planner",  system="Make a short plan.")
worker   = Agent("deepseek:deepseek-chat",          name="worker",   tools=True)
reviewer = Agent("claude:claude-sonnet-4-5",        name="reviewer")           # different model per agent

team = chain(planner, worker, reviewer, review=True)       # same as planner | worker | reviewer, plus a correction loop
print(team.chat("Write a one-page launch plan as a Word file"))

# parallel + merge
research = parallel([Agent(name="web", tools=["web"]), Agent(name="docs", tools=["files"])], merge=Agent(name="writer"))

# custom graph: nodes are agents, functions, or other graphs; edges can be conditional
g = Graph(max_steps=25)
g.add(lambda s: "code" if "bug" in s.task else "text", name="router")
g.add(coder); g.add(writer)
g.edge("router", coder, when=lambda s: s.last == "code")   # first matching condition wins
g.edge("router", writer)                                    # default
g.edge(coder, END)
```

How it works: nodes of the same level run at the same time; a node receives the task plus the outputs of the nodes
feeding it; a node with conditional edges branches (first match wins, unconditional edges are the default);
no edge to follow ends the run and its output is the answer. Loops are always bounded: `max_steps` node runs per
run (default 25), `max_visits` per node (default 10), and a node that repeats its previous output exactly stops the loop.
Reviewer nodes answer `APPROVED` or `CHANGES NEEDED: ...`; the final answer is the worker's latest result.

Which tools does each default agent get? (from the app's `tools=` / `skills=` / `shell=` flags, narrowed by role)

| Role | Tool categories | Skills |
|---|---|---|
| planner | inspect, web, compute (looks, never changes) | planning |
| worker | everything the app enabled | all |
| reviewer | inspect, compute, web, **run** (if you gave a shell). Never edits | verify-work |
| researcher | inspect, web, compute | research |
| writer | inspect, edit, documents | writing-documents |
| critic | inspect, compute | none |
| synthesizer | none (it only merges text) | none |

The reviewer verifies instead of trusting: it opens the real files, recomputes numbers and, if you gave `shell=`, runs the code or tests
and reads the output before saying `APPROVED` or `CHANGES NEEDED`. `BotKit(..., role_tools=False)` gives every agent everything instead.
Agents added as a plain model (`agents=["groq:other-model"]`) or a dict inherit the app's tools; keys in the dict (`{"role": "reviewer", "tools": [...]}`) win.

Details worth knowing:
- Agents in a graph start every visit with a clean slate (no growing context). The graph keeps the conversation (`memory=`).
- Agents that run side by side in the same workspace automatically get their own sub-folders.
- `on_error="continue"` (default for parallel) turns a failed branch into error text instead of stopping the run.
- A `Graph` is served, forked per user, streamed and closed exactly like an `Agent`.

## Watching and testing

```python
bot = BotKit(model, multi_agent=True)           # graphs always keep a trace; for a single agent use trace=True
bot.chat("...")
bot.trace.print()          # timeline: model calls, tool calls, nodes, edges, errors
bot.trace.summary()        # counts, seconds, tokens, tool errors
bot.trace.flow()           # ['planner', 'worker', 'reviewer']
import logging; logging.basicConfig(level=logging.INFO)    # the same events go to the "botkit" logger
```

```python
from botkit.testing import FakeLLM, say, call, assert_flow, assert_tools_called

planner = Agent(name="planner", llm=FakeLLM(say("plan")))          # test one agent without a model...
worker  = Agent(name="worker",  llm=FakeLLM(say("done")))
g = chain(planner, worker)
assert g.chat("go") == "done"                                       # ...or a whole workflow
assert_flow(g, ["planner", "worker"])
```

## Streaming and async

```python
for piece in bot.stream("tell me a story"): print(piece, end="")        # sync generator
async for piece in bot.astream("..."): ...                              # async generator
answer = await bot.achat("...")                                         # async call
bot.chat("...", on_token=print)                                         # or a callback
```
A graph streams only its final step (so users never see drafts as the answer).

## Providers

`provider:model` for any OpenAI-compatible API: `groq`, `openai`, `openrouter`, `gemini`, `deepseek`, `claude`
(Anthropic's OpenAI-compatible endpoint), `mistral`, `together`, `ollama`, `local` (LM Studio / vLLM). Different agents can use
different providers. Add your own: `register_provider("mycorp", "https://llm.corp/v1", "CORP_KEY")`. `botkit providers` lists them with their key variables.

## Deploying

```bash
botkit init              # writes app.py, Dockerfile, docker-compose.yml, .env.example, requirements.txt, .gitignore
cp .env.example .env     # add your key and BOTKIT_TOKEN
docker compose up --build
```
The server reads `BOTKIT_HOST`, `PORT`, `BOTKIT_TOKEN`, `BOTKIT_CORS` from the environment, shuts down cleanly on `docker stop`,
and `GET /health` returns the version. Put memory (`./data`) on a volume, or use Supabase.

## Performance notes (what changed inside)

- Several tool calls in one model reply run at the same time (never the shell).
- Context is managed by tokens: old tool outputs are shortened and the oldest turns dropped when over `max_context_tokens` (default 24000, estimated).
- `openai` is imported only when a real client is created, so `import botkit` is light.
- Failed model calls are retried with backoff; a retired/renamed model gives a clear error.

## The flags

| Flag | Default | What it switches on |
|---|---|---|
| `tools=` | none | `True` = all bundled tools · `["files", "web"]` groups · `["read_file"]` single tools · your own functions · any mix |
| `skills=` | none | `True` = bundled skills · `"research"` one by name · `"my_skills/"` your folder · `Skill(...)` · any mix |
| `memory=` | none (history in RAM only) | `"bot.db"` SQLite · `SupabaseStorage()` · your own `Storage` · `Memory(...)` |
| `mcp=` | none | command string, URL, `MCP(...)`, list, or an `mcpServers` JSON file |
| `shell=` | none | `"local"` or `"docker"` (or a dict / `Sandbox`). Turns on `run_command`. |
| `rules=` | none | string, list, or a text file; added to the prompt |
| `multi_agent=` / `mode=` / `agents=` / `graph=` / `review=` | off | multi-agent workflows (see above) |
| `trace=` | off (graphs: on) | record every model call, tool call and error |
| `max_context_tokens=` | 24000 | token budget for old turns |

## Bundled tools (17), by purpose: `botkit/tools/`

| Category | Tools | For |
|---|---|---|
| `inspect` | `list_files` `read_file` (also .docx/.xlsx/.pptx; can read a slice with `start`/`lines`) `search_files` `tree` `file_info` | look at what exists (read-only) |
| `edit` | `write_file` `edit_file` | create and change files |
| `documents` | `create_docx` `create_xlsx` `create_pptx` `create_pdf` | office/PDF files |
| `web` | `fetch_url` `web_search` (DuckDuckGo scraping, may be rate-limited) | read the internet |
| `compute` | `calculate` `current_time` | exact maths, the date |
| `run` | `run_command` `run_python` (only with `shell=`) | run programs and tests, read their output |

`tools=True` turns on all of them (the `run` tools also need `shell=`). Pick by category (`tools=["inspect", "web"]`), by tool name, or mix in your own functions. The older group names (`files`, `docs`, `util`, `shell`) still work.
Each agent's prompt lists its tools by purpose, so the model picks the right kind of tool.

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

**Supabase.** Run once in the Supabase SQL editor (`botkit sql` or `python -m botkit.memory` prints it, grants included), then set
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
grant usage on schema public to service_role;
grant all on table public.botkit_messages to service_role;
grant all on table public.botkit_facts to service_role;
grant usage, select on all sequences in schema public to service_role;
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
    kit.py          BotKit: one entry point, picks single / graph mode from flags
    agent.py        the loop (Agent)
    graph.py        multi-agent engine: Graph, chain, parallel, roles
    llm.py          providers, model client, streaming, retries
    trace.py        events, timings, logging
    testing.py      FakeLLM and assertions for tests
    cli.py          `botkit init | chat | sql | providers`
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

Image input, a `create_pdf` tool, Postgres/Neon/MongoDB backends in the box, file-storage backends (S3), scheduled tasks.
Token counts are estimates (no tokenizer). Streaming of tool-calling steps shows the model's text as it arrives.

## Tests

```bash
pip install -e ".[dev]" && pytest
```