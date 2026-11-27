# Changelog

## 0.4.4

- **`tool_mode="prompt"` (opt-in):** for a model or proxy that ignores the API `tools` field (for example a proxy in front of a chat website's web token, which otherwise creates files in its own sandbox). botkit describes the tools in the system prompt, reads a ```tool block from the model's text, and runs the tool in YOUR workspace. A bad block gets one correction round. Default is `tool_mode="native"`, which is unchanged: normal API users are not affected.
- Works on `Agent`, `BotKit`, `LLM`, and per fallback model: `fallback=[{"model": "deepseek-web", "base_url": "...", "tool_mode": "prompt"}]`. Streaming in this mode replies whole, then emits the text (the tool block is never shown).

## 0.4.3

- A graph with `checkpoint=` now saves its state after **every finished step**, not only on failure: a long run survives a crash or restart and `resume()` continues from the last finished step.
- `fallback=` no longer requires the first model's API key to exist at start-up (it is built lazily), so a missing key on the main model falls through to the fallbacks.
- `BotKit(..., max_visits=N)`: how many times one node may run in a graph (default 10), next to `max_graph_steps`. Raise both for long tasks.
- `Graph.fork(session, trace=Trace())`: a copy with its own event log, for many runs at the same time (used by the Django app's background jobs).

## 0.4.2

- **Fallback models:** `fallback=["groq:...", "claude:..."]` on Agent / BotKit / agent dicts. When a model hits a limit, quota or outage, the same request goes to the next model. Failed models cool down and are retried later; switches appear in the trace (`fallback`).
- **Limits understood:** daily quota / no credit / "try again in 12m" is no longer retried in vain; `LLMError.kind` (`quota`, `rate_limit`, `server`, `network`, `auth`, `exhausted`, ...) and `LLMError.retry_after`.
- **Resume, agent loop:** a turn stopped by an LLM error keeps its work; `agent.resume()` continues it (also after a restart when memory is on); `agent.interrupted`. (Before, a failed turn was rolled back. Other errors still roll back.)
- **Resume, graphs:** a failed node pauses the run; `graph.resume()` re-runs only that node and continues. Finished nodes and finished parallel branches are kept. `checkpoint="run.json"` saves the run on disk.
- Terminal chat: `/resume`. Testing: `fail(kind)` for scripted failures. New export: `FallbackLLM`.

## 0.4.1
- **Every default agent has tools, scoped by role.** Tools are grouped by purpose (inspect, edit, documents, web, compute, run).
  Planner: inspect/web/compute. Worker: everything the app enabled. Reviewer: inspect/compute/web + run (when the owner gave a
  shell), never edit. `role_tools=False` gives every agent everything. New models/dicts inherit the app's tools; dict keys override.
- **Reviewer verifies, does not trust:** it opens the real files, recomputes numbers and runs code/tests (if a shell exists) before judging.
- Skills by role (planner: planning, reviewer: verify-work, worker: all).
- The system prompt lists each agent's tools by purpose.
- **New tools (17 total):** `tree`, `file_info`, `current_time`, `run_python` (needs shell), `create_pdf` (`reportlab`, in the `docs` extra);
  `read_file` can read a slice (`start`, `lines`) so agents need not read whole files.
- Fix: a review loop no longer stops early when the worker repeats the same sentence after changing files (progress = new text OR tool use).
- Fix: servers that cannot stream (HTTP 501) fall back to normal replies, 501 is never retried; models that try to call tools they
  were not given ("Tool choice is none") are told they have no tools and the error is retried.
- Token counts are only recorded when the server reports usage.

## 0.4.0
- **Multi-agent:** `Graph` engine (nodes = agents / functions / graphs, conditional edges, loops with guards), `chain` (`a | b | c`),
  `parallel`, reviewer correction loop, roles (planner, worker, reviewer, researcher, writer, critic, synthesizer).
- **`BotKit(...)`** one entry point; flags `multi_agent`, `mode`, `agents`, `graph`, `review`, `merge`. `Agent(...)` unchanged.
- **Providers:** DeepSeek, Claude (OpenAI-compatible endpoint), Mistral, Together, local servers; `register_provider`; different model per agent.
- **Loop:** streaming (`stream`, `astream`, `on_token`), async (`achat`), parallel tool calls, token-based context trimming,
  old tool outputs shortened, lazy `openai` import, clearer errors for retired models.
- **Observability:** `Trace` (model/tool/node/edge/error events, summary, JSON) + `logging` ("botkit" logger).
- **Testing:** `botkit.testing` (`FakeLLM`, `say`, `call`, `assert_flow`, `assert_tools_called`).
- **Deployment:** `botkit init` (Dockerfile, compose, app.py, .env.example), server settings from the environment, `/health` version, clean SIGTERM.
- **Supabase:** `schema_sql()` includes the `service_role` grants; clear permission-denied message; new command `python -m botkit.memory` / `botkit sql`.
- MCP servers shared by several agents start once (thread-safe start).

## 0.3.0
First public release.
