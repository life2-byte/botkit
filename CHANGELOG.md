# Changelog

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
