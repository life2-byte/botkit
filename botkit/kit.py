"""BotKit: ONE entry point. You switch modes with flags; the library builds what you asked for.

    BotKit("groq:openai/gpt-oss-120b")                                  # a single agent loop (the default)
    BotKit(model, tools=True, memory="bot.db")                          # single agent + features
    BotKit(model, multi_agent=True)                                     # planner -> worker -> reviewer team
    BotKit(model, mode="sequential", agents=["planner", "worker"])      # your own chain
    BotKit(model, mode="parallel", agents=[...], merge="synthesizer")   # all at once, then merged
    BotKit(model, review=True)                                          # worker + reviewer correction loop
    BotKit(graph=my_graph)                                              # your own graph (loops, branching)
    BotKit(model, fallback=["groq:openai/gpt-oss-120b"])                # switch model when this one hits a limit
    team = BotKit(model, multi_agent=True, checkpoint="run.json")       # a failed run can be continued: team.resume()

It returns an `Agent` for single mode and a `Graph` for everything else. Both are used the same way:
chat(), stream(), run(), serve(), fork(), close(). Simple outside, complicated inside.
"""
from __future__ import annotations

import warnings
from typing import Any

from .agent import DEFAULT_MODEL, Agent
from .errors import GraphError
from .graph import ROLES, Graph, chain, parallel
from .llm import PROVIDERS
from .mcp import normalize
from .tools import CATEGORIES, build as _build_tools, category_of

_DEFAULT_TEAM = ["planner", "worker", "reviewer"]


def _is_model(text: str) -> bool:
    head, sep, _ = text.partition(":")
    return bool(sep) and head.lower() in PROVIDERS


def _selected_tool_names(tools: Any) -> list[str]:
    """Bundled tool names the app's `tools=` flag switches on (shell tools are decided separately)."""
    if tools in (None, False):
        return []
    items = tools if isinstance(tools, (list, tuple)) else [tools]
    chosen: list[str] = []
    for it in items:
        if it is True:
            chosen += [n for names in CATEGORIES.values() for n in names]
        elif isinstance(it, str):
            from .security import Policy

            try:  # resolve groups / categories / names with a throw-away policy (no files are touched)
                from pathlib import Path

                chosen += [t.name for t in _build_tools(Policy(workdir=Path(".").resolve()), None, [it])]
            except ValueError:
                pass
    return list(dict.fromkeys(chosen))


def _apply_role_access(opts: dict[str, Any], access: dict[str, Any]) -> None:
    """Narrow the app-level tools/skills/shell flags to what this role is meant to have."""
    cats, skills = access.get("tools"), access.get("skills")
    app_tools, app_skills, app_shell = opts.get("tools"), opts.get("skills"), opts.get("shell")
    if cats != "all":
        wanted = set(cats or [])
        names = [n for n in _selected_tool_names(app_tools) if category_of(n) in wanted and category_of(n) != "run"]
        opts["tools"] = names or None
        if "run" not in wanted:
            opts["shell"] = None            # no shell for roles that do not verify/run things
    if skills != "all":
        opts["skills"] = list(skills) if skills and app_skills is True else None


def _make_agent(spec: Any, index: int, model: str, shared: dict[str, Any], role_tools: bool = True) -> Agent:
    """An Agent from: an Agent | a role name ("planner") | "provider:model" | a dict of Agent options."""
    if isinstance(spec, Agent):
        return spec
    opts = dict(shared)
    if isinstance(spec, str):
        spec = {"model": spec} if _is_model(spec) else {"role": spec}
    if not isinstance(spec, dict):
        raise GraphError(f"cannot build an agent from {type(spec).__name__}; use an Agent, a role name or a dict")
    spec = dict(spec)
    role = spec.pop("role", None)
    name = spec.pop("name", None) or role or f"agent{index}"
    if role:
        system, access = ROLES.get(
            role, (f"You are the {role}. Do your part of the task well and stay on topic.", {"tools": "all", "skills": "all"}))
        opts.setdefault("system", system)
        if role_tools:
            _apply_role_access(opts, access)
    opts.update(spec)                       # anything you write in the dict wins over the role defaults
    opts.setdefault("model", model)
    opts["name"] = name
    return Agent(**opts)


class BotKit:
    """See the module docstring. `BotKit(...)` returns an Agent (single mode) or a Graph (multi-agent modes)."""

    def __new__(
        cls,
        model: str = DEFAULT_MODEL,
        *,
        mode: str | None = None,
        agents: list[Any] | None = None,
        graph: Graph | None = None,
        multi_agent: bool = False,
        review: bool = False,
        merge: Any = None,
        max_rounds: int = 3,
        max_graph_steps: int = 25,
        max_visits: int = 10,
        role_tools: bool = True,
        memory: Any = None,
        session: str = "default",
        trace: Any = None,
        checkpoint: Any = None,
        **agent_options: Any,
    ) -> Agent | Graph:
        if mode not in (None, "single", "sequential", "parallel", "custom"):
            raise GraphError('mode must be "single", "sequential", "parallel" or "custom"')

        # ---- which mode? one place decides, user flags only ask for it
        if graph is not None:
            if mode not in (None, "custom"):
                warnings.warn(f'graph= was given, so mode="{mode}" is ignored (custom graph wins).', stacklevel=2)
            if memory is not None and graph.memory is None:
                graph.memory = Agent._resolve_memory(memory, session, graph.max_history)
                graph.messages = graph.memory.load() if graph.memory else []
            if checkpoint and graph.checkpoint is None:
                from pathlib import Path

                graph.checkpoint = Path(checkpoint).expanduser()
            return graph
        if mode == "custom":
            raise GraphError('mode="custom" needs graph=... (build one with botkit.Graph)')
        wants_team = multi_agent or review or bool(agents) or mode in ("sequential", "parallel")
        if mode == "single" or not wants_team:
            first = (agents or [None])[0]
            if isinstance(first, Agent):
                return first
            opts = dict(agent_options, memory=memory, session=session, trace=trace)
            return _make_agent(first or {"name": "agent"}, 1, model, opts, role_tools)

        # ---- multi-agent: built agents share defaults, and MCP servers start only once
        shared = dict(agent_options)
        if shared.get("mcp"):
            shared["mcp"] = normalize(shared["mcp"])
        gopts = dict(max_steps=max_graph_steps, max_visits=max_visits, memory=memory, session=session, trace=trace if trace is not None else True,
                     checkpoint=checkpoint)

        if mode == "parallel":
            if not agents or len(agents) < 2:
                raise GraphError('mode="parallel" needs agents=[...] with at least two agents')
            if review:
                raise GraphError("review=True works with sequential mode (worker -> reviewer), not parallel")
            built = [_make_agent(a, i, model, shared, role_tools) for i, a in enumerate(agents, 1)]
            if merge is None or merge == "synthesizer":
                merge = _make_agent("synthesizer", 0, model, shared, role_tools)
            elif merge == "concat":
                merge = None
            elif isinstance(merge, (str, dict)):
                merge = _make_agent(merge, 0, model, shared, role_tools)
            return parallel(built, merge, **gopts)

        # sequential (also: multi_agent=True, review=True, or agents=[...] without a mode)
        team = agents or (["worker", "reviewer"] if review and not multi_agent else _DEFAULT_TEAM)
        built = [_make_agent(a, i, model, shared, role_tools) for i, a in enumerate(team, 1)]
        loop = review or (agents is None and multi_agent)  # the default team always reviews its own work
        if loop and len(built) < 2:
            raise GraphError("review=True needs at least a worker and a reviewer")
        return chain(*built, review=loop, max_rounds=max_rounds, **gopts)
