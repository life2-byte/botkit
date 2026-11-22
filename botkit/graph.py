"""Multi-agent workflows. Everything is one engine: a Graph of nodes (agents, functions, other graphs)
joined by edges. The three modes are just shortcuts that build a graph for you:

    chain(planner, worker, reviewer)              # sequential:  planner | worker | reviewer
    parallel([a, b, c], merge=summarizer)         # parallel + merge
    g = Graph(); g.add(...); g.edge(...)           # custom: branching, loops, anything

Rules of the engine:
- Nodes of the same "level" (fan-out) run at the same time; the next level starts when all of them are done.
- A node with conditional edges branches: the first edge whose `when(state)` is true wins; unconditional edges
  are the default. A node with only unconditional edges fans out to all of them.
- A node with no edge to follow (or an edge to "end") finishes the graph; its output is the answer.
- Loops are allowed but always bounded: `max_steps` node runs per graph run, `max_visits` per node,
  and a node that repeats its previous output exactly (without using any tool) is treated as "no progress" and stops the loop.
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .agent import Agent
from .chatio import ChatMixin
from .errors import GraphError
from .memory import Memory, SQLiteStorage, Storage
from .trace import Trace, log

END = "end"

# --------------------------------------------------------------------------- roles (ready-made agent personalities)
# Each role: (system prompt, access). access = which tool CATEGORIES and which bundled SKILLS the role gets from the
# app's `tools=` / `skills=` / `shell=` flags. "all" = everything the app enabled. None = nothing.
# (Turn this narrowing off with BotKit(..., role_tools=False): then every agent gets everything.)
ROLES: dict[str, tuple[str, dict[str, Any]]] = {
    "planner": (
        "You are the Planner. Turn the task into a short, concrete, numbered plan another agent can follow. "
        "If you have tools, use them only to LOOK at what already exists (files, the web) so the plan fits reality; "
        "never create or change anything. Do not do the work yourself. Say what a good result looks like.",
        {"tools": ["inspect", "web", "compute"], "skills": ["planning"]}),
    "worker": (
        "You are the Worker. Do the task fully and carefully, following any plan you are given. Use your tools to do "
        "the work AND to check it (open the file you made, run the code, recompute numbers). If you receive reviewer "
        "feedback, fix every point. Reply with the finished result and name any files you created.",
        {"tools": "all", "skills": "all"}),
    "reviewer": (
        "You are the Reviewer. Verify, do not trust: judge the ACTUAL result, not the worker's description of it. "
        "If you have tools: look at what was produced (list or tree first, then read only the files that matter; for big "
        "files use search_files or read_file with start/lines), recompute numbers with calculate, and if you can run "
        "commands, run the code or tests and read the output before judging. You never change anything yourself. "
        "Reply with APPROVED on the first line if it fully meets the task. Otherwise start with CHANGES NEEDED and list "
        "the concrete fixes, most important first.",
        {"tools": ["inspect", "compute", "run", "web"], "skills": ["verify-work"]}),
    "researcher": (
        "You are the Researcher. Find accurate, current information for the task, say where it came from, "
        "and separate facts from guesses.",
        {"tools": ["inspect", "web", "compute"], "skills": ["research"]}),
    "writer": (
        "You are the Writer. Produce clear, well-structured text for the task, in the right tone.",
        {"tools": ["inspect", "edit", "documents"], "skills": ["writing-documents"]}),
    "critic": (
        "You are the Critic. Find the weakest points, risks and mistakes. Be specific and brief.",
        {"tools": ["inspect", "compute"], "skills": None}),
    "synthesizer": (
        "You are the Synthesizer. Combine the given outputs into one coherent answer. Keep what agrees, "
        "point out real disagreements, drop repetition.",
        {"tools": None, "skills": None}),
}
REVIEW_HINT = (
    "Review the latest result above against the task. If you have tools, check the actual result (files, numbers, "
    "program output), not just the description. Reply with APPROVED on the first line if it fully meets the task. "
    "Otherwise start with CHANGES NEEDED and list concrete fixes."
)


class State:
    """What every node can read (and function nodes can write). Passed to `when=` conditions and `input=` callables."""

    def __init__(self, task: str, context: str = "") -> None:
        self.task = task                      # the user's request
        self.context = context                # earlier conversation + remembered facts (if memory is on)
        self.outputs: dict[str, str] = {}     # latest output of every node that has run: outputs["worker"]
        self.last = ""                        # output of the node that ran most recently
        self.shared: dict[str, Any] = {}      # free space for function nodes to pass data around
        self.visits: dict[str, int] = {}      # how many times each node has run
        self.approved: bool | None = None     # set by a reviewer node: True = APPROVED
        self.final = ""                       # the graph's answer (filled in when the run ends)
        self.worked: dict[str, bool] = {}     # did the node's latest visit use tools? (changed files count as progress)

    def __repr__(self) -> str:
        return f"State(task={self.task[:40]!r}, nodes={list(self.outputs)}, approved={self.approved})"


@dataclasses.dataclass
class _Node:
    name: str
    target: Any                    # Agent | Graph | callable(state)
    kind: str                      # "agent" | "graph" | "function"
    input: Any = None              # None | "template {task} {last} {name}" | callable(state) -> str
    reviewer: bool = False
    delivers: str | None = None    # when this node ends the graph, answer with THIS node's output instead


@dataclasses.dataclass
class _Edge:
    src: str
    dst: str
    when: Callable[[State], bool] | None = None


class _Safe(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _ident(name: str) -> str:
    return re.sub(r"\W", "_", name)


class Graph(ChatMixin):
    """A workflow of agents. Build it, then use it exactly like an Agent: `chat`, `stream`, `run`, `serve`.

        g = Graph()
        g.add(planner); g.add(worker); g.add(reviewer)
        g.edge(planner, worker)
        g.edge(worker, reviewer)
        g.edge(reviewer, worker, when=lambda s: not s.approved)   # send it back
        g.edge(reviewer, END)                                      # approved: done
        print(g.chat("write a launch plan"))

    Args:
        start: name/object (or a list for a parallel start) of the first node(s). Default: the first node added.
        max_steps: most node runs in one graph run (loop guard). max_visits: most runs of any one node.
        on_error: "raise" (a failing node stops the run with GraphError) or "continue" (its output becomes the
            error text and the graph goes on; used by parallel mode so one failed branch does not lose the rest).
        output: name of the node whose output is the final answer (default: the node(s) that end the run).
        memory / session: conversation memory for the graph as a whole (history + facts), like Agent's.
        trace: True/Trace records the full flow: node starts and ends, every edge taken, model and tool calls.
    """

    def __init__(
        self,
        *,
        start: Any = None,
        max_steps: int = 25,
        max_visits: int = 10,
        on_error: str = "raise",
        output: Any = None,
        memory: Memory | Storage | str | Path | bool | None = None,
        session: str = "default",
        max_history: int = 20,
        trace: Trace | bool | None = True,
        name: str = "graph",
        description: str = "",
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        if on_error not in ("raise", "continue"):
            raise ValueError('on_error must be "raise" or "continue"')
        self.name, self.description = name, description
        self.max_steps, self.max_visits, self.on_error = max_steps, max_visits, on_error
        self._start: list[str] = []
        self._output = output
        self.on_event = on_event
        self.session = session
        self.max_history = max_history
        self.trace: Trace = trace if isinstance(trace, Trace) else Trace()
        self.nodes: dict[str, _Node] = {}
        self.edges: list[_Edge] = []
        self._ids: dict[int, str] = {}
        self._chain = False
        self._lock = threading.RLock()
        self._prepared = False
        self._closed = False
        self._params = dict(max_steps=max_steps, max_visits=max_visits, on_error=on_error, max_history=max_history)
        self.memory: Memory | None = Agent._resolve_memory(memory, session, max_history)
        self.messages: list[dict] = self.memory.load() if self.memory else []
        if start is not None:
            self._start = [self._name_of(s, create=False) for s in (start if isinstance(start, (list, tuple)) else [start])]
        self._start_given = start is not None

    # ---- building ---------------------------------------------------------------
    def add(self, target: Any, name: str | None = None, *, input: Any = None, reviewer: bool = False) -> str:
        """Add a node: an Agent, another Graph, or a function `fn(state) -> str`. Returns its name."""
        if id(target) in self._ids and name is None:
            return self._ids[id(target)]
        if isinstance(target, Agent):
            kind, base = "agent", target.name
        elif isinstance(target, Graph):
            kind, base = "graph", target.name
        elif callable(target):
            kind, base = "function", getattr(target, "__name__", "function")
        else:
            raise GraphError(f"cannot use {type(target).__name__} as a node; use an Agent, a Graph or a function")
        name = name or base
        unique, i = name, 2
        while unique in self.nodes:
            unique, i = f"{name}_{i}", i + 1
        if unique == END:
            raise GraphError(f"'{END}' is reserved")
        self.nodes[unique] = _Node(unique, target, kind, input, reviewer)
        self._ids.setdefault(id(target), unique)
        if not self._start_given:
            self._start = self._start or [unique]
        self._prepared = False
        return unique

    def _name_of(self, ref: Any, create: bool = True) -> str:
        if isinstance(ref, str):
            if ref == END or ref in self.nodes or not create:
                return ref
            raise GraphError(f"unknown node '{ref}'. Nodes: {', '.join(self.nodes) or 'none'}")
        if id(ref) in self._ids:
            return self._ids[id(ref)]
        return self.add(ref) if create else getattr(ref, "name", str(ref))

    def edge(self, src: Any, dst: Any, when: Callable[[State], bool] | None = None) -> "Graph":
        """Connect nodes. `dst` may be a node, a list of nodes (run together), or "end". `when(state)` makes it conditional."""
        s = self._name_of(src)
        for d in dst if isinstance(dst, (list, tuple)) else [dst]:
            self.edges.append(_Edge(s, d if d == END else self._name_of(d), when))
        self._prepared = False
        return self

    def review(self, worker: Any, reviewer: Any, max_rounds: int = 3) -> "Graph":
        """Correction loop: worker -> reviewer; if the reviewer does not say APPROVED the worker tries again
        (at most `max_rounds` reviews). The answer is always the worker's latest result."""
        w = self._name_of(worker)
        r = self._name_of(reviewer)
        node = self.nodes[r]
        node.reviewer, node.delivers = True, w
        if node.input is None:
            node.input = lambda s, _w=w: _review_prompt(s, _w)
        self.edge(w, r)
        self.edge(r, w, when=lambda s, _r=r: s.approved is False and s.visits.get(_r, 0) < max_rounds)
        self.edge(r, END)
        return self

    @property
    def start(self) -> list[str]:
        return list(self._start)

    @start.setter
    def start(self, value: Any) -> None:
        self._start = [self._name_of(v) for v in (value if isinstance(value, (list, tuple)) else [value])]
        self._start_given = True

    @property
    def agents(self) -> list[Agent]:
        return [n.target for n in self.nodes.values() if n.kind == "agent"]

    def validate(self) -> None:
        if not self.nodes:
            raise GraphError("the graph has no nodes")
        if not self._start:
            raise GraphError("the graph has no start node")
        for s in self._start:
            if s not in self.nodes:
                raise GraphError(f"start node '{s}' does not exist")
        for e in self.edges:
            if e.src not in self.nodes or (e.dst != END and e.dst not in self.nodes):
                raise GraphError(f"edge {e.src} -> {e.dst} points to a node that does not exist")
        if self._output is not None and self._name_of(self._output) not in self.nodes:
            raise GraphError("output= is not a node of this graph")
        reachable, todo = set(), list(self._start)
        while todo:
            n = todo.pop()
            if n in reachable or n == END:
                continue
            reachable.add(n)
            todo.extend(e.dst for e in self.edges if e.src == n)
        lost = set(self.nodes) - reachable
        if lost:
            raise GraphError(f"these nodes can never run (no path from the start): {', '.join(sorted(lost))}")

    def describe(self) -> str:
        """Readable outline of the workflow, e.g. for logs or docs."""
        lines = [f"start: {', '.join(self._start)}"]
        for e in self.edges:
            lines.append(f"{e.src} --{'(if ...)' if e.when else ''}--> {e.dst}")
        return "\n".join(lines)

    # ---- running ----------------------------------------------------------------
    def chat(self, user_input: str, on_token: Callable[[str], None] | None = None) -> str:
        """One user turn through the whole workflow. Returns the final answer."""
        with self._lock:
            context = self._context()
            answer = self.run_state(user_input, context=context, on_token=on_token).final
            turn = [{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}]
            self.messages.extend(turn)
            if self.memory:
                self.memory.extend(turn)
            return answer

    def invoke(self, task: str) -> str:
        """Run the workflow once, without conversation memory."""
        return self.run_state(task).final

    def run_state(self, task: str, *, context: str = "", on_token: Callable[[str], None] | None = None) -> State:
        """Like invoke(), but returns the full State (outputs of every node, visits, approved, `final`)."""
        self.validate()
        self._prepare()
        state = State(task, context)
        t0 = time.monotonic()
        self._emit("graph_start", node=self.name, task=task[:200])
        try:
            final = self._execute(state, on_token)
        except BaseException as e:
            self._emit("error", node=self.name, error=f"{type(e).__name__}: {e}")
            raise
        state.final = final
        self._emit("graph_end", node=self.name, seconds=round(time.monotonic() - t0, 3), chars=len(final))
        return state

    def _context(self) -> str:
        parts = []
        if self.memory:
            facts = self.memory.list_facts(20)
            if facts:
                parts.append("Remembered about the user:\n" + "\n".join(f"- {f.text}" for f in facts))
        recent = self.messages[-self.max_history :]
        if recent:
            text = "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {str(m['content'])[:1500]}" for m in recent)
            parts.append("Conversation so far:\n" + text[-6000:])
        return "\n\n".join(parts)

    def _execute(self, state: State, on_token: Callable[[str], None] | None) -> str:
        frontier, steps, finals = list(self._start), 0, []
        while frontier:
            level: list[str] = []
            for n in dict.fromkeys(frontier):
                if state.visits.get(n, 0) >= self.max_visits:
                    self._emit("loop_limit", node=n, max_visits=self.max_visits)
                    continue
                level.append(n)
            if not level:
                break
            if steps + len(level) > self.max_steps:
                self._emit("step_limit", node=self.name, max_steps=self.max_steps)
                break
            steps += len(level)
            results = self._run_level(level, state, on_token if len(level) == 1 else None)
            frontier, finals = [], []
            for name, text, previous in results:
                state.outputs[name] = text
                state.last = text
                state.visits[name] = state.visits.get(name, 0) + 1
                node = self.nodes[name]
                if node.reviewer:
                    state.approved = text.strip().upper().startswith("APPROVED")
                if previous is not None and previous == text and state.visits[name] > 1 and not state.worked.get(name):
                    self._emit("no_progress", node=name)
                    finals.append(name)
                    continue
                targets = self._next(name, state)
                if not targets:
                    finals.append(name)
                for t in targets:
                    self._emit("edge", node=name, to=t)
                frontier.extend(targets)
        return self._final(state, finals)

    def _next(self, name: str, state: State) -> list[str]:
        out = [e for e in self.edges if e.src == name]
        for e in out:  # branching: the first matching condition wins
            if e.when is not None:
                try:
                    if e.when(state):
                        return [] if e.dst == END else [e.dst]
                except Exception as err:  # noqa: BLE001 - a broken condition must not crash the run
                    raise GraphError(f"condition on edge {e.src} -> {e.dst} failed: {err}") from err
        defaults = [e.dst for e in out if e.when is None]
        return [d for d in defaults if d != END]

    def _final(self, state: State, finals: list[str]) -> str:
        if self._output is not None:
            return state.outputs.get(self._name_of(self._output), "")
        if not finals:  # stopped by a limit
            finals = [list(state.outputs)[-1]] if state.outputs else []
        texts = []
        for n in finals:
            node = self.nodes[n]
            src = node.delivers if node.delivers and node.delivers in state.outputs else n
            texts.append((src, state.outputs.get(src, "")))
        texts = list(dict.fromkeys(texts))
        if len(texts) == 1:
            return texts[0][1]
        return "\n\n".join(f"## {n}\n{t}" for n, t in texts)

    def _run_level(self, level: list[str], state: State, on_token: Callable[[str], None] | None) -> list[tuple]:
        def one(name: str) -> tuple[str, str, str | None]:
            previous = state.outputs.get(name)
            return name, self._run_node(self.nodes[name], state, on_token), previous

        if len(level) == 1:
            return [one(level[0])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(level), 8)) as pool:
            futures = [pool.submit(one, n) for n in level]
            return [f.result() for f in futures]  # in level order; failures were already handled in _run_node

    def _prompt(self, node: _Node, state: State) -> str:
        if callable(node.input):
            return str(node.input(state))
        if isinstance(node.input, str):
            values = _Safe({_ident(k): v for k, v in state.outputs.items()})
            values.update(task=state.task, last=state.last, context=state.context)
            return node.input.format_map(values)
        parts = []
        if state.context:
            parts.append(f"Context:\n{state.context}")
        parts.append(f"Task:\n{state.task}")
        for e in self.edges:  # outputs of the nodes that feed this one
            if e.dst == node.name and e.src in state.outputs and e.src != node.name:
                parts.append(f"Output from {e.src}:\n{state.outputs[e.src]}")
        if node.name in state.outputs:
            parts.append(f"Your previous attempt:\n{state.outputs[node.name]}")
        return "\n\n".join(parts)

    def _run_node(self, node: _Node, state: State, on_token: Callable[[str], None] | None) -> str:
        t0 = time.monotonic()
        self._emit("node_start", node=node.name, node_kind=node.kind, visit=state.visits.get(node.name, 0) + 1)
        stream = on_token if self._ends_here(node) else None
        try:
            if node.kind == "agent":
                before = node.target.tool_calls
                text = node.target.ask(self._prompt(node, state), on_token=stream)
                state.worked[node.name] = node.target.tool_calls > before
            elif node.kind == "graph":
                text = node.target.run_state(self._prompt(node, state)).final  # type: ignore[attr-defined]
            else:
                res = node.target(state)
                if isinstance(res, dict):
                    state.shared.update(res)
                    res = json.dumps(res, ensure_ascii=False, default=str)
                text = "" if res is None else str(res)
        except Exception as e:  # noqa: BLE001
            self._emit("error", node=node.name, error=f"{type(e).__name__}: {e}")
            if self.on_error == "raise":
                raise GraphError(f"node '{node.name}' failed: {e}") from e
            text = f"Error: {type(e).__name__}: {e}"
        self._emit("node_end", node=node.name, seconds=round(time.monotonic() - t0, 3), chars=len(text))
        return text

    def _ends_here(self, node: _Node) -> bool:
        """Stream a node's text only when it is certainly the last step (so users never see drafts as the answer)."""
        out = [e for e in self.edges if e.src == node.name]
        return not node.reviewer and all(e.dst == END for e in out) and node.delivers is None and len(self._start) == 1

    def _emit(self, kind: str, **data: Any) -> None:
        self.trace.add(kind, **data)
        log.debug("[graph] %s %s", kind, data)
        if self.on_event:
            try:
                self.on_event(kind, data)
            except Exception:  # noqa: BLE001
                pass

    def _prepare(self) -> None:
        """Wire agents to the graph's trace and give agents that run side by side their own file folders."""
        if self._prepared:
            return
        for a in self.agents:
            if a.trace is None:
                a.trace = self.trace
        groups: list[list[str]] = []
        if len(self._start) > 1:
            groups.append(list(self._start))
        for src in {e.src for e in self.edges}:
            targets = [e.dst for e in self.edges if e.src == src and e.when is None and e.dst != END]
            if len(targets) > 1:
                groups.append(targets)
        for group in groups:
            agents = [self.nodes[n] for n in group if self.nodes[n].kind == "agent"]
            by_dir: dict[Path, list[_Node]] = {}
            for n in agents:
                by_dir.setdefault(n.target.policy.workdir, []).append(n)
            for base, members in by_dir.items():
                if len(members) > 1:  # same folder + running together: separate sub-folders prevent clashes
                    for n in members:
                        n.target.policy.workdir = base / _ident(n.name)
        self._prepared = True

    # ---- the Agent-like surface (so servers, apps and BotKit treat both the same) ------------
    @property
    def policy(self) -> Any:
        first = self.nodes[self._start[0]]
        for n in [first, *self.nodes.values()]:
            if n.kind == "agent":
                return n.target.policy
        raise GraphError("this graph has no agent nodes, so it has no workspace")

    @property
    def tool_names(self) -> list[str]:
        return sorted({t for a in self.agents for t in a.tool_names})

    def tools_info(self) -> dict[str, list[dict]]:
        return {n.name: n.target.tools_info() for n in self.nodes.values() if n.kind == "agent"}

    def reset(self) -> None:
        with self._lock:
            self.messages = []
            if self.memory:
                self.memory.clear_history()

    def fork(self, session: str) -> "Graph":
        """A copy of this workflow for another user: own conversation, own agent histories and workspace folders."""
        g = Graph(
            start=None, output=self._output, session=session, trace=self.trace, name=self.name,
            description=self.description, on_event=self.on_event, memory=None, **self._params,
        )
        g.memory = self.memory.for_session(session) if self.memory else None
        g.messages = g.memory.load() if g.memory else []
        done: dict[int, Any] = {}
        for n in self.nodes.values():
            if n.kind in ("agent", "graph"):
                forked = done.setdefault(id(n.target), n.target.fork(session))
            else:
                forked = n.target
            g.nodes[n.name] = dataclasses.replace(n, target=forked)
            g._ids.setdefault(id(forked), n.name)
        g.edges = list(self.edges)
        g._start, g._start_given = list(self._start), True
        return g

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for n in self.nodes.values():
            if n.kind in ("agent", "graph"):
                try:
                    n.target.close()
                except Exception:  # noqa: BLE001
                    pass

    def __or__(self, other: Any) -> "Graph":
        return chain(self, other)

    def __repr__(self) -> str:
        return f"Graph(nodes={list(self.nodes)}, start={self._start})"


def _review_prompt(state: State, worker: str) -> str:
    return (
        f"Task:\n{state.task}\n\nLatest result from {worker}:\n{state.outputs.get(worker, '')}\n\n{REVIEW_HINT}"
    )


# --------------------------------------------------------------------------- the two shortcut builders
def chain(*parts: Any, review: bool = False, max_rounds: int = 3, **graph_options: Any) -> Graph:
    """Sequential workflow: each part receives the task plus what the previous part produced.
    `planner | worker | reviewer` is the same thing. With review=True the last part is a reviewer that sends
    the one before it back to work until it approves (at most `max_rounds` times)."""
    if parts and isinstance(parts[0], Graph) and parts[0]._chain and not graph_options and not review:
        g = parts[0]
        parts = parts[1:]
    else:
        g = Graph(**graph_options)
        g._chain = True
    prev = list(g.nodes)[-1] if g.nodes else None
    for p in parts:
        name = g.add(p)
        if prev:
            g.edge(prev, name)
        prev = name
    if review:
        names = list(g.nodes)
        if len(names) < 2:
            raise GraphError("review=True needs at least two parts: a worker and a reviewer")
        g.edges = [e for e in g.edges if not (e.src == names[-2] and e.dst == names[-1])]
        g.review(names[-2], names[-1], max_rounds)
    return g


def parallel(agents: list[Any], merge: Any = None, **graph_options: Any) -> Graph:
    """Run all agents at the same time on the same task, then merge their results.
    `merge` is an Agent (e.g. a synthesizer), a function(state) -> str, or None to just join the outputs."""
    if len(agents) < 2:
        raise GraphError("parallel mode needs at least two agents")
    graph_options.setdefault("on_error", "continue")
    g = Graph(**graph_options)
    names = [g.add(a) for a in agents]
    g.start = names

    def join(state: State) -> str:
        return "\n\n".join(f"## {n}\n{state.outputs.get(n, '')}" for n in names)

    m = g.add(merge if merge is not None else join, name="merge" if not isinstance(merge, Agent) else None)
    for n in names:
        g.edge(n, m)
    return g
