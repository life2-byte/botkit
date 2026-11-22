"""The bundled tools (17), grouped by PURPOSE. Turn them on with Agent(tools=...):

    tools=True                          all of them
    tools=["inspect", "web"]            whole categories
    tools=["read_file", "calculate"]    single tools
    tools=["inspect", my_function]      mix with your own functions

    inspect    list_files read_file search_files tree file_info      look at what exists (read-only)
    edit       write_file edit_file                                  create and change files
    documents  create_docx create_xlsx create_pptx create_pdf       make office/PDF files   (pip install 'techtide-botkit[docs]')
    web        fetch_url web_search                                  read the internet
    compute    calculate current_time                                exact maths, the date and time
    run        run_command run_python                                run programs and tests  (only with Agent(shell=...))

Older group names still work: files (inspect+edit), docs (documents), util (compute), shell (run).
"""
from __future__ import annotations

from typing import Any

from ..tool import Tool
from . import docs, files, shell, util, web

GROUPS = {"files": files.make, "docs": docs.make, "web": web.make, "shell": shell.make, "util": util.make}

CATEGORIES: dict[str, list[str]] = {
    "inspect": ["list_files", "read_file", "search_files", "tree", "file_info"],
    "edit": ["write_file", "edit_file"],
    "documents": ["create_docx", "create_xlsx", "create_pptx", "create_pdf"],
    "web": ["fetch_url", "web_search"],
    "compute": ["calculate", "current_time"],
    "run": ["run_command", "run_python"],
}
CATEGORY_LABELS = {
    "inspect": "Inspect (look at what exists, read-only)",
    "edit": "Edit (create and change files)",
    "documents": "Documents (Word, Excel, PowerPoint, PDF)",
    "web": "Web (read the internet)",
    "compute": "Compute (exact maths, date and time)",
    "run": "Run and verify (run programs and tests, read their output)",
    "other": "Other",
}
_CATEGORY_OF = {name: cat for cat, names in CATEGORIES.items() for name in names}


def category_of(tool_name: str) -> str:
    """'inspect' | 'edit' | 'documents' | 'web' | 'compute' | 'run' | 'other' (your own and MCP tools)."""
    return _CATEGORY_OF.get(tool_name, "other")


def build(policy: Any, sandbox: Any, select: Any) -> list[Tool]:
    """Create bundled tools. `select` is True (everything) or a list of category / group / tool names."""
    by_group = {name: factory(policy, sandbox) for name, factory in GROUPS.items()}
    by_name = {t.name: t for tools in by_group.values() for t in tools}
    if select is True:
        return list(by_name.values())
    chosen: dict[str, Tool] = {}
    for item in select:
        if item in by_group:
            chosen.update({t.name: t for t in by_group[item]})
        elif item in CATEGORIES:  # a category without a shell (or without optional parts) just gives fewer tools
            chosen.update({n: by_name[n] for n in CATEGORIES[item] if n in by_name})
        elif item in by_name:
            chosen[item] = by_name[item]
        elif item in ("run_command", "run_python"):
            raise ValueError(f"'{item}' needs an environment: Agent(shell='local') or Agent(shell='docker')")
        else:
            raise ValueError(
                f"Unknown bundled tool or category {item!r}. Categories: {sorted(CATEGORIES)}. Tools: {sorted(_CATEGORY_OF)}"
            )
    return list(chosen.values())
