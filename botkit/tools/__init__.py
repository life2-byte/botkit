"""The bundled tools (12), in five groups. Turn them on with Agent(tools=...):

    tools=True                          all of them
    tools=["files", "web"]              whole groups
    tools=["read_file", "calculate"]    single tools
    tools=["files", my_function]        mix with your own functions

    files  list_files read_file write_file edit_file search_files
    docs   create_docx create_xlsx create_pptx          (pip install 'botkit[docs]')
    web    fetch_url web_search
    shell  run_command                                 (only exists with Agent(shell=...))
    util   calculate
"""
from __future__ import annotations

from typing import Any

from ..tool import Tool
from . import docs, files, shell, util, web

GROUPS = {"files": files.make, "docs": docs.make, "web": web.make, "shell": shell.make, "util": util.make}


def build(policy: Any, sandbox: Any, select: Any) -> list[Tool]:
    """Create bundled tools. `select` is True (everything) or a list of group / tool names."""
    by_group = {name: factory(policy, sandbox) for name, factory in GROUPS.items()}
    if select is True:
        return [t for tools in by_group.values() for t in tools]
    by_name = {t.name: t for tools in by_group.values() for t in tools}
    chosen: dict[str, Tool] = {}
    for item in select:
        if item in by_group:
            chosen.update({t.name: t for t in by_group[item]})
        elif item in by_name:
            chosen[item] = by_name[item]
        elif item == "run_command":
            raise ValueError("'run_command' needs an environment: Agent(shell='local') or Agent(shell='docker')")
        else:
            raise ValueError(
                f"Unknown bundled tool or group {item!r}. Groups: {sorted(GROUPS)}. Tools: {sorted(by_name)}"
            )
    return list(chosen.values())
