"""Shell tools (2): run_command, run_python. They only exist when you pick an environment with Agent(shell=...)."""
from __future__ import annotations

import uuid

from ..security import Policy
from ..tool import Tool, tool


def make(policy: Policy, sandbox=None) -> list[Tool]:
    if sandbox is None:
        return []

    @tool(sensitive=True)
    def run_command(command: str, timeout: int = 60) -> str:
        """Run a shell command in the sandbox. The working directory is the workspace. Use it to run programs and tests and read their output.

        Args:
            command: The shell command.
            timeout: Seconds before the command is killed.
        """
        code, output = sandbox.run(command, min(max(int(timeout), 1), policy.shell_timeout_max))
        return f"exit code: {code}\n{output}"

    @tool(sensitive=True)
    def run_python(code: str, timeout: int = 60) -> str:
        """Run a short Python snippet in the sandbox (workspace is the working directory) and return its output. Good for quick checks and calculations on files.

        Args:
            code: The Python code to run. Print what you want to see.
            timeout: Seconds before it is killed.
        """
        name = f".botkit_snippet_{uuid.uuid4().hex[:8]}.py"
        path = policy.resolve(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        try:
            cmd = f"if command -v python3 >/dev/null 2>&1; then python3 {name}; else python {name}; fi"
            status, output = sandbox.run(cmd, min(max(int(timeout), 1), policy.shell_timeout_max))
        finally:
            path.unlink(missing_ok=True)
        return f"exit code: {status}\n{output}"

    return [run_command, run_python]
