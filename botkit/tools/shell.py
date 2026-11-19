"""Shell tool (1): run_command. Only exists when you pick an environment with Agent(shell=...)."""
from __future__ import annotations

from ..security import Policy
from ..tool import Tool, tool


def make(policy: Policy, sandbox=None) -> list[Tool]:
    if sandbox is None:
        return []

    @tool(sensitive=True)
    def run_command(command: str, timeout: int = 60) -> str:
        """Run a shell command in the sandbox. The working directory is the workspace.

        Args:
            command: The shell command.
            timeout: Seconds before the command is killed.
        """
        code, output = sandbox.run(command, min(max(int(timeout), 1), policy.shell_timeout_max))
        return f"exit code: {code}\n{output}"

    return [run_command]
