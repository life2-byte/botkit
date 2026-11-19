"""Where shell commands run. You choose: Agent(shell="local" | "docker" | Sandbox(...)). Default: no shell."""
from __future__ import annotations

import copy
import os
import signal
import subprocess
import uuid
from typing import Any

from .errors import BotError
from .security import Policy


def _clip(data: bytes, limit: int) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) > limit:
        text = text[:limit] + f"\n...[truncated {len(text) - limit} characters]"
    return text


class Sandbox:
    policy: Policy

    def run(self, command: str, timeout: float) -> tuple[int, str]:  # pragma: no cover - interface
        raise NotImplementedError

    def rebind(self, policy: Policy) -> "Sandbox":
        clone = copy.copy(self)
        clone.policy = policy
        return clone


class LocalSandbox(Sandbox):
    """Runs commands on this machine, in the workspace folder, with a scrubbed environment.

    NOT isolated: a determined user can still do damage the filter does not foresee.
    Use it for your own machine / trusted users; use DockerSandbox for anyone else.
    """

    def __init__(self, policy: Policy | None = None, env: dict[str, str] | None = None) -> None:
        self.policy = policy  # type: ignore[assignment]
        self.env = env

    def run(self, command: str, timeout: float) -> tuple[int, str]:
        self.policy.check_command(command, confine=self.policy.confine_shell)
        root = self.policy.workdir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            ["/bin/sh", "-c", command],
            cwd=root,
            env=self.policy.safe_env(self.env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return 124, _clip(out, self.policy.max_output) + f"\n[killed: timed out after {timeout:g}s]"
        return proc.returncode, _clip(out, self.policy.max_output)


class DockerSandbox(Sandbox):
    """Runs each command in a throwaway container: only the workspace is mounted, no network by default,
    read-only root filesystem, no capabilities, memory/cpu/process limits."""

    def __init__(
        self,
        policy: Policy | None = None,
        image: str = "python:3.12-slim",
        network: bool = False,
        memory: str = "512m",
        cpus: str = "1",
        env: dict[str, str] | None = None,
    ) -> None:
        self.policy = policy  # type: ignore[assignment]
        self.image, self.network, self.memory, self.cpus, self.env = image, network, memory, cpus, env or {}

    def check(self) -> None:
        try:
            r = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True, timeout=10
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise BotError(f"Docker is not available: {e}") from e
        if r.returncode != 0:
            raise BotError("Docker is not available: " + (r.stderr.strip() or "is the daemon running?"))

    def build_argv(self, command: str, name: str) -> list[str]:
        argv = [
            "docker", "run", "--rm", "--name", name,
            "--network", "bridge" if self.network else "none",
            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "256",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--read-only", "--tmpfs", "/tmp",
            "-v", f"{self.policy.workdir.resolve()}:/work", "-w", "/work",
        ]
        if hasattr(os, "getuid"):
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        for key, value in self.env.items():
            argv += ["-e", f"{key}={value}"]
        return argv + [self.image, "sh", "-c", command]

    def run(self, command: str, timeout: float) -> tuple[int, str]:
        self.policy.check_command(command, confine=False)
        self.policy.workdir.resolve().mkdir(parents=True, exist_ok=True)
        name = f"botkit-{uuid.uuid4().hex[:12]}"
        proc = subprocess.Popen(
            self.build_argv(command, name), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True, timeout=15)
            proc.kill()
            out, _ = proc.communicate()
            return 124, _clip(out, self.policy.max_output) + f"\n[killed: timed out after {timeout:g}s]"
        return proc.returncode, _clip(out, self.policy.max_output)


def make_sandbox(spec: Any, policy: Policy) -> Sandbox | None:
    """shell=None/False -> no shell | "local" | "docker" | {"env": "docker", "image": "node:20", "network": True} | Sandbox."""
    if spec in (None, False):
        return None
    if isinstance(spec, Sandbox):
        return spec.rebind(policy)
    options: dict[str, Any] = {}
    if isinstance(spec, dict):
        options = dict(spec)
        spec = options.pop("env", None)
    if spec == "local":
        return LocalSandbox(policy, **options)
    if spec == "docker":
        return DockerSandbox(policy, **options)
    raise ValueError(f"Unknown shell environment {spec!r}. Use None, 'local', 'docker', a dict or a Sandbox.")
