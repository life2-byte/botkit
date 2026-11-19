"""Security policy, enforced in code (not by asking the model nicely).

What it guarantees:
- File tools can only touch files inside the workspace folder (symlinks and `..` are resolved first).
- Shell commands run with a scrubbed environment (your API keys are NOT passed to them).
- Obviously destructive commands are blocked, and in local mode so are paths outside the workspace.
- Secret values (API keys, tokens) are redacted from anything a tool returns to the model.

What it does NOT guarantee: the command filter is a speed bump, not a jail. For untrusted users
use the Docker sandbox, which gives real isolation (no network, read-only root, memory/cpu limits).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# (regex, reason) - blocked in every environment
DEFAULT_DENY: list[tuple[str, str]] = [
    (r"\b(sudo|doas)\b|(^|[;&|]\s*)su(\s|$)", "privilege escalation"),
    (r"\brm\b[^;&|\n]*\s(/|/\*|~|~/|\$HOME)(\s|$)", "deleting outside the workspace"),
    (r":\s*\(\s*\)\s*\{", "fork bomb"),
    (r"\bmkfs(\.\w+)?\b", "formatting a disk"),
    (r"\bdd\b[^\n]*\bof=/dev/", "writing to a device"),
    (r">\s*/dev/(sd|nvme|hd|vd)", "writing to a device"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "power control"),
    (r"\b(curl|wget)\b[^|\n]*\|\s*(sudo\s+)?(ba|z)?sh\b", "piping a download into a shell"),
    (r"/dev/tcp/|\b(nc|ncat|netcat)\b[^\n]*\s-e\b", "reverse shell"),
]

_ABS_PATH = re.compile(r"(?<![\w.:/-])(/[\w.-][\w./-]*)")
_DOTDOT = re.compile(r"(^|[\s/'\"=])\.\.($|[\s/'\"])")
_ALLOWED_ABS = ("/usr/", "/bin/", "/lib", "/dev/null", "/dev/stdout", "/dev/stderr")
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.I)


@dataclass
class Policy:
    """Limits applied to every built-in tool.

    Args:
        workdir: the only folder the file tools and shell may use.
        deny: extra regexes; a shell command matching any is refused (e.g. ["git push", "pip install"]).
        allow_network: set False to remove the web tool.
        confine_shell: in local mode, refuse commands that mention paths outside the workspace.
    """

    workdir: Path
    deny: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    allow_network: bool = True
    confine_shell: bool = True
    max_file_bytes: int = 5_000_000
    max_output: int = 20_000
    shell_timeout_max: int = 300

    def resolve(self, path: str) -> Path:
        """Map a user/model supplied path to a real path inside the workspace, or raise PermissionError."""
        root = self.workdir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = (root / str(path)).resolve()
        if target != root and root not in target.parents:
            raise PermissionError(f"path '{path}' is outside the workspace")
        return target

    def check_command(self, command: str, confine: bool = True) -> None:
        for pattern, reason in DEFAULT_DENY:
            if re.search(pattern, command, re.I):
                raise PermissionError(f"command blocked ({reason})")
        for pattern in self.deny:
            if re.search(pattern, command, re.I):
                raise PermissionError(f"command blocked by app rule: {pattern}")
        if not confine:
            return
        if "~" in command or "$HOME" in command or "${HOME}" in command or _DOTDOT.search(command):
            raise PermissionError("command blocked (paths outside the workspace are not allowed)")
        root = str(self.workdir.resolve())
        for m in _ABS_PATH.finditer(command):
            p = m.group(1)
            if not (p == root or p.startswith(root + "/") or p.startswith(_ALLOWED_ABS)):
                raise PermissionError(f"command blocked (path '{p}' is outside the workspace)")

    def safe_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(self.workdir.resolve()),
            "LANG": "C.UTF-8",
        }
        env.update(extra or {})
        return env

    def redact(self, text: str) -> str:
        values = {s for s in self.secrets if s and len(s) >= 8}
        values |= {v for k, v in os.environ.items() if _SECRET_NAME.search(k) and len(v) >= 8}
        for v in sorted(values, key=len, reverse=True):
            text = text.replace(v, "[REDACTED]")
        return text
