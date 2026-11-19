"""File tools (5): list_files, read_file, write_file, edit_file, search_files. Confined to the workspace."""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from ..security import Policy
from ..tool import Tool, clip, tool
from .docs import OFFICE_EXT, read_office


def make(policy: Policy, sandbox=None) -> list[Tool]:
    @tool
    def list_files(path: str = ".") -> str:
        """List files and folders in the workspace.

        Args:
            path: Folder relative to the workspace (default: the workspace root).
        """
        p = policy.resolve(path)
        if not p.is_dir():
            raise NotADirectoryError(f"'{path}' is not a folder")
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = [f"{e.name}/" if e.is_dir() else f"{e.name} ({e.stat().st_size} bytes)" for e in entries[:200]]
        return "\n".join(lines) or "(empty)"

    @tool
    def read_file(path: str, max_chars: int = 20000) -> str:
        """Read a file from the workspace. Works for text files and .docx/.xlsx/.pptx documents.

        Args:
            path: File path relative to the workspace.
            max_chars: Maximum characters to return.
        """
        p = policy.resolve(path)
        if not p.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        if p.stat().st_size > policy.max_file_bytes:
            raise ValueError("file is too large to read")
        if p.suffix.lower() in OFFICE_EXT:
            return clip(read_office(p), max_chars)
        raw = p.read_bytes()
        if b"\0" in raw[:2048]:
            return f"[binary file, {len(raw)} bytes]"
        return clip(raw.decode("utf-8", errors="replace"), max_chars)

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file in the workspace.

        Args:
            path: File path relative to the workspace.
            content: The full file content.
        """
        data = content.encode("utf-8")
        if len(data) > policy.max_file_bytes:
            raise ValueError("content is too large")
        p = policy.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return f"Wrote {len(data)} bytes to {path}"

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace one exact piece of text in a workspace file (the text must appear exactly once).

        Args:
            path: File path relative to the workspace.
            old: The exact text to replace.
            new: The replacement text.
        """
        p = policy.resolve(path)
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise ValueError(f"'old' must match exactly once, found {count} matches")
        p.write_text(text.replace(old, new), encoding="utf-8")
        return f"Edited {path}"

    @tool
    def search_files(pattern: str, path: str = ".", glob: str = "*", max_results: int = 50) -> str:
        """Search text files in the workspace for a regular expression (like grep). Returns file:line: text.

        Args:
            pattern: Regular expression to look for (case-insensitive).
            path: Folder or file to search, relative to the workspace.
            glob: Only search files whose name matches this pattern, e.g. "*.py".
            max_results: Maximum number of matching lines to return.
        """
        try:
            rx = re.compile(pattern, re.I)
        except re.error as e:
            raise ValueError(f"invalid regular expression: {e}") from None
        base = policy.workdir.resolve()
        start = policy.resolve(path)
        if start.is_file():
            candidates = [start]
        else:
            candidates = []
            for dirpath, dirs, names in os.walk(start):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                candidates += [Path(dirpath) / n for n in sorted(names) if fnmatch.fnmatch(n, glob)]
                if len(candidates) > 5000:
                    break
        hits: list[str] = []
        for file in candidates:
            try:
                real = policy.resolve(str(file.relative_to(base)))  # symlinks pointing outside are skipped
                if not real.is_file() or real.stat().st_size > 1_000_000:
                    continue
                raw = real.read_bytes()
            except (PermissionError, ValueError, OSError):
                continue
            if b"\0" in raw[:2048]:
                continue
            for lineno, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{file.relative_to(base)}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= max_results:
                        return "\n".join(hits) + f"\n[stopped after {max_results} matches]"
        return "\n".join(hits) or "No matches."

    return [list_files, read_file, write_file, edit_file, search_files]
