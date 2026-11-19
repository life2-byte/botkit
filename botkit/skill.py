"""Skills: folders of instructions the model loads on demand (like Claude Skills).

    skills/
      pirate/SKILL.md      <- frontmatter (name, description) + instructions
      sql-helper/SKILL.md
      tone.md              <- a single .md file also works

Only the name + description sit in the system prompt; the full text is loaded via the `load_skill` tool
when the model decides it needs it, so many skills cost almost no context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .tool import Tool


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path | None = None


def _parse(text: str, default_name: str, path: Path | None) -> Skill:
    meta: dict[str, str] = {}
    body = text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line and not line.startswith((" ", "\t", "#")):
                key, _, val = line.partition(":")
                meta[key.strip().lower()] = val.strip().strip("\"'")
        body = m.group(2)
    body = body.strip()
    first_line = next((ln.strip("# ").strip() for ln in body.splitlines() if ln.strip()), "")
    return Skill(meta.get("name") or default_name, meta.get("description") or first_line[:200], body, path)


def load_skills(source: Any) -> list[Skill]:
    """Accepts a folder path, a Skill, or a list of those."""
    if source is None:
        return []
    if isinstance(source, Skill):
        return [source]
    if isinstance(source, (str, Path)):
        root = Path(source).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Skills path not found: {root}")
        if root.is_file():
            return [_parse(root.read_text(encoding="utf-8"), root.stem, root)]
        if (root / "SKILL.md").is_file():
            return [_parse((root / "SKILL.md").read_text(encoding="utf-8"), root.name, root)]
        skills: list[Skill] = []
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                skills.append(_parse((child / "SKILL.md").read_text(encoding="utf-8"), child.name, child))
            elif child.is_file() and child.suffix == ".md" and child.name.upper() != "README.MD":
                skills.append(_parse(child.read_text(encoding="utf-8"), child.stem, child))
        return skills
    if isinstance(source, Iterable):
        out: list[Skill] = []
        for item in source:
            out.extend(load_skills(item))
        return out
    raise TypeError(f"Unsupported skills source: {source!r}")


class SkillBook:
    def __init__(self, skills: list[Skill]) -> None:
        self.skills = {s.name: s for s in skills}

    def __bool__(self) -> bool:
        return bool(self.skills)

    def prompt(self) -> str:
        lines = "\n".join(f"- {s.name}: {s.description}" for s in self.skills.values())
        return (
            "You have these skills. When a request matches one, call `load_skill` with its name "
            "and follow the instructions it returns.\n" + lines
        )

    def tool(self) -> Tool:
        skills = self.skills

        def load_skill(name: str) -> str:
            if name not in skills:
                raise ValueError(f"unknown skill '{name}'. Available: {', '.join(skills)}")
            return skills[name].body

        return Tool(
            name="load_skill",
            description="Load the full instructions of a skill by name.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "enum": list(skills), "description": "Skill name"}},
                "required": ["name"],
            },
            fn=load_skill,
        )


BUNDLED_DIR = Path(__file__).parent / "skills"


def bundled_names() -> list[str]:
    return sorted(d.name for d in BUNDLED_DIR.iterdir() if (d / "SKILL.md").is_file())


def resolve_skills(spec: Any) -> list[Skill]:
    """What Agent(skills=...) accepts:

        True                      the bundled skills
        "my_skills/"              a folder (or one SKILL.md / .md file) of your own
        "research"                one bundled skill by name
        Skill(...)                an inline skill
        [True, "my_skills/"]      any mix of the above
    """
    if spec in (None, False):
        return []
    if spec is True:
        return load_skills(BUNDLED_DIR)
    if isinstance(spec, Skill):
        return [spec]
    if isinstance(spec, (str, Path)):
        text = str(spec)
        if not Path(text).expanduser().exists() and text in bundled_names():
            return load_skills(BUNDLED_DIR / text)
        return load_skills(spec)
    out: list[Skill] = []
    for item in spec:
        out.extend(resolve_skills(item))
    return out
