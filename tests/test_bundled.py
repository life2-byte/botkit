import json
import subprocess
import sys
from pathlib import Path

import pytest

from botkit import Agent, Skill, tool
from botkit.skill import BUNDLED_DIR, bundled_names, resolve_skills
from botkit.tools import GROUPS
from botkit.tools.util import safe_eval
from botkit.tools.web import parse_search_results
from test_botkit import FakeLLM, add, call, say


def run(bot, name, args):
    return bot._exec({"id": "x", "function": {"name": name, "arguments": json.dumps(args)}}, {})


# ---------------------------------------------------------------- default = only the loop
def test_default_agent_is_just_the_loop(tmp_path):
    llm = FakeLLM(say("hi"))
    bot = Agent(llm=llm, workdir=tmp_path)
    assert bot.tool_names == [] and bot.skills is None and bot.memory is None and bot.sandbox is None
    assert bot.chat("hello") == "hi"
    system = llm.seen[0][0][0]["content"]
    assert "Security rules" in system and "skills" not in system.lower().replace("security rules", "")
    assert llm.seen[0][1] is None                                    # no tools sent to the model


# ---------------------------------------------------------------- tools flag
def test_bundled_tool_inventory(tmp_path):
    bot = Agent(llm=FakeLLM(), tools=True, shell="local", workdir=tmp_path)
    assert sorted(bot.tool_names) == sorted([
        "list_files", "read_file", "write_file", "edit_file", "search_files",
        "create_docx", "create_xlsx", "create_pptx",
        "fetch_url", "web_search", "run_command", "calculate",
    ])
    assert set(GROUPS) == {"files", "docs", "web", "shell", "util"}


def test_tools_flag_forms(tmp_path):
    def names(**kw):
        return set(Agent(llm=FakeLLM(), workdir=tmp_path, **kw).tool_names)

    assert names(tools=True) >= {"read_file", "calculate"} and "run_command" not in names(tools=True)
    assert names(tools="util") == {"calculate"}
    assert names(tools=["files"]) == {"list_files", "read_file", "write_file", "edit_file", "search_files"}
    assert names(tools=["read_file", "calculate"]) == {"read_file", "calculate"}
    assert names(tools=[add]) == {"add"} and names(tools=add) == {"add"}
    assert names(tools=["util", add]) == {"calculate", "add"}
    assert names(tools=[True, add]) >= {"add", "read_file"}
    assert names(tools=None) == set() and names(tools=False) == set()
    assert names(shell="local") == {"run_command"}                   # a shell environment turns on run_command alone
    with pytest.raises(ValueError, match="Unknown bundled tool"):
        names(tools=["nope"])
    with pytest.raises(ValueError, match="needs an environment"):
        names(tools=["run_command"])


def test_user_tool_overrides_bundled_with_same_name(tmp_path):
    @tool
    def calculate(expression: str) -> str:
        """Mine."""
        return "mine"

    bot = Agent(llm=FakeLLM(), tools=[True, calculate], workdir=tmp_path)
    assert run(bot, "calculate", {"expression": "1+1"}) == "mine"


def test_docs_group_without_libraries_gives_install_hint(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("docx", "openpyxl", "pptx"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    bot = Agent(llm=FakeLLM(), tools=["docs"], workdir=tmp_path)
    assert "botkit[docs]" in run(bot, "create_docx", {"path": "a", "title": "t", "content": "c"})


def test_core_imports_do_not_need_optional_packages():
    code = (
        "import sys\n"
        "for m in ('docx','pptx','openpyxl','mcp'): sys.modules[m] = None\n"
        "import botkit\n"
        "from botkit import Agent\n"
        "class L:\n"
        "    def chat(self, m, t=None): return {'role':'assistant','content':'ok'}\n"
        "print(Agent(llm=L(), tools=['files','util']).chat('hi'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "ok", out.stderr


# ---------------------------------------------------------------- calculate
def test_calculate_is_exact_and_safe():
    assert safe_eval("(1250 * 1.18) / 12") == pytest.approx(122.9166666)
    assert safe_eval("2 ** 10 + sqrt(16) - -3") == 1031
    assert safe_eval("max(1, 5, 3) % 4") == 1 and safe_eval("pi") == pytest.approx(3.14159265)
    for bad in ["__import__('os').system('ls')", "open('x')", "().__class__", "a + 1", "1; 2", "lambda: 1",
                "'a' * 5", "9 ** 9 ** 9", "x" * 600, "sqrt(1, key=2)", "[1,2]"]:
        with pytest.raises(ValueError):
            safe_eval(bad)
    with pytest.raises(ZeroDivisionError):
        safe_eval("1/0")


def test_calculate_through_agent(tmp_path):
    bot = Agent(llm=FakeLLM(), tools="util", workdir=tmp_path)
    assert run(bot, "calculate", {"expression": "2+2"}) == "4"
    assert run(bot, "calculate", {"expression": "__import__('os')"}).startswith("Error: ValueError")


# ---------------------------------------------------------------- search_files / web_search parser
def test_search_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("import os\nprint('Hello')\n")
    (tmp_path / "b.txt").write_text("hello world\nnothing\n")
    (tmp_path / "bin.dat").write_bytes(b"\0hello")
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("hello secret")
    (tmp_path / "link.txt").symlink_to(outside)
    ws = tmp_path
    bot = Agent(llm=FakeLLM(), tools="files", workdir=ws)
    out = run(bot, "search_files", {"pattern": "hello"})
    assert "src/a.py:2: print('Hello')" in out and "b.txt:1: hello world" in out
    assert "bin.dat" not in out and "secret" not in out and "link.txt" not in out
    assert run(bot, "search_files", {"pattern": "hello", "glob": "*.py"}).count("\n") == 0
    assert run(bot, "search_files", {"pattern": "zzz"}) == "No matches."
    assert "invalid regular expression" in run(bot, "search_files", {"pattern": "("})
    assert "outside the workspace" in run(bot, "search_files", {"pattern": "x", "path": ".."})
    assert "stopped after 1" in run(bot, "search_files", {"pattern": "hello", "max_results": 1})


DDG = """
<div class="result"><h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc">First &amp; <b>best</b></a></h2>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">The <b>first</b> snippet.</a></div>
<div class="result"><h2><a class="result__a" href="https://direct.example.org/page">Second</a></h2>
<a class="result__snippet">Second snippet</a></div>
<div class="result"><h2><a class="result__a" href="https://third.example.org">No snippet here</a></h2></div>
"""


def test_web_search_parser_and_tool(tmp_path, monkeypatch):
    r = parse_search_results(DDG)
    assert r[0] == {"title": "First & best", "url": "https://example.com/a?x=1", "snippet": "The first snippet."}
    assert r[1]["url"] == "https://direct.example.org/page" and r[1]["snippet"] == "Second snippet"
    assert r[2]["snippet"] == "" and len(r) == 3 and len(parse_search_results(DDG, 2)) == 2
    assert parse_search_results("<html>blocked</html>") == []

    from botkit.tools import web
    monkeypatch.setattr(web, "_fetch", lambda url, max_bytes=0: ("text/html", DDG))
    bot = Agent(llm=FakeLLM(), tools="web", workdir=tmp_path)
    out = run(bot, "web_search", {"query": "tea"})
    assert out.startswith("1. First & best\n   https://example.com/a?x=1") and "3. No snippet here" in out
    monkeypatch.setattr(web, "_fetch", lambda url, max_bytes=0: ("text/html", "<html>captcha</html>"))
    assert "No results" in run(bot, "web_search", {"query": "tea"})


# ---------------------------------------------------------------- skills flag: 10 bundled skills
def test_ten_bundled_skills_are_well_formed():
    names = bundled_names()
    assert names == sorted([
        "research", "planning", "verify-work", "writing-documents", "presentations",
        "spreadsheets", "data-analysis", "coding", "debugging", "summarizing",
    ])
    for skill in resolve_skills(True):
        assert skill.name in names and len(skill.description) > 30 and len(skill.body.split()) > 60, skill.name
        assert skill.description.startswith("Use ")
    tool_names = set().union(*[{t.name for t in f(__import__("botkit").Policy(workdir=Path("."), ), object())}
                               for f in GROUPS.values()])
    import re
    for skill in resolve_skills(True):  # every tool a skill mentions must really exist
        for mentioned in re.findall(r"`([a-z_]+)`", skill.body):
            if mentioned in {"verify-work"}:
                continue
            assert mentioned in tool_names | {"verify-work"}, f"{skill.name} mentions unknown tool {mentioned}"


def test_skills_flag_forms(tmp_path):
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "SKILL.md").write_text("---\nname: mine\ndescription: my skill\n---\nbody text")

    def names(spec):
        bot = Agent(llm=FakeLLM(), skills=spec, workdir=tmp_path / "w")
        return set(bot.skills.skills) if bot.skills else set()

    assert len(names(True)) == 10
    assert names("research") == {"research"}
    assert names(["coding", "debugging"]) == {"coding", "debugging"}
    assert names(tmp_path / "mine") == {"mine"} and names(str(tmp_path)) == {"mine"}
    assert len(names([True, tmp_path / "mine"])) == 11
    assert names(Skill("inline", "d", "b")) == {"inline"}
    assert names(None) == set() and names(False) == set()
    with pytest.raises(FileNotFoundError):
        names("no_such_skill")


def test_bundled_skill_loads_through_the_loop(tmp_path):
    llm = FakeLLM(call("load_skill", {"name": "research"}), say("done"))
    bot = Agent(llm=llm, skills=True, workdir=tmp_path)
    bot.chat("what is the price of gold today?")
    system = llm.seen[0][0][0]["content"]
    assert "- research: Use for questions about current facts" in system and "- coding:" in system
    assert "Sources:" in llm.seen[1][0][-1]["content"]


def test_skills_are_packaged():
    assert BUNDLED_DIR.is_dir() and all((BUNDLED_DIR / n / "SKILL.md").is_file() for n in bundled_names())
    assert "skills/*/SKILL.md" in (Path(__file__).parent.parent / "pyproject.toml").read_text()
