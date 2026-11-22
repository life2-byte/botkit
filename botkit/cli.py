"""Command line:  botkit init | chat | sql | providers | version   (also: python -m botkit ...)"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__

APP_PY = '''"""Production entry point. Configure with environment variables (see .env.example)."""
import os

from botkit import BotKit

bot = BotKit(
    os.environ.get("BOTKIT_MODEL", "groq:openai/gpt-oss-120b"),
    # tools=True,                         # files, Word/Excel/PowerPoint, web, calculator
    # skills=True,                        # bundled skills
    # memory=os.environ.get("BOTKIT_DB", "data/bot.db"),   # survives restarts (mount ./data as a volume)
    # multi_agent=True,                   # planner -> worker -> reviewer
    rules=["Keep answers short and friendly."],
    workdir=os.environ.get("BOTKIT_WORKDIR", "data/workspace"),
)

if __name__ == "__main__":
    # BOTKIT_HOST / PORT / BOTKIT_TOKEN / BOTKIT_CORS are read from the environment.
    bot.serve(host=os.environ.get("BOTKIT_HOST", "0.0.0.0"))
'''

DOCKERFILE = '''FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd -m bot && mkdir -p /app/data && chown -R bot /app/data
USER bot
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health')"
CMD ["python", "app.py"]
'''

COMPOSE = '''services:
  bot:
    build: .
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data      # memory database and the bot's files live here
    restart: unless-stopped
'''

ENV_EXAMPLE = '''# Copy to .env and fill in. Never commit .env.
GROQ_API_KEY=
# OPENAI_API_KEY=
# DEEPSEEK_API_KEY=
# ANTHROPIC_API_KEY=
BOTKIT_MODEL=groq:openai/gpt-oss-120b
# Required when the server is reachable from outside (anyone with the token can use your bot):
BOTKIT_TOKEN=change-me-to-a-long-random-string
# Websites allowed to call the API from a browser, comma separated:
# BOTKIT_CORS=https://myapp.com
# Supabase memory (secret key, server only):
# SUPABASE_URL=
# SUPABASE_KEY=
'''

FILES = {
    "app.py": APP_PY,
    "Dockerfile": DOCKERFILE,
    "docker-compose.yml": COMPOSE,
    ".env.example": ENV_EXAMPLE,
    "requirements.txt": f"techtide-botkit[all]>={__version__}\n",
    ".dockerignore": ".env\n.git\n__pycache__\nvenv\n.venv\ndata\n",
    ".gitignore": ".env\n__pycache__/\nvenv/\n.venv/\ndata/\nworkspace/\n*.db\n",
}


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, text in FILES.items():
        target = root / name
        if target.exists() and not args.force:
            print(f"skip   {name} (exists; use --force to overwrite)")
            continue
        target.write_text(text, encoding="utf-8")
        print(f"wrote  {name}")
    print("\nNext:  cp .env.example .env  (add your key)  ->  docker compose up --build  ->  POST /chat")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from .kit import BotKit

    opts: dict = {}
    if args.tools:
        opts["tools"] = True
    if args.skills:
        opts["skills"] = True
    if args.memory:
        opts["memory"] = args.memory
    if args.multi_agent:
        opts["multi_agent"] = True
    bot = BotKit(args.model, **opts)
    bot.run()
    return 0


def cmd_sql(args: argparse.Namespace) -> int:
    from .memory import schema_sql

    print(schema_sql(args.prefix))
    return 0


def cmd_providers(_: argparse.Namespace) -> int:
    from .llm import PROVIDERS

    for name, (url, env) in PROVIDERS.items():
        print(f"{name + ':':<12} key from {env or '(none needed)':<20} {url or 'OpenAI default'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="botkit", description="botkit command line")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="write production files (app.py, Dockerfile, compose, .env.example)")
    s.add_argument("directory", nargs="?", default=".")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("chat", help="chat in the terminal")
    s.add_argument("model", nargs="?", default="groq:openai/gpt-oss-120b")
    s.add_argument("--tools", action="store_true")
    s.add_argument("--skills", action="store_true")
    s.add_argument("--memory", metavar="FILE")
    s.add_argument("--multi-agent", action="store_true", dest="multi_agent")
    s.set_defaults(fn=cmd_chat)
    s = sub.add_parser("sql", help="print the Supabase tables + grants SQL")
    s.add_argument("prefix", nargs="?", default="botkit_")
    s.set_defaults(fn=cmd_sql)
    s = sub.add_parser("providers", help="list built-in providers and their key variables")
    s.set_defaults(fn=cmd_providers)
    s = sub.add_parser("version")
    s.set_defaults(fn=lambda _: print(__version__) or 0)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
