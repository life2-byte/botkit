"""Shared chat helpers for anything with `chat(text, on_token=None)`: Agent and Graph both use them."""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

from .errors import BotError

_DONE = object()


class ChatMixin:
    def stream(self, user_input: str):
        """Generator that yields the reply piece by piece:  for piece in bot.stream("hi"): print(piece, end="")"""
        q: queue.Queue = queue.Queue()

        def work() -> None:
            try:
                streamed: list[str] = []

                def push(t: str) -> None:
                    streamed.append(t)
                    q.put(t)

                answer = self.chat(user_input, on_token=push)  # type: ignore[attr-defined]
                if not streamed:  # nothing streamed (custom llm, or a graph's final step): deliver the whole answer
                    q.put(answer)
                q.put(_DONE)
            except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
                q.put(e)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def achat(self, user_input: str) -> str:
        """Async version of chat (the loop runs in a worker thread, so your event loop stays free)."""
        return await asyncio.to_thread(self.chat, user_input)  # type: ignore[attr-defined]

    async def astream(self, user_input: str):
        """Async generator: `async for piece in bot.astream("hi"): ...`"""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def work() -> None:
            try:
                streamed: list[str] = []

                def push(t: str) -> None:
                    streamed.append(t)
                    loop.call_soon_threadsafe(q.put_nowait, t)

                answer = self.chat(user_input, on_token=push)  # type: ignore[attr-defined]
                if not streamed:
                    loop.call_soon_threadsafe(q.put_nowait, answer)
                loop.call_soon_threadsafe(q.put_nowait, _DONE)
            except BaseException as e:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, e)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = await q.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def run(self, prompt: str = "You: ", stream: bool = True) -> None:
        """Interactive terminal chat. Type 'exit' to quit, '/clear' to forget this conversation."""
        print("Chat started. Type 'exit' to quit, '/clear' to reset.")
        while True:
            try:
                text = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in ("exit", "quit", "/exit", "/quit"):
                break
            if text == "/clear":
                self.reset()  # type: ignore[attr-defined]
                print("(conversation cleared)")
                continue
            try:
                if stream:
                    print("Bot: ", end="", flush=True)
                    for piece in self.stream(text):
                        print(piece, end="", flush=True)
                    print()
                else:
                    print(f"Bot: {self.chat(text)}")  # type: ignore[attr-defined]
            except KeyboardInterrupt:
                print("\n(interrupted)")
            except BotError as e:
                print(f"\n[error] {e}")
            except Exception as e:  # noqa: BLE001 - keep the REPL alive
                print(f"\n[error] {type(e).__name__}: {e}")
        self.close()  # type: ignore[attr-defined]

    def serve(self, host: str | None = None, port: int | None = None, **options: Any) -> Any:
        """Expose this bot as an HTTP API for a web/mobile UI (see botkit.server.serve for options)."""
        from .server import serve

        return serve(self, host=host, port=port, **options)

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()  # type: ignore[attr-defined]
