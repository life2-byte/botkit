"""LLM clients. One provider-neutral interface (`chat` / `stream`), many providers behind it.

Any OpenAI-compatible endpoint works: OpenAI, Groq, DeepSeek, Claude (Anthropic's OpenAI-compatible endpoint),
Gemini, OpenRouter, Mistral, Together, Ollama, LM Studio, vLLM... A provider is just a name, a base URL and the
environment variable that holds its key, so you can add your own with `register_provider`.

Different agents can use different providers: Agent("deepseek:deepseek-chat"), Agent("claude:claude-sonnet-4-5").
The `openai` package is imported only when a client is actually created (custom `llm=` objects never need it).
"""
from __future__ import annotations

import os
import random
import time
from typing import Any, Callable

from .errors import LLMError

# provider -> (base_url, env var holding the API key)
PROVIDERS: dict[str, tuple[str | None, str | None]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "claude": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "local": ("http://localhost:1234/v1", None),  # LM Studio, vLLM, llama.cpp server... (override with base_url=)
}


def register_provider(name: str, base_url: str | None, api_key_env: str | None = None) -> None:
    """Add (or replace) a provider so "name:model" works: register_provider("mycorp", "https://llm.corp/v1", "CORP_KEY")."""
    PROVIDERS[name.strip().lower()] = (base_url, api_key_env)


def parse_model(model: str) -> tuple[str, str]:
    """'groq:openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b'). No known prefix -> openai."""
    head, sep, rest = model.partition(":")
    if sep and head.lower() in PROVIDERS:
        return head.lower(), rest
    return "openai", model


def make_llm(model: str, **kwargs: Any) -> "LLM":
    """Build a client from "provider:model". Same as LLM(model, ...)."""
    return LLM(model, **kwargs)


class LLM:
    """Chat-completions client with retries. `chat()` returns one assistant message dict;
    `stream()` does the same but calls `on_token(text)` for every piece of text as it arrives."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_retries: int = 4,
        timeout: float = 60.0,
    ) -> None:
        import openai  # lazy: only needed when a real client is built

        self._openai = openai
        provider, self.model = parse_model(model)
        self.provider = provider
        default_url, key_env = PROVIDERS[provider]
        api_key = api_key or (os.environ.get(key_env) if key_env else "none")
        if not api_key:
            raise LLMError(f"No API key for '{provider}'. Set the {key_env} environment variable or pass api_key=.")
        self.temperature = temperature
        self.max_retries = max_retries
        self._sleep = time.sleep
        self._no_stream = False  # set when the server turns out not to support streaming
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url or default_url, timeout=timeout, max_retries=0)

    # ---- helpers --------------------------------------------------------------
    @staticmethod
    def _delay(err: Exception, attempt: int) -> float:
        response = getattr(err, "response", None)
        retry_after = response.headers.get("retry-after") if response is not None else None
        try:
            if retry_after:
                return min(float(retry_after), 30.0)
        except ValueError:
            pass
        return min(2.0**attempt, 20.0) + random.random()

    def _kwargs(self, messages: list[dict], tools: list[dict] | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _request(self, attempt_fn: Callable[[], Any]) -> Any:
        """Run one API attempt with retries for rate limits, network trouble, 5xx and malformed tool calls."""
        openai = self._openai
        retryable = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return attempt_fn()
            except retryable as e:
                if getattr(e, "status_code", None) == 501:  # "Not Implemented" will never succeed: do not retry
                    raise LLMError(f"The server does not support this request (501 Not Implemented): {e}") from e
                last, delay = e, self._delay(e, attempt)
            except openai.AuthenticationError as e:
                raise LLMError("Authentication failed - check your API key.") from e
            except openai.NotFoundError as e:
                raise LLMError(
                    f"Model '{self.model}' was not found on '{self.provider}' (it may have been renamed or retired): {e}"
                ) from e
            except openai.BadRequestError as e:
                # Some models emit a malformed tool call now and then; a retry usually fixes it.
                text = str(e).lower()
                if "tool_use_failed" in text or "failed to call a function" in text or "tool choice is none" in text:
                    last, delay = e, 0.5
                else:
                    raise LLMError(f"Bad request: {e}") from e
            except openai.APIStatusError as e:
                raise LLMError(f"API error {e.status_code}: {e}") from e
            if attempt < self.max_retries:
                self._sleep(delay)
        raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {last}") from last

    def _count(self, usage: Any) -> None:
        self.usage["calls"] += 1
        
        if usage is not None and (getattr(usage, "prompt_tokens", 0) or 0) > 0:
            self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0

    # ---- the two calls --------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        kwargs = self._kwargs(messages, tools)
        resp = self._request(lambda: self._client.chat.completions.create(**kwargs))
        if not resp.choices:
            raise LLMError("The model returned no choices.")
        self._count(getattr(resp, "usage", None))
        msg = resp.choices[0].message
        out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in msg.tool_calls
            ]
        return out
    def _chat_as_stream(self, messages: list[dict], tools: list[dict] | None, on_token: Callable[[str], None] | None) -> dict:
        reply = self.chat(messages, tools)
        if on_token and reply.get("content"):
            on_token(reply["content"])
        return reply
    def stream(
        self, messages: list[dict], tools: list[dict] | None = None, on_token: Callable[[str], None] | None = None
    ) -> dict:
        if self._no_stream:
            return self._chat_as_stream(messages, tools, on_token)
        kwargs = self._kwargs(messages, tools)
        kwargs["stream"] = True
        emitted = False
        for attempt in range(2):  # one clean retry if the stream breaks before any text reached the caller
            content: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            try:
                chunks = self._request(lambda: self._client.chat.completions.create(**kwargs))
                for chunk in chunks:
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        content.append(delta.content)
                        emitted = True
                        if on_token:
                            on_token(delta.content)
                    for tc in getattr(delta, "tool_calls", None) or []:
                        slot = calls.setdefault(tc.index or 0, {"id": "", "name": "", "arguments": ""})
                        slot["id"] = tc.id or slot["id"]
                        if tc.function:
                            slot["name"] += tc.function.name or ""
                            slot["arguments"] += tc.function.arguments or ""
                break
            except LLMError as e:
                # Some OpenAI-compatible servers cannot stream (501 and similar): remember it and use normal replies.
                if not emitted and ("501" in str(e) or "stream" in str(e).lower()):
                    self._no_stream = True
                    return self._chat_as_stream(messages, tools, on_token)
                raise
            except (self._openai.APIConnectionError, self._openai.APITimeoutError) as e:
                if emitted or attempt == 1:
                    raise LLMError(f"The stream was interrupted: {e}") from e
        self._count(None)
        out: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if calls:
            out["tool_calls"] = [
                {
                    "id": c["id"] or f"call_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                }
                for i, c in sorted(calls.items())
            ]
        return out
