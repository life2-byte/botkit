"""LLM client: any OpenAI-compatible API (Groq, OpenAI, OpenRouter, Gemini, Ollama...) with retries."""
from __future__ import annotations

import os
import random
import time
from typing import Any

import openai

from .errors import LLMError

# provider -> (base_url, env var holding the API key)
PROVIDERS: dict[str, tuple[str | None, str | None]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}

_RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)


def parse_model(model: str) -> tuple[str, str]:
    """'groq:openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b'). No prefix -> openai."""
    head, sep, rest = model.partition(":")
    if sep and head in PROVIDERS:
        return head, rest
    return "openai", model


class LLM:
    """Thin wrapper over the chat-completions API. `chat()` returns one assistant message dict."""

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
        provider, self.model = parse_model(model)
        default_url, key_env = PROVIDERS[provider]
        api_key = api_key or (os.environ.get(key_env) if key_env else "ollama")
        if not api_key:
            raise LLMError(f"No API key for '{provider}'. Set the {key_env} environment variable or pass api_key=.")
        self.temperature = temperature
        self.max_retries = max_retries
        self._sleep = time.sleep
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url or default_url, timeout=timeout, max_retries=0
        )

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

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except _RETRYABLE as e:
                last, delay = e, self._delay(e, attempt)
            except openai.AuthenticationError as e:
                raise LLMError("Authentication failed - check your API key.") from e
            except openai.BadRequestError as e:
                # Llama models on Groq sometimes emit a malformed tool call; a retry usually fixes it.
                text = str(e).lower()
                if "tool_use_failed" in text or "failed to call a function" in text:
                    last, delay = e, 0.5
                else:
                    raise LLMError(f"Bad request: {e}") from e
            except openai.APIStatusError as e:
                raise LLMError(f"API error {e.status_code}: {e}") from e
            if attempt < self.max_retries:
                self._sleep(delay)
        else:
            raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {last}") from last

        if not resp.choices:
            raise LLMError("The model returned no choices.")
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
