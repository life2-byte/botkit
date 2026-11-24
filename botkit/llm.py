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
import re
import threading
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


_QUOTA_WORDS = (
    "insufficient_quota", "exceeded your current quota", "quota", "billing", "per day", "tokens per day",
    "requests per day", "(tpd)", "(rpd)", "credit", "balance", "payment required",
)
LONG_WAIT = 60.0  # a rate limit that asks for more than this many seconds is treated as a hard limit: do not sit and wait


def _parse_wait(text: str) -> float | None:
    """'Please try again in 12m38.4s' / '1h2m' / '45s' / '350ms' -> seconds."""
    m = re.search(r"try again in\s+((?:\d+(?:\.\d+)?\s*(?:h|ms|m|s)\s*)+)", text, re.I)
    if not m:
        return None
    total = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", m.group(1), re.I):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit.lower()]
    return total


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

    @staticmethod
    def _hard_limit(err: Exception) -> float | None:
        """Seconds to wait if this rate-limit error is a hard limit (quota, daily cap, long wait), else None. 0 = unknown."""
        text = str(err).lower()
        response = getattr(err, "response", None)
        header = response.headers.get("retry-after") if response is not None else None
        wait = None
        try:
            wait = float(header) if header else None
        except ValueError:
            wait = None
        wait = wait if wait is not None else _parse_wait(str(err))
        if any(w in text for w in _QUOTA_WORDS):
            return wait or 0.0
        if wait is not None and wait > LONG_WAIT:
            return wait
        return None

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
                    raise LLMError(f"The server does not support this request (501 Not Implemented): {e}", kind="unsupported") from e
                if isinstance(e, openai.RateLimitError):
                    wait = self._hard_limit(e)
                    if wait is not None:  # daily quota / no credit / a very long wait: retrying now only wastes time
                        raise LLMError(
                            f"'{self.provider}:{self.model}' hit a usage limit (try again in about {int(wait // 60)} min): {e}",
                            kind="quota", retry_after=wait or None,
                        ) from e
                last, delay = e, self._delay(e, attempt)
            except openai.AuthenticationError as e:
                raise LLMError("Authentication failed - check your API key.", kind="auth") from e
            except openai.NotFoundError as e:
                raise LLMError(
                    f"Model '{self.model}' was not found on '{self.provider}' (it may have been renamed or retired): {e}",
                    kind="not_found",
                ) from e
            except openai.BadRequestError as e:
                # Some models emit a malformed tool call now and then; a retry usually fixes it.
                text = str(e).lower()
                if "tool_use_failed" in text or "failed to call a function" in text or "tool choice is none" in text:
                    last, delay = e, 0.5
                else:
                    raise LLMError(f"Bad request: {e}", kind="bad_request") from e
            except openai.APIStatusError as e:
                kind = "quota" if e.status_code == 402 else ("auth" if e.status_code == 403 else "server")
                raise LLMError(f"API error {e.status_code}: {e}", kind=kind) from e
            if attempt < self.max_retries:
                self._sleep(delay)
        kind = "rate_limit" if isinstance(last, openai.RateLimitError) else (
            "server" if isinstance(last, openai.InternalServerError) else "network")
        raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {last}", kind=kind) from last

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
            raise LLMError("The model returned no choices.", kind="server")
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
                    raise LLMError(f"The stream was interrupted: {e}", kind="network") from e
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


class FallbackLLM:
    """Several models behind one client. If the first fails (usage limit, quota, outage, bad key...) the same request
    goes to the next one, so the work continues instead of stopping. A model that just failed is skipped for a while
    (`cooldown` seconds, or as long as the provider said) and tried again afterwards.

        Agent("deepseek:deepseek-chat", fallback=["groq:openai/gpt-oss-120b", "claude:claude-sonnet-4-5"])

    Entries are "provider:model" strings, dicts like {"model": ..., "api_key": ..., "base_url": ...}, or any
    object with `.chat(messages, tools)`. A fallback whose API key is missing is skipped, not fatal.
    `on_switch(from_label, to_label, error)` is called every time a model is given up on.
    """

    def __init__(
        self,
        primary: Any,
        fallbacks: list[Any] | tuple[Any, ...] | str,
        *,
        cooldown: float = 120.0,
        temperature: float = 0.3,
        on_switch: Callable[[str, str, Exception], None] | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self._specs: list[Any] = [primary, *([fallbacks] if isinstance(fallbacks, (str, dict)) else list(fallbacks))]
        self._built: dict[int, Any] = {}
        self._dead: dict[int, str] = {}                 # entries that cannot be built (no key...) and why
        self._until: dict[int, float] = {}              # entry -> monotonic time it may be tried again
        self._kwargs = dict(llm_kwargs, temperature=temperature)
        self.cooldown = cooldown
        self.on_switch = on_switch
        self._lock = threading.Lock()
        self.current = 0
        self._no_stream: set[int] = set()

    # ---- bookkeeping -----------------------------------------------------------
    def _label(self, i: int) -> str:
        spec = self._specs[i]
        if isinstance(spec, str):
            return spec
        if isinstance(spec, dict):
            return str(spec.get("model", f"model{i}"))
        return getattr(spec, "model", None) and f"{getattr(spec, 'provider', '')}:{spec.model}".lstrip(":") or f"model{i}"

    def _get(self, i: int) -> Any:
        if i in self._dead:
            raise LLMError(self._dead[i], kind="auth")
        if i not in self._built:
            spec = self._specs[i]
            try:
                if isinstance(spec, str):
                    self._built[i] = LLM(spec, **self._kwargs)
                elif isinstance(spec, dict):
                    self._built[i] = LLM(**{**self._kwargs, **spec})
                else:
                    self._built[i] = spec
            except LLMError as e:
                self._dead[i] = str(e)
                raise
        return self._built[i]

    @property
    def model(self) -> str:
        return self._label(self.current)

    @property
    def usage(self) -> dict[str, int]:
        total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for llm in self._built.values():
            u = getattr(llm, "usage", None)
            if isinstance(u, dict):
                for k in total:
                    total[k] += int(u.get(k, 0))
        return total

    # ---- the two calls ---------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        return self._run("chat", messages, tools)

    def stream(self, messages: list[dict], tools: list[dict] | None = None, on_token: Callable[[str], None] | None = None) -> dict:
        return self._run("stream", messages, tools, on_token)

    def _run(self, method: str, messages: list[dict], tools: list[dict] | None, on_token: Any = None) -> dict:
        now = time.monotonic()
        order = sorted(range(len(self._specs)), key=lambda i: (self._until.get(i, 0) > now, i))  # ready ones first, in order
        errors: list[tuple[str, Exception]] = []
        for pos, i in enumerate(order):
            emitted = [False]

            def push(t: str, _e=emitted) -> None:
                _e[0] = True
                if on_token:
                    on_token(t)

            try:
                llm = self._get(i)
                if method == "stream" and hasattr(llm, "stream"):
                    reply = llm.stream(messages, tools, push)
                else:
                    reply = llm.chat(messages, tools)
                    if method == "stream" and on_token and reply.get("content"):
                        on_token(reply["content"])
                with self._lock:
                    self.current = i
                return reply
            except LLMError as e:
                errors.append((self._label(i), e))
                kind = getattr(e, "kind", "error")
                if kind not in ("bad_request", "unsupported"):  # these say something about THIS request, not the model
                    pause = e.retry_after if getattr(e, "retry_after", None) else (
                        600.0 if kind == "quota" else self.cooldown)
                    self._until[i] = time.monotonic() + pause
                if emitted[0]:  # text already reached the caller: switching now would repeat it. Stop; resume() continues.
                    raise
                nxt = next((order[p] for p in range(pos + 1, len(order))), None)
                if nxt is not None and self.on_switch:
                    try:
                        self.on_switch(self._label(i), self._label(nxt), e)
                    except Exception:  # noqa: BLE001
                        pass
        waits = [e.retry_after for _, e in errors if getattr(e, "retry_after", None)]
        summary = "; ".join(f"{label}: {str(err)[:160]}" for label, err in errors)
        raise LLMError(f"Every model failed. {summary}", kind="exhausted", retry_after=min(waits) if waits else None)
