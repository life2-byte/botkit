class BotError(Exception):
    """Base class for all botkit errors."""


class LLMError(BotError):
    """The language model call failed (after retries and after any fallback models).

    `kind` says why: "quota" (daily/monthly limit or no credit), "rate_limit", "server", "network", "auth",
    "not_found", "bad_request", "unsupported" or "exhausted" (every model in a fallback list failed).
    `retry_after` is the number of seconds the provider asked you to wait, when it said so.
    A failed turn is NOT lost: call `bot.resume()` later and it continues where it stopped.
    """

    def __init__(self, message: str = "", *, kind: str = "error", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after


class GraphError(BotError):
    """A node of a multi-agent graph failed, or the graph is built wrongly."""
