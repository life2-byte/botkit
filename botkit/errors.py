class BotError(Exception):
    """Base class for all botkit errors."""


class LLMError(BotError):
    """The language model call failed (after retries)."""
