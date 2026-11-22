class BotError(Exception):
    """Base class for all botkit errors."""


class LLMError(BotError):
    """The language model call failed (after retries)."""


class GraphError(BotError):
    """A node of a multi-agent graph failed, or the graph is built wrongly."""
