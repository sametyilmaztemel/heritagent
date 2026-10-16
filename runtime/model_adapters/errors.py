"""Adapter-layer error types (fail-closed, backend-agnostic)."""


class AdapterError(Exception):
    """Base class for ModelAdapter failures."""


class BackendUnavailableError(AdapterError):
    """The optional backend (e.g. vLLM) or its dependency is unavailable."""


class UnsupportedFeatureError(AdapterError):
    """The backend cannot serve the requested feature (e.g. guided decoding)."""


class ContextWindowExceeded(AdapterError):
    """The prompt does not fit the model's context window. The runtime loop
    ends the episode as budget_exhausted (budget context_window), counted as
    a failure (ADR-0009 D4); the trait miner sends the pair's evidence without
    the admissible-command lists, then rejects the pair (ADR-0009 D8). A
    prompt is never truncated and no call is retried as it was."""


class ModelResponseError(AdapterError):
    """The model's response violated the requested contract (invalid JSON,
    schema mismatch, malformed tool call). `raw_text` preserves the exact
    teacher output for audit when available; never regex-recovered."""

    def __init__(self, message, raw_text: str | None = None):
        self.raw_text = raw_text
        super().__init__(message)


class ScriptExhaustedError(AdapterError):
    """A scripted (test) adapter ran out of scripted responses."""
