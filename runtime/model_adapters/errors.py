"""Adapter-layer error types (fail-closed, backend-agnostic)."""


class AdapterError(Exception):
    """Base class for ModelAdapter failures."""


class BackendUnavailableError(AdapterError):
    """The optional backend (e.g. vLLM) or its dependency is unavailable."""


class UnsupportedFeatureError(AdapterError):
    """The backend cannot serve the requested feature (e.g. guided decoding)."""


class ModelResponseError(AdapterError):
    """The model's response violated the requested contract (invalid JSON,
    schema mismatch, malformed tool call)."""


class ScriptExhaustedError(AdapterError):
    """A scripted (test) adapter ran out of scripted responses."""
