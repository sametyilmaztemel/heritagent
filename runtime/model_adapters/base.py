"""ModelAdapter contract (SPEC §21; issue #7 criterion 1).

Provider-neutral, typed request/response boundary. Backend-specific objects
must never cross this module: every request and response is a plain
dataclass, and generation settings are explicit per-call inputs rather than
hidden globals. Foundation-model weights stay frozen — this layer only
performs inference.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator

from runtime.model_adapters.errors import (
    BackendUnavailableError,
    ModelResponseError,
)


@dataclass(frozen=True)
class GenerationSettings:
    """Explicit per-call generation settings (no hidden globals)."""

    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelMessage:
    """One chat message. `role` in {system, user, assistant, tool}."""

    role: str
    content: str


@dataclass(frozen=True)
class ToolSpec:
    """A tool exposed to the model (provider-neutral description)."""

    name: str
    description: str
    parameters_schema: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """Raw text completion plus usage metadata when the backend reports it."""

    text: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ToolCallRequest:
    """The model's chosen tool invocation."""

    name: str
    arguments: dict
    raw_text: str = ""


@dataclass(frozen=True)
class ModelMetadata:
    """Identity information required later for reproducibility (EXP-0001
    records the served weights hash at run time; #13)."""

    model_id: str
    revision: str | None
    backend: str
    backend_version: str | None
    capabilities: dict


class ModelAdapter(ABC):
    """Provider-neutral model boundary used by the runtime core."""

    @abstractmethod
    def generate(self, messages: list[ModelMessage], settings: GenerationSettings) -> GenerationResult:
        """Free-form text generation."""

    @abstractmethod
    def structured_generate(self, messages: list[ModelMessage], settings: GenerationSettings,
                            schema: dict) -> GenerationResult:
        """Generation constrained to a JSON Schema; `result.text` is the JSON document.

        Implementations must validate the payload against `schema` and raise
        ModelResponseError when the model output does not conform.
        """

    @abstractmethod
    def tool_call(self, messages: list[ModelMessage], settings: GenerationSettings,
                  tools: list[ToolSpec]) -> ToolCallRequest:
        """Ask the model to pick and parameterize one of `tools`."""

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Deterministic token estimate for budgeting."""

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Model identity / backend metadata."""

    # -- shared helper ----------------------------------------------------
    @staticmethod
    def _validate_structured(text: str, schema: dict, finish_reason: str | None = None) -> dict:
        """The reply as a JSON object that satisfies ``schema``, else a
        ModelResponseError carrying the reply and its finish_reason. NaN and
        Infinity are not JSON (RFC 8259) and pass jsonschema's bounds, so
        they are rejected here, as is a payload that cannot be stored as
        strict UTF-8 JSON."""
        def reject(message):
            return ModelResponseError(message, raw_text=text, finish_reason=finish_reason)

        def no_constant(name):
            raise reject(f"structured output contains {name}, which is not valid JSON")

        try:
            payload = json.loads(text, parse_constant=no_constant)
        except ValueError as exc:  # also an integer past Python's digit limit
            raise reject(f"structured output is not valid JSON: {exc}") from None
        try:  # an overflowing literal (inf) or a lone surrogate escape
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except ValueError as exc:
            raise reject(f"structured output cannot be stored as strict JSON: {exc}") from None
        if not isinstance(payload, dict):
            raise reject("structured output must be a JSON object")
        errors = [e.message for e in Draft202012Validator(schema).iter_errors(payload)]
        if errors:
            raise reject(sorted(errors))
        return payload


__all__ = [
    "BackendUnavailableError",
    "GenerationResult",
    "GenerationSettings",
    "ModelAdapter",
    "ModelMessage",
    "ModelMetadata",
    "ModelResponseError",
    "ToolCallRequest",
    "ToolSpec",
]
