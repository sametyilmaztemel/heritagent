"""Deterministic scripted ModelAdapter for tests (issue #7 criterion 3).

No GPU, no network, no randomness: every response is queued up front and
consumed in order. All requests (messages + settings) are recorded so tests
can assert exactly what the runtime sent. #8/#11 test suites can reuse this
adapter instead of a live model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from runtime.model_adapters.base import (
    GenerationResult,
    GenerationSettings,
    ModelAdapter,
    ModelMessage,
    ModelMetadata,
    ToolCallRequest,
    ToolSpec,
)
from runtime.model_adapters.errors import ScriptExhaustedError


@dataclass
class _Scripted:
    """One queued response: either text (generate/structured) or a tool call."""

    text: str | None = None
    structured: dict | None = None
    tool_call: ToolCallRequest | None = None
    finish_reason: str = "stop"
    completion_tokens: int | None = None


@dataclass
class RecordedRequest:
    kind: str  # "generate" | "structured_generate" | "tool_call"
    messages: list[ModelMessage]
    settings: GenerationSettings
    schema: dict | None = None
    tools: list[ToolSpec] = field(default_factory=list)


def _render(messages: list[ModelMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


class ScriptedAdapter(ModelAdapter):
    """Queue-based deterministic adapter.

    Responses are consumed FIFO across all three call kinds. When the queue
    is empty, ScriptExhaustedError is raised (fail closed) so tests fail
    loudly instead of hanging on a live model.
    """

    def __init__(self, responses: list[_Scripted | str | dict | ToolCallRequest] | None = None,
                 model_id: str = "scripted-test-model", revision: str = "test-revision"):
        self._queue: list[_Scripted] = []
        for r in responses or []:
            self._queue.append(self._coerce(r))
        self._model_id = model_id
        self._revision = revision
        self.requests: list[RecordedRequest] = []

    @staticmethod
    def _coerce(response) -> _Scripted:
        if isinstance(response, _Scripted):
            return response
        if isinstance(response, ToolCallRequest):
            return _Scripted(tool_call=response, finish_reason="tool_call")
        if isinstance(response, dict):
            return _Scripted(structured=response, text=json.dumps(response, sort_keys=True))
        if isinstance(response, str):
            return _Scripted(text=response)
        raise TypeError(f"cannot script a response from {type(response)!r}")

    # -- scripting helpers -------------------------------------------------
    def enqueue_text(self, text: str, finish_reason: str = "stop",
                     completion_tokens: int | None = None) -> None:
        self._queue.append(_Scripted(text=text, finish_reason=finish_reason,
                                     completion_tokens=completion_tokens))

    def enqueue_structured(self, payload: dict) -> None:
        self._queue.append(self._coerce(payload))

    def enqueue_tool_call(self, name: str, arguments: dict) -> None:
        self._queue.append(self._coerce(ToolCallRequest(name=name, arguments=arguments)))

    # -- ModelAdapter -------------------------------------------------------
    def _pop(self, kind: str) -> _Scripted:
        if not self._queue:
            raise ScriptExhaustedError(
                f"scripted adapter ran out of responses at a {kind} call "
                f"({len(self.requests)} requests recorded)")
        return self._queue.pop(0)

    def generate(self, messages: list[ModelMessage], settings: GenerationSettings) -> GenerationResult:
        response = self._pop("generate")
        self.requests.append(RecordedRequest("generate", list(messages), settings))
        assert response.text is not None, "scripted text response required for generate()"
        return GenerationResult(text=response.text, finish_reason=response.finish_reason,
                                completion_tokens=response.completion_tokens)

    def structured_generate(self, messages: list[ModelMessage], settings: GenerationSettings,
                            schema: dict) -> GenerationResult:
        response = self._pop("structured_generate")
        self.requests.append(RecordedRequest("structured_generate", list(messages), settings, schema=schema))
        if response.structured is not None:
            text = json.dumps(response.structured, sort_keys=True)
        else:
            text = response.text
        payload = self._validate_structured(text, schema)
        return GenerationResult(text=text, finish_reason=response.finish_reason)

    def tool_call(self, messages: list[ModelMessage], settings: GenerationSettings,
                  tools: list[ToolSpec]) -> ToolCallRequest:
        response = self._pop("tool_call")
        self.requests.append(RecordedRequest("tool_call", list(messages), settings, tools=list(tools)))
        assert response.tool_call is not None, "scripted tool call required for tool_call()"
        requested = response.tool_call
        known = {t.name for t in tools}
        if requested.name not in known:
            raise ScriptExhaustedError(f"scripted tool call {requested.name!r} is not among offered tools {sorted(known)}")
        return requested

    def estimate_tokens(self, text: str) -> int:
        # deterministic, content-derived estimate: ~4 chars per token
        return max(1, (len(text) + 3) // 4)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(model_id=self._model_id, revision=self._revision, backend="scripted",
                             backend_version=None, capabilities={"generate": True, "structured": True,
                                                                  "tools": True})
