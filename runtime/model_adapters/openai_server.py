"""A vLLM OpenAI-compatible server behind the ModelAdapter contract (issue
#13 Phase B; ADR-0008).

Parallel EXP-0001 episodes share one ``vllm serve`` process, which batches
their requests (continuous batching). The adapter holds only the server
URL and the model identity, so it pickles into worker processes. It uses
the standard library's HTTP client only.

- ``generate``: POST ``/v1/chat/completions`` with the messages and the
  explicit settings (temperature, max_tokens, seed, stop), ``n`` = 1.
- ``structured_generate``: the same with a ``json_schema`` response
  format. The server is sent the schema without the keywords its grammar
  backends do not implement (``uniqueItems``: vLLM 0.29.0's xgrammar and
  guidance backends reject it with HTTP 400), and the reply is validated
  here against the full schema, so a reply that breaks a dropped keyword
  is a ModelResponseError, as any malformed reply is.
- ``tool_call``: provider-neutral, as in the offline adapter: a
  constrained oneOf schema over the offered tools.

Every request also sends the neutral sampling values (top_p 1, top_k 0,
min_p 0, repetition_penalty 1, no presence or frequency penalty). The
server otherwise fills unset fields from the model's
``generation_config.json`` (vLLM's default ``--generation-config auto``;
Qwen2.5-7B-Instruct's sets repetition_penalty 1.05, top_p 0.8, top_k 20),
which the offline ``SamplingParams`` path never applies.

Token counts come from the server's ``usage`` block, counted by the
server's own tokenizer. A transport failure (connection refused, a 5xx, a
429, a timeout) is retried a bounded number of times with backoff, then
raised as BackendUnavailableError; a rejected request (another 4xx) is
raised at once. Neither is ever turned into a model answer."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from jsonschema import Draft202012Validator

from runtime.model_adapters.base import (
    GenerationResult,
    GenerationSettings,
    ModelAdapter,
    ModelMessage,
    ModelMetadata,
    ToolCallRequest,
    ToolSpec,
)
from runtime.model_adapters.errors import (
    AdapterError,
    BackendUnavailableError,
    ModelResponseError,
)

SERVER_BACKEND = "vllm-server"
UNSUPPORTED_SCHEMA_KEYWORDS = ("uniqueItems",)  # not implemented by vLLM 0.29.0's grammars


def server_constraint(schema):
    """The schema without the keywords the server cannot constrain; the full
    schema still validates the reply."""
    if isinstance(schema, dict):
        return {key: server_constraint(value) for key, value in schema.items()
                if key not in UNSUPPORTED_SCHEMA_KEYWORDS}
    if isinstance(schema, list):
        return [server_constraint(value) for value in schema]
    return schema
NEUTRAL_SAMPLING = {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
                    "presence_penalty": 0.0, "frequency_penalty": 0.0}


class ServerRequestError(AdapterError):
    """The server rejected a request (a 4xx other than 429), or its reply is
    not a chat completion."""


class VLLMServerAdapter(ModelAdapter):
    def __init__(self, base_url: str, model: str, revision: str, *,
                 backend_version: str | None = None, timeout_seconds: float = 600.0,
                 transport_retries: int = 6, backoff_seconds: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep):
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("the served model revision must be pinned")
        self._url = base_url.rstrip("/")
        self._model = model
        self._revision = revision
        self._backend_version = backend_version
        self._timeout = timeout_seconds
        self._retries = transport_retries
        self._backoff = backoff_seconds
        self._sleep = sleep

    # -- transport --------------------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            if attempt:
                self._sleep(min(self._backoff * 2 ** (attempt - 1), 60.0))
            request = urllib.request.Request(self._url + path, data=data, method=method,
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    body = response.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:2000].decode("utf-8", "replace")
                if exc.code >= 500 or exc.code == 429:
                    last = exc
                    continue
                raise ServerRequestError(f"{method} {path} was rejected ({exc.code}): "
                                         f"{detail}") from None
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                    TimeoutError) as exc:
                last = exc
                continue
            try:
                return json.loads(body.decode("utf-8"))
            except ValueError as exc:
                raise ServerRequestError(f"{method} {path} returned no JSON: {exc}") from None
        raise BackendUnavailableError(f"the model server at {self._url} did not answer "
                                      f"{method} {path} after {self._retries + 1} attempts: "
                                      f"{last}")

    def served_models(self) -> list[str]:
        body = self._request("GET", "/v1/models")
        return [entry.get("id") for entry in body.get("data") or []]

    def require_served(self) -> None:
        """Fail closed unless the server serves this adapter's model id."""
        served = self.served_models()
        if self._model not in served:
            raise BackendUnavailableError(f"the server at {self._url} serves {served}, not "
                                          f"{self._model!r}")

    # -- requests -----------------------------------------------------------------
    def _payload(self, messages: list[ModelMessage], settings: GenerationSettings) -> dict:
        payload = {"model": self._model, "n": 1,
                   "messages": [{"role": m.role, "content": m.content} for m in messages],
                   "temperature": settings.temperature, "max_tokens": settings.max_tokens,
                   **NEUTRAL_SAMPLING}
        if settings.seed is not None:
            payload["seed"] = settings.seed
        if settings.stop:
            payload["stop"] = list(settings.stop)
        return payload

    def _complete(self, payload: dict) -> GenerationResult:
        body = self._request("POST", "/v1/chat/completions", payload)
        try:
            choice = body["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServerRequestError(f"the server's reply is not a chat completion: "
                                     f"{str(body)[:500]}") from exc
        usage = body.get("usage") or {}
        return GenerationResult(text=text or "", finish_reason=choice.get("finish_reason") or "stop",
                                prompt_tokens=usage.get("prompt_tokens"),
                                completion_tokens=usage.get("completion_tokens"))

    @staticmethod
    def _json_schema_format(schema: dict) -> dict:
        return {"type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": True}}

    # -- ModelAdapter -----------------------------------------------------------------
    def generate(self, messages: list[ModelMessage], settings: GenerationSettings) -> GenerationResult:
        return self._complete(self._payload(messages, settings))

    def structured_generate(self, messages: list[ModelMessage], settings: GenerationSettings,
                            schema: dict) -> GenerationResult:
        payload = self._payload(messages, settings)
        payload["response_format"] = self._json_schema_format(server_constraint(schema))
        result = self._complete(payload)
        self._validate_structured(result.text, schema)  # the full schema
        return result

    def tool_call(self, messages: list[ModelMessage], settings: GenerationSettings,
                  tools: list[ToolSpec]) -> ToolCallRequest:
        if not tools:
            raise ModelResponseError("tool_call requires at least one offered tool")
        schema = {"oneOf": [
            {"type": "object", "required": ["name", "arguments"],
             "properties": {"name": {"const": tool.name}, "arguments": tool.parameters_schema},
             "additionalProperties": False}
            for tool in tools]}
        result = self.structured_generate(messages, settings, schema)
        payload = json.loads(result.text)
        selected = next((t for t in tools if t.name == payload.get("name")), None)
        if selected is None:
            raise ModelResponseError(f"tool-call names unknown tool {payload.get('name')!r}")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ModelResponseError(f"tool-call arguments for {selected.name!r} must be a "
                                     f"JSON object")
        errors = [e.message for e in
                  Draft202012Validator(selected.parameters_schema).iter_errors(arguments)]
        if errors:
            raise ModelResponseError(f"tool-call arguments violate {selected.name!r} "
                                     f"parameters_schema: {sorted(errors)}")
        return ToolCallRequest(name=selected.name, arguments=arguments, raw_text=result.text)

    def estimate_tokens(self, text: str) -> int:
        # budgeting estimate only; the server's usage counts are authoritative
        return max(1, (len(text) + 3) // 4)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(model_id=self._model, revision=self._revision,
                             backend=SERVER_BACKEND, backend_version=self._backend_version,
                             capabilities={"generate": True, "structured": True,
                                           "constrained_tool_call": True,
                                           "server": self._url})
