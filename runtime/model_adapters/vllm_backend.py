"""Optional vLLM backend (Qwen2.5-7B-Instruct) behind the ModelAdapter contract.

Dependency strategy (issue #7 criterion 2; API surface per critic review):
vLLM is imported lazily inside `_engine()` so the core test suite never
requires vLLM or a GPU. When the dependency is missing, BackendUnavailableError
is raised with a clear message — never a bare ImportError.

The backend targets the CURRENT official offline API only:
- structured outputs via `vllm.sampling_params.StructuredOutputsParams`
  passed as `SamplingParams(structured_outputs=...)`;
- offline `LLM.chat(messages, sampling_params, ...)` — `tool_choice` is NOT
  a parameter of the offline chat API and is never sent;
- offline `tool_call()` is implemented provider-neutrally: a constrained
  JSON schema (oneOf over the offered tools, each with its own argument
  schema) drives structured generation, and the returned call is validated
  against the selected tool's parameters_schema. Native tool-choice
  semantics belong to a future server/OpenAI-compatible adapter.

An installed vLLM that lacks `StructuredOutputsParams` fails closed with
UnsupportedFeatureError for structured features. The served model identity
is surfaced through metadata(); the weights hash for EXP-0001
reproducibility is pinned and recorded at run time (#13).
"""

from __future__ import annotations

import importlib
from typing import Any

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
    BackendUnavailableError,
    ModelResponseError,
    UnsupportedFeatureError,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _import_vllm() -> Any:
    try:
        vllm_module = importlib.import_module("vllm")
    except ImportError as exc:
        raise BackendUnavailableError(
            f"vLLM is not installed; install the optional runtime extra "
            f"(pip install vllm) — original error: {exc}") from None
    if vllm_module is None:  # sentinel-injected absence (deterministic tests)
        raise BackendUnavailableError("vLLM is not installed (module unavailable)")
    return vllm_module


class VLLMAdapter(ModelAdapter):
    """vLLM-backed adapter; the engine is created lazily on first use."""

    def __init__(self, model: str = DEFAULT_MODEL, revision: str | None = None,
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 8192):
        self._model = model
        self._revision = revision
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._llm: Any | None = None

    # -- engine lifecycle ---------------------------------------------------
    def _engine(self) -> Any:
        if self._llm is None:
            llm_cls = _import_vllm().LLM  # raises BackendUnavailableError when absent
            kwargs: dict[str, Any] = {
                "model": self._model,
                "gpu_memory_utilization": self._gpu_memory_utilization,
                "max_model_len": self._max_model_len,
            }
            if self._revision:
                kwargs["revision"] = self._revision
            self._llm = llm_cls(**kwargs)
        return self._llm

    def _structured_outputs_params_cls(self) -> Any:
        """Current-API probe: structured outputs require
        `vllm.sampling_params.StructuredOutputsParams`; older installations
        (GuidedDecodingParams era) fail closed here."""
        try:
            module = importlib.import_module("vllm.sampling_params")
            if module is None:
                raise ImportError("vllm.sampling_params is unavailable")
            params_cls = getattr(module, "StructuredOutputsParams", None)
        except ImportError as exc:
            raise UnsupportedFeatureError(
                f"installed vLLM does not expose vllm.sampling_params: {exc}") from None
        if params_cls is None:
            raise UnsupportedFeatureError(
                "installed vLLM lacks StructuredOutputsParams (current structured-outputs "
                "interface); update vLLM to use structured_generate/tool_call")
        return params_cls

    # -- request translation -------------------------------------------------
    @staticmethod
    def _prompts(messages: list[ModelMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def _sampling(settings: GenerationSettings, vllm_module: Any,
                  structured_outputs: Any | None = None) -> Any:
        kwargs: dict[str, Any] = {"temperature": settings.temperature, "max_tokens": settings.max_tokens}
        if settings.seed is not None:
            kwargs["seed"] = settings.seed
        if settings.stop:
            kwargs["stop"] = list(settings.stop)
        if structured_outputs is not None:
            kwargs["structured_outputs"] = structured_outputs
        return vllm_module.SamplingParams(**kwargs)

    @staticmethod
    def _extract(result: Any) -> GenerationResult:
        output = result.outputs[0]
        usage = getattr(result, "usage", None)
        return GenerationResult(
            text=output.text,
            finish_reason=getattr(output, "finish_reason", "stop") or "stop",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    # -- ModelAdapter ---------------------------------------------------------
    def generate(self, messages: list[ModelMessage], settings: GenerationSettings) -> GenerationResult:
        vllm_module = _import_vllm()
        engine = self._engine()  # raises BackendUnavailableError when vLLM is missing
        result = engine.chat(self._prompts(messages), self._sampling(settings, vllm_module))
        return self._extract(result[0] if isinstance(result, list) else result)

    def structured_generate(self, messages: list[ModelMessage], settings: GenerationSettings,
                            schema: dict) -> GenerationResult:
        vllm_module = _import_vllm()
        params_cls = self._structured_outputs_params_cls()
        engine = self._engine()
        structured_outputs = params_cls(json=schema)
        result = engine.chat(self._prompts(messages),
                             self._sampling(settings, vllm_module, structured_outputs))
        extracted = self._extract(result[0] if isinstance(result, list) else result)
        self._validate_structured(extracted.text, schema)
        return extracted

    def tool_call(self, messages: list[ModelMessage], settings: GenerationSettings,
                  tools: list[ToolSpec]) -> ToolCallRequest:
        """Provider-neutral offline tool calling: a constrained JSON schema
        (oneOf over the offered tools, each carrying its own argument schema)
        drives structured generation; the returned call is validated against
        the selected tool's parameters_schema. The offline chat API has no
        `tool_choice` parameter and none is ever sent."""
        if not tools:
            raise ModelResponseError("tool_call requires at least one offered tool")
        vllm_module = _import_vllm()
        params_cls = self._structured_outputs_params_cls()
        engine = self._engine()

        schema = {"oneOf": [
            {"type": "object",
             "required": ["name", "arguments"],
             "properties": {"name": {"const": tool.name},
                            "arguments": tool.parameters_schema},
             "additionalProperties": False}
            for tool in tools
        ]}
        structured_outputs = params_cls(json=schema)
        result = engine.chat(self._prompts(messages),
                             self._sampling(settings, vllm_module, structured_outputs))
        extracted = self._extract(result[0] if isinstance(result, list) else result)
        payload = self._validate_structured(extracted.text, {"type": "object"})
        name = payload.get("name")
        selected = next((t for t in tools if t.name == name), None)
        if selected is None:
            raise ModelResponseError(f"tool-call names unknown tool {name!r}; offered: "
                                     f"{[t.name for t in tools]}")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ModelResponseError(f"tool-call arguments for {name!r} must be a JSON object")
        arg_errors = [e.message for e in
                      Draft202012Validator(selected.parameters_schema).iter_errors(arguments)]
        if arg_errors:
            raise ModelResponseError(f"tool-call arguments violate {name!r} parameters_schema: "
                                     f"{sorted(arg_errors)}")
        return ToolCallRequest(name=selected.name, arguments=arguments, raw_text=extracted.text)

    def estimate_tokens(self, text: str) -> int:
        # budgeting estimate only; the engine's own tokenizer is authoritative
        return max(1, (len(text) + 3) // 4)

    def metadata(self) -> ModelMetadata:
        backend_version = None
        structured_supported = False
        try:
            module = _import_vllm()
            backend_version = getattr(module, "__version__", None)
            sampling_params = importlib.import_module("vllm.sampling_params")
            if sampling_params is not None:
                structured_supported = hasattr(sampling_params, "StructuredOutputsParams")
        except ImportError:
            pass  # absent installation: defaults stand
        return ModelMetadata(model_id=self._model, revision=self._revision, backend="vllm",
                             backend_version=backend_version,
                             capabilities={"generate": True,
                                           "structured": structured_supported,
                                           "offline_constrained_tool_call": structured_supported,
                                           "engine_created": self._llm is not None})
