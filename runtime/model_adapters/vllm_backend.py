"""Optional vLLM backend (Qwen2.5-7B-Instruct) behind the ModelAdapter contract.

Dependency strategy (issue #7 criterion 2): vLLM is imported lazily inside
`_engine()` so the core test suite never requires vLLM or a GPU. When the
dependency is missing, AdapterError/BackendUnavailableError is raised with a
clear message — never a bare ImportError. The served model identity is
surfaced through metadata(); the weights hash for EXP-0001 reproducibility
is pinned and recorded at run time (#13).

NOTE: the engine-call paths are written against the vLLM offline LLM API
(`vllm.LLM`, `SamplingParams`, guided decoding, tool chat). They are covered
by contract/failure-path tests only in this PR; parameter names may need a
small adjustment pass when first run against a real engine (#13).
"""

from __future__ import annotations

import importlib
from typing import Any

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
    UnsupportedFeatureError,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


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
            try:
                vllm_module = importlib.import_module("vllm")  # lazy: heavy, GPU-bound
                if vllm_module is None:  # sentinel-injected absence (deterministic tests)
                    raise ImportError("vllm is not installed (module unavailable)")
                llm_cls = vllm_module.LLM
            except ImportError as exc:
                raise BackendUnavailableError(
                    f"vLLM is not installed; install the optional runtime extra "
                    f"(pip install vllm) to use {self._model!r} — original error: {exc}") from None
            kwargs: dict[str, Any] = {
                "model": self._model,
                "gpu_memory_utilization": self._gpu_memory_utilization,
                "max_model_len": self._max_model_len,
            }
            if self._revision:
                kwargs["revision"] = self._revision
            self._llm = llm_cls(**kwargs)
        return self._llm

    # -- request translation -------------------------------------------------
    @staticmethod
    def _prompts(messages: list[ModelMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _sampling(self, settings: GenerationSettings) -> Any:
        # engine creation precedes any _sampling call, so vLLM is importable here
        vllm_module = importlib.import_module("vllm")
        kwargs: dict[str, Any] = {"temperature": settings.temperature, "max_tokens": settings.max_tokens}
        if settings.seed is not None:
            kwargs["seed"] = settings.seed
        if settings.stop:
            kwargs["stop"] = list(settings.stop)
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
        engine = self._engine()  # raises BackendUnavailableError when vLLM is missing
        result = engine.chat(self._prompts(messages), self._sampling(settings))
        return self._extract(result[0] if isinstance(result, list) else result)

    def structured_generate(self, messages: list[ModelMessage], settings: GenerationSettings,
                            schema: dict) -> GenerationResult:
        try:
            guided_module = importlib.import_module("vllm.sampling_params")
            if guided_module is None:
                raise ImportError("vllm.sampling_params is unavailable")
            guided_cls = guided_module.GuidedDecodingParams
        except ImportError as exc:
            raise UnsupportedFeatureError(
                f"vLLM guided decoding is unavailable in this installation: {exc}") from None
        engine = self._engine()
        params = self._sampling(settings)
        params.guided_decoding = guided_cls(json=schema)
        result = engine.chat(self._prompts(messages), params)
        extracted = self._extract(result[0] if isinstance(result, list) else result)
        self._validate_structured(extracted.text, schema)
        return extracted

    def tool_call(self, messages: list[ModelMessage], settings: GenerationSettings,
                  tools: list[ToolSpec]) -> ToolCallRequest:
        engine = self._engine()
        chat_tools = [{"type": "function",
                       "function": {"name": t.name, "description": t.description,
                                    "parameters": t.parameters_schema}} for t in tools]
        result = engine.chat(self._prompts(messages), self._sampling(settings),
                             tools=chat_tools, tool_choice="required")
        output = (result[0] if isinstance(result, list) else result).outputs[0]
        calls = getattr(output, "tool_calls", None) or []
        if not calls:
            raise UnsupportedFeatureError("vLLM returned no tool call despite tool_choice=required")
        call = calls[0]
        function = getattr(call, "function", call)
        arguments = function.arguments if isinstance(function.arguments, dict) \
            else __import__("json").loads(function.arguments)
        return ToolCallRequest(name=function.name, arguments=arguments,
                               raw_text=getattr(output, "text", "") or "")

    def estimate_tokens(self, text: str) -> int:
        # budgeting estimate only; the engine's own tokenizer is authoritative
        return max(1, (len(text) + 3) // 4)

    def metadata(self) -> ModelMetadata:
        backend_version = None
        try:
            import vllm  # noqa: PLC0415 — cheap version probe; vLLM may be absent
            backend_version = getattr(vllm, "__version__", None)
        except ImportError:
            backend_version = None
        return ModelMetadata(model_id=self._model, revision=self._revision, backend="vllm",
                             backend_version=backend_version,
                             capabilities={"generate": True, "structured": True, "tools": True,
                                           "engine_created": self._llm is not None})
