"""ModelAdapter contract, scripted test adapter, and the optional vLLM backend."""

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
    ScriptExhaustedError,
    UnsupportedFeatureError,
)
from runtime.model_adapters.fake import RecordedRequest, ScriptedAdapter
from runtime.model_adapters.vllm_backend import VLLMAdapter

__all__ = [
    "AdapterError",
    "BackendUnavailableError",
    "GenerationResult",
    "GenerationSettings",
    "ModelAdapter",
    "ModelMessage",
    "ModelMetadata",
    "ModelResponseError",
    "RecordedRequest",
    "ScriptExhaustedError",
    "ScriptedAdapter",
    "ToolCallRequest",
    "ToolSpec",
    "UnsupportedFeatureError",
    "VLLMAdapter",
]
