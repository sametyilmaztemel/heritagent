"""vLLM backend contract tests — scripted fake module, no GPU, no vLLM install.

The fake module pins the contract the adapter relies on (LLM, SamplingParams,
GuidedDecodingParams) and lets us test request translation, settings
propagation, and the failure paths deterministically.
"""

import json
import sys
import types
from types import SimpleNamespace

import pytest

from runtime.model_adapters import (
    BackendUnavailableError,
    GenerationSettings,
    ModelMessage,
    ToolCallRequest,
    ToolSpec,
    UnsupportedFeatureError,
)
from runtime.model_adapters.vllm_backend import VLLMAdapter


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeOutput:
    def __init__(self, text="fake-output", finish_reason="stop", tool_calls=None):
        self.text = text
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls


class FakeUsage:
    prompt_tokens = 11
    completion_tokens = 7


class FakeResult:
    def __init__(self, output):
        self.outputs = [output]
        self.usage = FakeUsage()


class FakeLLM:
    last_kwargs = None
    last_call = None

    def __init__(self, **kwargs):
        FakeLLM.last_kwargs = kwargs
        self.kwargs = kwargs

    def chat(self, messages, params, **kwargs):
        FakeLLM.last_call = (messages, params, kwargs)
        return [FakeResult(FakeOutput())]


@pytest.fixture
def fake_vllm(monkeypatch):
    module = types.ModuleType("vllm")
    module.LLM = FakeLLM
    module.SamplingParams = FakeSamplingParams
    module.__version__ = "0.0-fake"
    monkeypatch.setitem(sys.modules, "vllm", module)
    FakeLLM.last_kwargs = None
    FakeLLM.last_call = None
    return module


def test_missing_vllm_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)  # force ImportError deterministically
    adapter = VLLMAdapter()
    with pytest.raises(BackendUnavailableError, match="vLLM is not installed"):
        adapter.generate([ModelMessage("user", "hi")], GenerationSettings())


def test_metadata_without_engine(fake_vllm):
    metadata = VLLMAdapter(revision="rev-123").metadata()
    assert metadata.backend == "vllm"
    assert metadata.revision == "rev-123"
    assert metadata.backend_version == "0.0-fake"
    assert metadata.capabilities["engine_created"] is False


def test_generate_translates_settings_and_records_revision(fake_vllm):
    adapter = VLLMAdapter(revision="rev-123")
    result = adapter.generate([ModelMessage("system", "s"), ModelMessage("user", "u")],
                              GenerationSettings(temperature=0.4, max_tokens=77, seed=5))
    assert result.text == "fake-output"
    assert result.completion_tokens == 7
    assert FakeLLM.last_kwargs["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert FakeLLM.last_kwargs["revision"] == "rev-123"
    params = FakeLLM.last_call[1]
    assert (params.temperature, params.max_tokens, params.seed) == (0.4, 77, 5)
    assert [m["role"] for m in FakeLLM.last_call[0]] == ["system", "user"]


def test_structured_generate_uses_guided_decoding_and_validates(fake_vllm, monkeypatch):
    guided = types.ModuleType("vllm.sampling_params")

    class GuidedDecodingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    guided.GuidedDecodingParams = GuidedDecodingParams
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", guided)

    def chat(self, messages, params, **kwargs):
        FakeLLM.last_call = (messages, params, kwargs)
        return [FakeResult(FakeOutput(text='{"x": 1}'))]

    FakeLLM.chat = chat
    adapter = VLLMAdapter()
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    result = adapter.structured_generate([ModelMessage("user", "hi")], GenerationSettings(), schema)
    assert json.loads(result.text) == {"x": 1}
    assert FakeLLM.last_call[1].guided_decoding.json == schema


def test_structured_generate_without_guided_decoding_fails_clearly(fake_vllm, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", None)  # import -> ImportError
    adapter = VLLMAdapter()
    with pytest.raises(UnsupportedFeatureError, match="guided decoding"):
        adapter.structured_generate([ModelMessage("user", "hi")], GenerationSettings(),
                                    {"type": "object"})


def test_tool_call_translates_tools_and_result(fake_vllm):
    class FakeFunction:
        name = "heat_object"
        arguments = '{"object": "plate"}'

    class FakeCall:
        function = FakeFunction()

    def chat(self, messages, params, tools=None, tool_choice=None):
        FakeLLM.last_call = (messages, params, {"tools": tools, "tool_choice": tool_choice})
        return [FakeResult(FakeOutput(tool_calls=[FakeCall()]))]

    FakeLLM.chat = chat
    adapter = VLLMAdapter()
    request = adapter.tool_call([ModelMessage("user", "hi")], GenerationSettings(),
                                [ToolSpec(name="heat_object", description="",
                                          parameters_schema={"type": "object"})])
    assert isinstance(request, ToolCallRequest)
    assert request.arguments == {"object": "plate"}
    assert FakeLLM.last_call[2]["tool_choice"] == "required"
    assert FakeLLM.last_call[2]["tools"][0]["function"]["name"] == "heat_object"
