"""vLLM backend contract tests — scripted fake module mirroring the CURRENT
official offline API (StructuredOutputsParams via SamplingParams; no
tool_choice on offline LLM.chat). No GPU, no vLLM install required."""

import json
import sys
import types

import pytest

from runtime.model_adapters import (
    BackendUnavailableError,
    GenerationSettings,
    ModelMessage,
    ModelResponseError,
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
    """Mirrors the current offline surface: chat(messages, sampling_params, ...).
    There is deliberately NO tool_choice parameter (it does not exist on the
    offline LLM.chat API)."""

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
    assert metadata.capabilities["structured"] is False  # no sampling_params submodule yet


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
    assert FakeLLM.last_call[2] == {}  # no tool_choice / no extra kwargs on offline chat


def test_structured_generate_uses_structured_outputs_params(fake_vllm, monkeypatch):
    sampling_params = types.ModuleType("vllm.sampling_params")

    class StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    sampling_params.StructuredOutputsParams = StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    def chat(self, messages, params, **kwargs):
        FakeLLM.last_call = (messages, params, kwargs)
        return [FakeResult(FakeOutput(text='{"x": 1}'))]

    FakeLLM.chat = chat
    adapter = VLLMAdapter()
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    result = adapter.structured_generate([ModelMessage("user", "hi")], GenerationSettings(), schema)
    assert json.loads(result.text) == {"x": 1}
    assert FakeLLM.last_call[1].structured_outputs.json == schema  # current API field name
    assert "guided_decoding" not in vars(FakeLLM.last_call[1])


def test_structured_generate_fails_closed_on_older_vllm(fake_vllm, monkeypatch):
    # older installation: only the GuidedDecodingParams era interface exists
    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.GuidedDecodingParams = type("GuidedDecodingParams", (), {})
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)
    adapter = VLLMAdapter()
    with pytest.raises(UnsupportedFeatureError, match="lacks StructuredOutputsParams"):
        adapter.structured_generate([ModelMessage("user", "hi")], GenerationSettings(),
                                    {"type": "object"})


def test_tool_call_uses_constrained_schema_without_tool_choice(fake_vllm, monkeypatch):
    sampling_params = types.ModuleType("vllm.sampling_params")

    class StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    sampling_params.StructuredOutputsParams = StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    def chat(self, messages, params, **kwargs):
        FakeLLM.last_call = (messages, params, kwargs)
        return [FakeResult(FakeOutput(text=json.dumps(
            {"name": "heat_object", "arguments": {"object": "plate"}})))]

    FakeLLM.chat = chat
    adapter = VLLMAdapter()
    request = adapter.tool_call([ModelMessage("user", "hi")], GenerationSettings(),
                                [ToolSpec(name="heat_object", description="heat things",
                                          parameters_schema={"type": "object",
                                                             "required": ["object"],
                                                             "properties": {"object": {"type": "string"}}})])
    assert request.name == "heat_object"
    assert request.arguments == {"object": "plate"}
    so = FakeLLM.last_call[1].structured_outputs
    assert "heat_object" in json.dumps(so.json)  # oneOf constrained schema embeds the tool
    assert FakeLLM.last_call[2] == {}  # offline chat received NO tool_choice kwarg


def test_tool_call_rejects_unknown_tool_and_invalid_arguments(fake_vllm, monkeypatch):
    sampling_params = types.ModuleType("vllm.sampling_params")

    class StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.json = kwargs.get("json")

    sampling_params.StructuredOutputsParams = StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    responses = ['{"name": "ghost_tool", "arguments": {}}',
                 '{"name": "heat_object", "arguments": {"wrong": true}}']
    queue = list(responses)

    def chat(self, messages, params, **kwargs):
        return [FakeResult(FakeOutput(text=queue.pop(0)))]

    FakeLLM.chat = chat
    adapter = VLLMAdapter()
    tools = [ToolSpec(name="heat_object", description="",
                      parameters_schema={"type": "object", "required": ["object"],
                                          "properties": {"object": {"type": "string"}}})]
    with pytest.raises(ModelResponseError, match="unknown tool"):
        adapter.tool_call([ModelMessage("user", "hi")], GenerationSettings(), tools)
    with pytest.raises(ModelResponseError, match="parameters_schema"):
        adapter.tool_call([ModelMessage("user", "hi")], GenerationSettings(), tools)
