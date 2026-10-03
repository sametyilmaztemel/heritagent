"""ModelAdapter contract tests (issue #7 criteria 1, 3) — scripted, no GPU."""

import json

import pytest

from runtime.model_adapters import (
    GenerationSettings,
    ModelMessage,
    ScriptExhaustedError,
    ScriptedAdapter,
    ToolCallRequest,
    ToolSpec,
)


def test_scripted_responses_are_consumed_fifo_across_kinds():
    adapter = ScriptedAdapter(["plain text", {"answer": 42}])
    adapter.enqueue_tool_call("pick", {"k": 1})
    assert adapter.generate([ModelMessage("user", "hi")],
                            GenerationSettings()).text == "plain text"
    structured = adapter.structured_generate([ModelMessage("user", "hi")],
                                             GenerationSettings(), {"type": "object"})
    assert json.loads(structured.text) == {"answer": 42}
    call = adapter.tool_call([ModelMessage("user", "hi")], GenerationSettings(),
                             [ToolSpec(name="pick", description="")])
    assert (call.name, call.arguments) == ("pick", {"k": 1})


def test_exhausted_script_fails_closed():
    adapter = ScriptedAdapter([])
    with pytest.raises(ScriptExhaustedError, match="ran out of responses"):
        adapter.generate([ModelMessage("user", "hi")], GenerationSettings())


def test_structured_generate_validates_schema():
    adapter = ScriptedAdapter([{"name": "x"}])
    schema = {"type": "object", "required": ["name"], "additionalProperties": False,
              "properties": {"name": {"type": "string"}}}
    result = adapter.structured_generate([ModelMessage("user", "hi")], GenerationSettings(), schema)
    assert json.loads(result.text)["name"] == "x"

    bad = ScriptedAdapter([{"name": 5}])
    from runtime.model_adapters import ModelResponseError
    with pytest.raises(ModelResponseError):
        bad.structured_generate([ModelMessage("user", "hi")], GenerationSettings(), schema)

    unknown_field = ScriptedAdapter([{"name": "x", "extra": 1}])
    with pytest.raises(ModelResponseError, match="Additional properties"):
        unknown_field.structured_generate([ModelMessage("user", "hi")], GenerationSettings(), schema)


def test_generation_settings_are_recorded_explicitly():
    adapter = ScriptedAdapter(["ok"])
    settings = GenerationSettings(temperature=0.7, max_tokens=64, seed=11, stop=("END",))
    adapter.generate([ModelMessage("system", "s"), ModelMessage("user", "u")], settings)
    request = adapter.requests[0]
    assert request.settings == settings
    assert [m.role for m in request.messages] == ["system", "user"]


def test_tool_call_rejects_unknown_tool():
    adapter = ScriptedAdapter([ToolCallRequest(name="missing", arguments={})])
    with pytest.raises(ScriptExhaustedError, match="not among offered tools"):
        adapter.tool_call([ModelMessage("user", "hi")], GenerationSettings(),
                          [ToolSpec(name="known", description="")])


def test_estimate_tokens_is_deterministic():
    adapter = ScriptedAdapter([])
    assert adapter.estimate_tokens("abcd") == 1
    assert adapter.estimate_tokens("a" * 400) == 100
    assert adapter.estimate_tokens("a" * 400) == adapter.estimate_tokens("a" * 400)


def test_metadata_exposes_identity_for_reproducibility():
    metadata = ScriptedAdapter(model_id="m", revision="r").metadata()
    assert (metadata.model_id, metadata.revision, metadata.backend) == ("m", "r", "scripted")
