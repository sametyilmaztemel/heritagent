"""ModelAdapter contract tests (issue #7 criteria 1, 3) — scripted, no GPU."""

import json
import sys

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


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_structured_output_with_a_non_json_number_is_rejected(constant):
    from runtime.model_adapters import ModelResponseError
    schema = {"type": "object", "properties": {"x": {"type": "number", "minimum": 0,
                                                       "maximum": 1}}}
    adapter = ScriptedAdapter([f'{{"x": {constant}}}'])
    with pytest.raises(ModelResponseError, match=f"contains {constant}") as raised:
        adapter.structured_generate([ModelMessage("user", "hi")], GenerationSettings(), schema)
    assert raised.value.raw_text == f'{{"x": {constant}}}'


@pytest.mark.skipif(not getattr(sys, "get_int_max_str_digits", lambda: 0)(),
                    reason="this interpreter has no integer digit limit")
def test_structured_output_with_an_integer_past_pythons_digit_limit_is_rejected():
    from runtime.model_adapters import ModelResponseError
    adapter = ScriptedAdapter(['{"x": ' + "1" * 5000 + "}"])
    with pytest.raises(ModelResponseError, match="not valid JSON"):
        adapter.structured_generate([ModelMessage("user", "hi")], GenerationSettings(),
                                    {"type": "object"})


@pytest.mark.parametrize("value", ["1e999", '"\\ud800"'])
def test_structured_output_that_parses_but_cannot_be_stored_is_rejected(value):
    """An overflowing literal loads as inf and a lone surrogate escape as a
    string that UTF-8 cannot encode; neither can be stored as strict JSON."""
    from runtime.model_adapters import ModelResponseError
    adapter = ScriptedAdapter([f'{{"x": {value}}}'])
    with pytest.raises(ModelResponseError, match="cannot be stored as strict JSON") as raised:
        adapter.structured_generate([ModelMessage("user", "hi")], GenerationSettings(),
                                    {"type": "object"})
    assert raised.value.raw_text == f'{{"x": {value}}}'


def test_a_structured_reply_cut_at_max_tokens_carries_its_finish_reason():
    from runtime.model_adapters import ModelResponseError
    from runtime.model_adapters.fake import _Scripted
    adapter = ScriptedAdapter([_Scripted(text='{"proposals": [{"na', finish_reason="length")])
    with pytest.raises(ModelResponseError) as raised:
        adapter.structured_generate([ModelMessage("user", "hi")], GenerationSettings(),
                                    {"type": "object"})
    assert raised.value.finish_reason == "length"
    assert raised.value.raw_text == '{"proposals": [{"na'


# -- the vLLM server adapter's transport, without a server ----------------------

class _Reply:
    def __init__(self, body: dict):
        self._data = json.dumps(body).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _server_adapter(monkeypatch, replies):
    """A VLLMServerAdapter whose urlopen pops ``replies``: a dict is a 200
    body, an exception is raised."""
    from runtime.model_adapters import openai_server
    calls = []

    def urlopen(request, timeout):
        calls.append(json.loads(request.data) if request.data else None)
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return _Reply(reply)

    monkeypatch.setattr(openai_server.urllib.request, "urlopen", urlopen)
    adapter = openai_server.VLLMServerAdapter("http://server", "m", "a" * 40,
                                              transport_retries=1, sleep=lambda _: None)
    return adapter, calls


def _chat(text, finish_reason):
    return {"choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}}


def test_the_server_adapters_truncated_structured_reply_carries_its_finish_reason(monkeypatch):
    from runtime.model_adapters import ModelResponseError
    adapter, _ = _server_adapter(monkeypatch, [_chat('{"a": [1, 2', "length")])
    with pytest.raises(ModelResponseError, match="not valid JSON") as raised:
        adapter.structured_generate([ModelMessage("user", "u")], GenerationSettings(),
                                    {"type": "object"})
    assert raised.value.finish_reason == "length"


def _http_error(code, body_error):
    import io
    import urllib.error

    class Body(io.BytesIO):
        def read(self, *args):
            raise body_error

    return urllib.error.HTTPError("http://server/v1/chat/completions", code, "error", {},
                                  Body())


@pytest.mark.parametrize("code", [400, 503])
def test_an_error_body_that_cannot_be_read_is_a_transport_failure(monkeypatch, code):
    """A connection reset while the error body is read leaves the error
    unclassified (an overflow cannot be told from a rejection), so it is
    retried as a transport failure and then fails closed."""
    from runtime.model_adapters import BackendUnavailableError
    reset = ConnectionResetError("connection reset by peer")
    adapter, calls = _server_adapter(monkeypatch, [_http_error(code, reset),
                                                   _chat("Final Answer: x", "stop")])
    assert adapter.generate([ModelMessage("user", "u")], GenerationSettings()).text == \
        "Final Answer: x"
    assert len(calls) == 2
    adapter, calls = _server_adapter(monkeypatch, [_http_error(code, reset),
                                                   _http_error(code, reset)])
    with pytest.raises(BackendUnavailableError, match="after 2 attempts.*connection reset"):
        adapter.generate([ModelMessage("user", "u")], GenerationSettings())


def test_an_error_body_cut_short_is_a_transport_failure(monkeypatch):
    """http.client.IncompleteRead is not an OSError, so it is named apart."""
    import http.client
    cut = http.client.IncompleteRead(b"The decoder prompt", 40)
    adapter, calls = _server_adapter(monkeypatch, [_http_error(400, cut),
                                                   _chat("Final Answer: x", "stop")])
    assert adapter.generate([ModelMessage("user", "u")], GenerationSettings()).text == \
        "Final Answer: x"
    assert len(calls) == 2
