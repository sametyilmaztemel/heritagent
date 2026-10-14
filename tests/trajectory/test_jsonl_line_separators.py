"""JSONL record splitting (module audit, 2026-10-13): records are split on
"\\n" only. Canonical JSON keeps U+2028, U+2029 and U+0085 raw inside
strings, and str.splitlines() would cut a record there. A tail cut inside a
multi-byte UTF-8 character is a truncated crash artifact, not a crash of
the reader."""

import json
from pathlib import Path

import pytest

from tests.runtime.conftest import ToolObservation, react_action, react_final
from tests.trajectory.conftest import SteppedClock, run_captured
from trajectory.recorder.errors import MalformedTrajectoryError
from trajectory.storage.canonical import canonical_json
from trajectory.storage.jsonl import load_trajectory

SEPARATORS = "one\u2028two\u2029three\u0085four\x0bfive\x0csix\x1cseven"


def capture(tmp_path, registry, context, content="The plate is now heated.", name="t"):
    _, recorder, path, _, _ = run_captured(
        tmp_path / name, registry, context, name=name,
        script=[react_action("heat_object", {"object": "plate"}), react_final("Done.")],
        env_results={"heat_object": [ToolObservation(ok=True, content=content)]},
        clock=SteppedClock())
    return recorder.finalize(), path


def test_a_payload_with_unicode_line_separators_round_trips(tmp_path, registry, context):
    finalized, path = capture(tmp_path, registry, context, content=SEPARATORS)
    raw = path.read_bytes()
    assert "\u2028".encode() in raw  # canonical JSON keeps it raw (ensure_ascii=False)
    loaded = load_trajectory(path)   # strict path
    assert loaded.status == "finished" and loaded.complete
    results = [e for e in loaded.events if e["kind"] == "tool_result"]
    assert results[0]["data"]["content"] == SEPARATORS
    assert loaded.content_sha256 == finalized.content_sha256


def _prefix_and_cut_tail(path: Path, out: Path) -> None:
    """Keep the header and the first two events, then a partial event line
    cut between the two bytes of 'é'."""
    records = [json.loads(line) for line in path.read_bytes().split(b"\n") if line]
    keep = [r for r in records if r.get("record") == "header"
            or (r.get("record") == "event" and r.get("seq", 0) <= 2)]
    partial = '{"record":"event","note":"café"}'.encode("utf-8")
    cut = partial[:partial.index("é".encode("utf-8")) + 1]  # first byte of 'é' only
    out.write_bytes(b"".join(canonical_json(r).encode() + b"\n" for r in keep) + cut)


def test_a_tail_cut_inside_a_utf8_character_is_recoverable(tmp_path, registry, context):
    _, path = capture(tmp_path, registry, context)
    broken = tmp_path / "cut.jsonl"
    _prefix_and_cut_tail(path, broken)
    with pytest.raises(MalformedTrajectoryError, match="invalid UTF-8 at line 4"):
        load_trajectory(broken)      # strict path refuses, without a UnicodeDecodeError
    recovered = load_trajectory(broken, allow_incomplete=True)
    assert recovered.status == "incomplete" and recovered.complete is False
    assert len(recovered.events) == 2


def test_invalid_utf8_before_the_tail_is_never_recovered(tmp_path, registry, context):
    _, path = capture(tmp_path, registry, context)
    lines = path.read_bytes().split(b"\n")
    lines[1] = lines[1][:-2] + b"\xff" + lines[1][-2:]
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b"\n".join(lines))
    with pytest.raises(MalformedTrajectoryError, match="invalid UTF-8 at line 2"):
        load_trajectory(bad, allow_incomplete=True)


def test_crlf_line_endings_still_load(tmp_path, registry, context):
    finalized, path = capture(tmp_path, registry, context)
    crlf = tmp_path / "crlf.jsonl"
    crlf.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    loaded = load_trajectory(crlf)
    assert loaded.complete and loaded.content_sha256 == finalized.content_sha256
