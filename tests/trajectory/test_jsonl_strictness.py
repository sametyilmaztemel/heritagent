"""Trajectory reader strictness (review of #54, 2026-10-13): invalid
UTF-8 after a finalization record is never recovered, a line that is not
a JSON object is refused with a typed error, and an overflowing number
(1e999, which would parse as inf) is refused like NaN and Infinity."""

import pytest

from tests.runtime.conftest import ToolObservation, react_action, react_final
from tests.trajectory.conftest import SteppedClock, run_captured
from trajectory.recorder.errors import MalformedTrajectoryError
from trajectory.storage.jsonl import load_trajectory


def finalized(tmp_path, registry, context):
    _, recorder, path, _, _ = run_captured(
        tmp_path / "t", registry, context, name="t",
        script=[react_action("heat_object", {"object": "plate"}), react_final("Done.")],
        env_results={"heat_object": [ToolObservation(ok=True, content="heated")]},
        clock=SteppedClock())
    recorder.finalize()
    return path


@pytest.mark.parametrize("tail", [b"\xc3", b"\xff\n"])
def test_invalid_utf8_after_finalization_is_never_recovered(tmp_path, registry, context, tail):
    path = finalized(tmp_path, registry, context)
    path.write_bytes(path.read_bytes() + tail)
    with pytest.raises(MalformedTrajectoryError, match="appended after a finalized trajectory"):
        load_trajectory(path, allow_incomplete=True)


@pytest.mark.parametrize("line, message", [
    (b"[1, 2]\n", "line 2 is not a JSON object"),
    (b"42\n", "line 2 is not a JSON object"),
    (b'{"record": "event", "latency_ms": 1e999}\n', "non-finite JSON number '1e999'"),
])
def test_a_hostile_line_fails_closed_with_a_typed_error(tmp_path, registry, context, line, message):
    header = finalized(tmp_path, registry, context).read_bytes().split(b"\n", 1)[0] + b"\n"
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(header + line)
    for allow_incomplete in (False, True):
        with pytest.raises(MalformedTrajectoryError, match=message):
            load_trajectory(bad, allow_incomplete=allow_incomplete)
