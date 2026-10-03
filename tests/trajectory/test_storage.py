"""JSONL storage tests: append-only persistence, reload, integrity digests,
incomplete/crash recovery (issue #8 criteria 5, 6, 11)."""

import json
from pathlib import Path

import pytest

from trajectory.recorder.errors import (
    MalformedTrajectoryError,
    RecorderError,
    StreamInvariantError,
    TrajectoryIntegrityError,
)
from trajectory.storage.canonical import canonical_json
from runtime.events import RuntimeEvent
from trajectory.recorder.recorder import TrajectoryRecorder
from trajectory.storage.jsonl import load_trajectory
from tests.runtime.conftest import ToolObservation, react_action, react_final
from tests.trajectory.conftest import SteppedClock, run_captured

GOLDEN_ENV = {"heat_object": [ToolObservation(ok=True, content="The plate is now heated.")]}
GOLDEN_SCRIPT = [react_action("heat_object", {"object": "plate"}),
                 react_final("The plate is heated.")]


def golden_capture(tmp_path, registry, context, name="golden"):
    _, recorder, path, _, _ = run_captured(tmp_path / name, registry, context, name=name,
                                            script=GOLDEN_SCRIPT, env_results=GOLDEN_ENV,
                                            clock=SteppedClock())
    recorder.finalize()
    return path


def test_jsonl_round_trip(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    loaded = load_trajectory(path)
    assert loaded.status == "finished"
    assert loaded.complete is True
    import re as _re
    assert _re.match(r"^T-[0-9a-f]{12}$", loaded.trajectory_id)  # opaque id from header
    assert loaded.context.genome_id == "G-0f1e2d3c"
    assert loaded.context.model.model_id == "scripted-test-model"
    assert loaded.context.budget.max_steps == 4
    assert loaded.events[0]["kind"] == "run_started"
    assert loaded.events[-1]["kind"] == "finished"
    loaded2 = load_trajectory(path)
    assert loaded2.document == loaded.document  # reload twice: identical evidence


def test_jsonl_lines_are_canonical(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    for line in path.read_text().splitlines():
        record = json.loads(line)
        assert line == canonical_json(record)  # UTF-8, sorted keys, compact separators


def test_digest_verification_detects_event_tampering(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if record.get("record") == "event" and record["kind"] == "tool_result":
            record["data"]["content"] = "TAMPERED RESULT"
    path.write_text("\n".join(canonical_json(r) for r in records) + "\n")
    with pytest.raises(TrajectoryIntegrityError, match="integrity digest mismatch"):
        load_trajectory(path)


def test_digest_verification_detects_context_tampering(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["context"]["seed"] = 999
    lines[0] = canonical_json(header)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(TrajectoryIntegrityError):
        load_trajectory(path)


def test_digest_verification_detects_outcome_tampering(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    lines = path.read_text().splitlines()
    finalization = json.loads(lines[-1])
    finalization["outcome"] = {"success": True, "reward": 100.0}
    lines[-1] = canonical_json(finalization)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(TrajectoryIntegrityError):
        load_trajectory(path)


def test_digest_verification_detects_event_order_swap(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    lines = path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]  # swap two event records
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(TrajectoryIntegrityError):
        load_trajectory(path)


def test_crashed_run_without_finalization_is_incomplete(tmp_path, registry, context):
    _, recorder, path, _, _ = run_captured(tmp_path / "crash", registry, context, name="crash",
                                            script=[react_action("heat_object",
                                                                  {"object": "plate"}),
                                                    react_final("never captured")],
                                            env_results={"heat_object": [ToolObservation(
                                                ok=True, content="ok")]})
    # simulate a crash mid-run: keep header + captured events, drop the
    # finalization record AND the terminal event (neither was persisted yet)
    lines = path.read_text().splitlines()
    crashed = Path(str(path) + ".crash.jsonl")
    records = [json.loads(line) for line in lines]
    first_result = next(i for i, r in enumerate(records)
                        if r.get("kind") == "tool_result")
    keep = records[: first_result + 1]  # crash right after the first tool result
    crashed.write_text("\n".join(canonical_json(r) for r in keep) + "\n")

    with pytest.raises(MalformedTrajectoryError, match="no finalization record"):
        load_trajectory(crashed)  # strict path refuses

    recovered = load_trajectory(crashed, allow_incomplete=True)  # explicit recovery
    assert recovered.status == "incomplete"
    assert recovered.complete is False
    assert recovered.document["finalization"]["incomplete_reason"] == \
        "no finalization record (crashed run)"
    assert [e["kind"] for e in recovered.events][-1] == "tool_result"  # prefix preserved


def test_truncated_trailing_line_requires_explicit_recovery(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    keep = [r for r in records if r.get("record") == "header"
            or (r.get("record") == "event" and r.get("seq", 0) <= 2)]
    broken = Path(str(path) + ".broken.jsonl")
    broken.write_text("\n".join(canonical_json(r) for r in keep)
                       + '\n{"record":"finalization","status":"incomp')  # partial line

    with pytest.raises(MalformedTrajectoryError, match="malformed JSONL"):
        load_trajectory(broken)
    recovered = load_trajectory(broken, allow_incomplete=True)  # explicit recovery path
    assert recovered.status == "incomplete"
    assert recovered.document["events"][-1]["kind"] == "step_started"
    assert recovered.document["events"][-1]["kind"] == "step_started"


def test_unknown_record_type_rejected(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    lines = path.read_text().splitlines()
    lines.insert(1, canonical_json({"record": "mystery", "payload": {"x": 1}}))
    altered = Path(str(path) + ".mystery.jsonl")
    altered.write_text("\n".join(lines) + "\n")
    with pytest.raises(MalformedTrajectoryError, match="unknown record types"):
        load_trajectory(altered, allow_incomplete=True)


def test_wrong_header_schema_version_rejected(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["schema_version"] = "9.9"
    lines[0] = canonical_json(header)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(MalformedTrajectoryError, match="unsupported header schema_version"):
        load_trajectory(path)


def test_non_finite_json_rejected_by_reader(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    lines = path.read_text().splitlines()
    event = json.loads(lines[1])
    event["data"]["score"] = float("nan")  # re-serialized with Python's NaN extension
    lines[1] = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=True)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(MalformedTrajectoryError, match="non-standard JSON"):
        load_trajectory(path, allow_incomplete=True)


def test_inconsistent_complete_flag_rejected(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    document = load_trajectory(path).document
    document["complete"] = True  # contradictory: status is finished but pretend incomplete
    document["status"] = "incomplete"
    document["finalization"]["incomplete_reason"] = "tampered"
    document["finalization"]["content_sha256"] = "0" * 64
    from trajectory.verification import verify_document
    with pytest.raises(TrajectoryIntegrityError, match="inconsistent"):
        verify_document(document)


def test_invalid_event_timestamp_type_rejected(tmp_path, registry, context):
    recorder = TrajectoryRecorder(context, tmp_path / "ts.jsonl")
    recorder(RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.25,
                          data={"task": "t"}))
    with pytest.raises(StreamInvariantError, match="is not a number"):
        recorder(RuntimeEvent(seq=2, kind="step_started", step=1,
                              monotonic_s="not-a-number", data={}))


def test_incomplete_recovery_preserves_header_identity(tmp_path, registry, context):
    """Crash recovery rebuilds the document under the SAME opaque id from
    the header — the id survives, the digest is recomputed for the prefix."""
    recorder = TrajectoryRecorder(context, tmp_path / "prefix.jsonl",
                                   trajectory_id="T-bbbbbbbbbbbb")
    recorder(RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.25,
                          data={"task": "t"}))
    recorder(RuntimeEvent(seq=2, kind="step_started", step=1, monotonic_s=0.5, data={}))
    recorder.finalize_incomplete("simulated crash")
    loaded = load_trajectory(tmp_path / "prefix.jsonl")
    assert loaded.trajectory_id == "T-bbbbbbbbbbbb"  # header identity preserved
    assert loaded.status == "incomplete"


def test_atomic_creation_refuses_existing_empty_file(tmp_path, registry, context):
    """Even an EMPTY existing file is refused: open("x") is atomic
    (O_CREAT|O_EXCL) — no check-then-open race, bytes never change."""
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(RecorderError, match="path already exists"):
        TrajectoryRecorder(context, path)
    assert path.read_text() == ""  # untouched


def test_atomic_creation_refuses_existing_nonempty_file(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    bytes_before = path.read_bytes()
    with pytest.raises(RecorderError, match="path already exists"):
        TrajectoryRecorder(context, path)
    assert path.read_bytes() == bytes_before  # evidence byte-identical


def test_finalize_verification_failure_writes_no_footer(tmp_path, registry, context):
    """A stream that passes the incremental checks but violates the document
    schema (boolean step on a terminal event) cannot be finalized — and no
    finalization record claiming completion is persisted."""
    from trajectory.verification import TrajectoryIntegrityError as _TIE
    path = tmp_path / "bad.jsonl"
    recorder = TrajectoryRecorder(context, path,
                                   trajectory_id="T-cccccccccccc")
    recorder(RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.25,
                          data={"task": "t"}))
    recorder(RuntimeEvent(seq=2, kind="finished", step=True, monotonic_s=0.5,
                          data={"answer": "x"}))  # boolean step: incremental passes
    with pytest.raises(RecorderError):  # TrajectoryIntegrityError before any footer
        recorder.finalize()
    lines = path.read_text().splitlines()
    assert len(lines) == 3  # header + 2 events
    assert all(json.loads(line).get("record") != "finalization" for line in lines)
    assert recorder._finalized is False  # white-box: recorder stays open for recovery


def test_extra_header_field_rejected(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["unexpected"] = "tampered"
    lines[0] = canonical_json(header)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(MalformedTrajectoryError, match="header envelope fields mismatch"):
        load_trajectory(path)


def test_extra_finalization_field_rejected(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    lines = path.read_text().splitlines()
    finalization = json.loads(lines[-1])
    finalization["unexpected"] = "tampered"
    lines[-1] = canonical_json(finalization)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(MalformedTrajectoryError, match="finalization envelope fields mismatch"):
        load_trajectory(path)


def test_missing_envelope_field_rejected(tmp_path, registry, context):
    path = golden_capture(tmp_path, registry, context)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    del header["trajectory_id"]  # required envelope field
    lines[0] = canonical_json(header)
    finalization = json.loads(lines[-1])
    del finalization["complete"]  # required envelope field
    lines[-1] = canonical_json(finalization)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(MalformedTrajectoryError, match="envelope fields mismatch"):
        load_trajectory(path)
