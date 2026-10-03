"""Recorder tests: capture, invariants, finalization, outcome attachment."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from runtime.events import RuntimeEvent
from trajectory.recorder.context import OutcomeAnnotation
from trajectory.recorder.errors import (
    RecorderError,
    StreamInvariantError,
)
from trajectory.recorder.context import TrajectoryContext
from trajectory.recorder.recorder import TrajectoryRecorder
from trajectory.storage.canonical import canonical_json, sha256_hex
from trajectory.storage.jsonl import load_trajectory
from tests.trajectory.conftest import SteppedClock, run_captured


def golden_script():
    from tests.runtime.conftest import react_action, react_final
    return [react_action("heat_object", {"object": "plate"}),
            react_final("The plate is heated.")]


def test_golden_successful_capture_schema_valid(tmp_path, registry, context):
    from tests.runtime.conftest import ToolObservation
    result, recorder, path, env, adapter = run_captured(
        tmp_path, registry, context, script=golden_script(),
        env_results={"heat_object": [ToolObservation(ok=True, content="The plate is now heated.")]})
    finalized = recorder.finalize()
    assert result.status == "finished"

    schema = json.loads(Path("trajectory/schema/trajectory-0.1.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(finalized.document)  # envelope validates

    assert finalized.status == "finished"
    import re as _re
    assert _re.match(r"^T-[0-9a-f]{12}$", finalized.trajectory_id)  # opaque, not digest-derived
    assert finalized.trajectory_id != finalized.content_sha256[:12]
    assert finalized.document["complete"] is True
    assert finalized.document["context"]["genome_id"] == "G-0f1e2d3c"
    kinds = [e["kind"] for e in finalized.document["events"]]
    assert kinds[0] == "run_started" and kinds[-1] == "finished"
    assert finalized.document["events"][-1]["data"]["answer"] == "The plate is heated."


def test_trajectory_id_and_digest_deterministic(tmp_path, registry, context):
    from tests.runtime.conftest import ToolObservation
    digests, ids = [], []
    for name in ("a", "b"):
        _, recorder, _, _, _ = run_captured(
            tmp_path / name, registry, context, script=golden_script(),
            env_results={"heat_object": [ToolObservation(ok=True, content="heated")]},
            name=name, clock=SteppedClock(), trajectory_id="T-aaaaaaaaaaaa")
        finalized = recorder.finalize()
        digests.append(finalized.content_sha256)
        ids.append(finalized.trajectory_id)
    assert digests[0] == digests[1]  # identical evidence + id -> identical digest
    assert ids == ["T-aaaaaaaaaaaa", "T-aaaaaaaaaaaa"]  # caller-supplied id stable


def test_two_independent_runs_get_distinct_ids(tmp_path, registry, context):
    """Default allocation is opaque/random: identical evidence, different ids."""
    from tests.runtime.conftest import ToolObservation
    ids = []
    for name in ("a", "b"):
        _, recorder, _, _, _ = run_captured(
            tmp_path / name, registry, context, script=golden_script(),
            env_results={"heat_object": [ToolObservation(ok=True, content="heated")]},
            name=name, clock=SteppedClock())
        ids.append(recorder.finalize().trajectory_id)
    assert ids[0] != ids[1]
    assert ids[0].startswith("T-") and ids[1].startswith("T-")


def test_outcome_change_changes_digest_but_not_identity(tmp_path, registry, context):
    from trajectory.recorder.context import OutcomeAnnotation
    from tests.runtime.conftest import ToolObservation
    _, recorder, path, _, _ = run_captured(
        tmp_path, registry, context, script=golden_script(),
        env_results={"heat_object": [ToolObservation(ok=True, content="heated")]},
        trajectory_id="T-aaaaaaaaaaaa")
    finalized = recorder.finalize()
    first_digest = finalized.content_sha256
    first_id = finalized.trajectory_id

    # a second recorder re-evaluates the same evidence with another outcome
    _, recorder2, _, _, _ = run_captured(
        tmp_path / "other", registry, context, script=golden_script(),
        env_results={"heat_object": [ToolObservation(ok=True, content="heated")]},
        trajectory_id="T-aaaaaaaaaaaa", clock=SteppedClock())
    recorder2.attach_outcome(OutcomeAnnotation(success=False, reward=0.0))
    finalized2 = recorder2.finalize()
    assert finalized2.trajectory_id == first_id         # identity unchanged
    assert finalized2.content_sha256 != first_digest    # integrity content changed


def test_incomplete_recovery_preserves_header_identity(tmp_path, registry, context):
    """Crash recovery rebuilds under the SAME opaque id from the header:
    the id survives, the digest is recomputed for the surviving prefix."""
    recorder = TrajectoryRecorder(context, tmp_path / "prefix.jsonl",
                                   trajectory_id="T-bbbbbbbbbbbb")
    # synthetic crashed run: the loop died before any terminal event
    recorder(RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.25,
                          data={"task": "t"}))
    recorder(RuntimeEvent(seq=2, kind="step_started", step=1, monotonic_s=0.5, data={}))
    recorder.finalize_incomplete("simulated crash")
    loaded = load_trajectory(tmp_path / "prefix.jsonl")
    assert loaded.trajectory_id == "T-bbbbbbbbbbbb"  # header identity preserved
    assert loaded.status == "incomplete"
    assert loaded.complete is False


def test_jsonl_bytes_are_canonical_and_identical(tmp_path, registry, context):
    from tests.runtime.conftest import ToolObservation
    files = []
    for name in ("a", "b"):
        _, recorder, path, _, _ = run_captured(
            tmp_path / name, registry, context, script=golden_script(),
            env_results={"heat_object": [ToolObservation(ok=True, content="heated")]},
            name=name, clock=SteppedClock(), trajectory_id="T-aaaaaaaaaaaa")
        recorder.finalize()
        files.append(path)
    raw_a = files[0].read_bytes()
    raw_b = files[1].read_bytes()
    assert raw_a == raw_b  # canonical serialization + same evidence + same clock
    for line in raw_a.decode().splitlines():
        record = json.loads(line)
        assert line == canonical_json(record)  # every line is canonical JSON


def test_input_mutation_after_capture_cannot_change_evidence(context, tmp_path):
    recorder = TrajectoryRecorder(context, tmp_path / "t.jsonl")
    data = {"raw_response": "original"}
    event = RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.0, data=data)
    recorder(event)
    data["raw_response"] = "tampered"       # mutate the caller's dict afterwards
    event.data["injected"] = True           # and the event object's payload
    recorder.finalize_incomplete("unit test")
    print("FILE:", (tmp_path / "t.jsonl").read_text())
    loaded = load_trajectory(tmp_path / "t.jsonl")
    assert loaded.events[0]["data"]["raw_response"] == "original"
    assert "injected" not in loaded.events[0]["data"]


def test_finalization_rules(context, tmp_path):
    recorder = TrajectoryRecorder(context, tmp_path / "t.jsonl")
    event = RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.0, data={"task": "t"})
    recorder(event)
    with pytest.raises(RecorderError, match="requires a reason"):
        recorder.finalize_incomplete("   ")
    finalized = recorder.finalize_incomplete("interrupted")
    assert finalized.status == "incomplete"
    with pytest.raises(RecorderError, match="already finalized"):
        recorder.finalize()
    with pytest.raises(RecorderError, match="after finalization"):
        recorder(RuntimeEvent(seq=2, kind="step_started", step=1, monotonic_s=1.0, data={}))
    with pytest.raises(RecorderError, match="after finalization"):
        recorder.attach_outcome(OutcomeAnnotation(success=True))


def test_finalize_incomplete_refuses_completed_stream(tmp_path, registry, context):
    from tests.runtime.conftest import ToolObservation
    _, recorder, _, _, _ = run_captured(tmp_path, registry, context, script=golden_script(),
                                        env_results={"heat_object": [ToolObservation(ok=True,
                                                                                      content="heated")]})
    with pytest.raises(RecorderError, match="terminal event"):
        recorder.finalize_incomplete("crash")  # terminal already captured: use finalize()



def test_seq_gap_rejected(context, tmp_path):
    recorder = _base_recorder(context, tmp_path)
    with pytest.raises(StreamInvariantError, match="contiguous"):
        recorder(RuntimeEvent(seq=4, kind="step_started", step=2, monotonic_s=0.75, data={}))


def test_duplicate_run_started_rejected(context, tmp_path):
    recorder = _base_recorder(context, tmp_path)
    with pytest.raises(StreamInvariantError, match="duplicate run_started"):
        recorder(RuntimeEvent(seq=3, kind="run_started", step=None, monotonic_s=0.75, data={}))


def test_timestamp_backwards_rejected(context, tmp_path):
    recorder = _base_recorder(context, tmp_path)
    with pytest.raises(StreamInvariantError, match="backwards"):
        recorder(RuntimeEvent(seq=3, kind="skills_expressed", step=1, monotonic_s=0.1, data={}))


def test_step_backwards_rejected(context, tmp_path):
    recorder = _base_recorder(context, tmp_path)
    recorder(RuntimeEvent(seq=3, kind="step_started", step=2, monotonic_s=0.75, data={}))
    with pytest.raises(StreamInvariantError, match="backwards"):
        recorder(RuntimeEvent(seq=4, kind="step_started", step=1, monotonic_s=1.0, data={}))


def test_second_terminal_rejected(context, tmp_path):
    recorder = _base_recorder(context, tmp_path)
    recorder(RuntimeEvent(seq=3, kind="finished", step=1, monotonic_s=0.75,
                          data={"answer": "done"}))
    with pytest.raises(StreamInvariantError, match="second terminal"):
        recorder(RuntimeEvent(seq=4, kind="budget_exhausted", step=1, monotonic_s=1.0, data={}))


def test_event_after_terminal_rejected(context, tmp_path):
    recorder = _base_recorder(context, tmp_path)
    recorder(RuntimeEvent(seq=3, kind="finished", step=1, monotonic_s=0.75,
                          data={"answer": "done"}))
    with pytest.raises(StreamInvariantError, match="after the terminal"):
        recorder(RuntimeEvent(seq=4, kind="skills_expressed", step=1, monotonic_s=1.0, data={}))


def test_unstarted_step_reference_rejected(context, tmp_path):
    recorder = TrajectoryRecorder(context, tmp_path / "unstarted.jsonl")
    recorder(RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.25, data={"task": "t"}))
    with pytest.raises(StreamInvariantError, match="has not started"):
        recorder(RuntimeEvent(seq=2, kind="model_called", step=1, monotonic_s=0.5,
                              data={"raw_response": "x"}))


def _base_recorder(context, tmp_path):
    """Recorder with the valid prefix: run_started + step_started(1)."""
    recorder = TrajectoryRecorder(context, tmp_path / "invariants.jsonl")
    recorder(RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.25,
                          data={"task": "t"}))
    recorder(RuntimeEvent(seq=2, kind="step_started", step=1, monotonic_s=0.5, data={}))
    return recorder


def test_outcome_attachment_round_trip(tmp_path, registry, context):
    from tests.runtime.conftest import ToolObservation
    _, recorder, path, _, _ = run_captured(tmp_path, registry, context, script=golden_script(),
                                           env_results={"heat_object": [ToolObservation(ok=True,
                                                                                         content="heated")]})
    recorder.attach_outcome(OutcomeAnnotation(success=True, reward=1.0,
                                               evaluator_id="scripted-evaluator",
                                               evaluator_version="1.0",
                                               metadata={"notes": "golden"}))
    finalized = recorder.finalize()
    assert finalized.outcome["success"] is True
    loaded = load_trajectory(path)
    assert loaded.outcome["reward"] == 1.0
    assert loaded.outcome["evaluator_id"] == "scripted-evaluator"


def test_digest_domain_sensitivity(tmp_path, registry, context):
    """Any change to context/events/finalization metadata changes the digest."""
    from tests.runtime.conftest import ToolObservation
    _, recorder, _, _, _ = run_captured(tmp_path, registry, context, script=golden_script(),
                                        env_results={"heat_object": [ToolObservation(ok=True,
                                                                                      content="heated")]})
    finalized = recorder.finalize()
    base = finalized.content_sha256

    # same events, different context (run_id) -> different digest and id
    other_fields = {k: v for k, v in vars(context).items() if k != "run_id"}
    context2 = TrajectoryContext(**other_fields, run_id="R-other")
    recorder2 = TrajectoryRecorder(context2, tmp_path / "other.jsonl")
    for record in finalized.document["events"]:
        recorder2(RuntimeEvent(seq=record["seq"], kind=record["kind"], step=record["step"],
                               monotonic_s=record["monotonic_s"], data=record["data"]))
    other = recorder2.finalize()
    assert other.content_sha256 != base
    assert other.trajectory_id != finalized.trajectory_id


def test_second_recorder_on_same_path_fails_before_writing(registry, context, tmp_path):
    from tests.runtime.conftest import ToolObservation
    _, _, path, _, _ = run_captured(tmp_path, registry, context, name="occupied",
                                     script=golden_script(),
                                     env_results={"heat_object": [ToolObservation(
                                         ok=True, content="heated")]})
    bytes_before = path.read_bytes()
    with pytest.raises(RecorderError, match="already contains data"):
        TrajectoryRecorder(context, path)  # fail closed BEFORE any write
    assert path.read_bytes() == bytes_before  # original evidence byte-identical


def test_non_finite_payload_rejected_at_capture(context, tmp_path):
    recorder = TrajectoryRecorder(context, tmp_path / "nan.jsonl")
    event = RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.25,
                         data={"task": "t", "score": float("nan")})
    with pytest.raises(RecorderError, match="not strict JSON"):
        recorder(event)
