"""Deterministic evidence renderer tests (issue #9 criterion 3)."""

import json

import pytest

from trajectory.storage.canonical import canonical_json, sha256_hex
from traits.miner.evidence import render_pair_evidence
from tests.miner.conftest import TASK, make_pair


def test_render_is_canonical_and_deterministic(tmp_path, registry):
    pair = make_pair(tmp_path, registry)
    text_a, digest_a = render_pair_evidence(pair)
    text_b, digest_b = render_pair_evidence(pair)  # same evidence rendered twice
    assert text_a == text_b
    assert digest_a == digest_b
    assert digest_a == sha256_hex(text_a.encode("utf-8"))
    assert text_a == canonical_json(json.loads(text_a))  # canonical serialization


def test_identical_behavior_identical_evidence_across_ids(tmp_path, registry):
    """Identical behavioral evidence under different opaque trajectory ids
    yields IDENTICAL evidence text and hash — ids are provenance, not
    teacher input."""
    pair_a = make_pair(tmp_path / "a", registry,
                        success_trajectory_id="T-aaaa1111aaaa",
                        failure_trajectory_id="T-ffff5555ffff")
    pair_b = make_pair(tmp_path / "b", registry,
                        success_trajectory_id="T-bbbb2222bbbb",
                        failure_trajectory_id="T-eeee3333eeee")
    text_a, digest_a = render_pair_evidence(pair_a)
    text_b, digest_b = render_pair_evidence(pair_b)
    assert text_a == text_b
    assert digest_a == digest_b


def test_evidence_preserves_failures(tmp_path, registry):
    pair = make_pair(tmp_path, registry)
    text, _ = render_pair_evidence(pair)
    document = json.loads(text)
    failed = document["failed_trajectory"]
    assert failed["label"] == "failure"
    assert failed["outcome"]["success"] is False
    assert failed["steps"][0]["tool_calls"][0]["ok"] is False
    assert failed["steps"][0]["tool_calls"][0]["result_content"] == "heater offline"
    assert document["successful_trajectory"]["label"] == "success"


def test_evidence_excludes_provenance_and_trajectory_ids(tmp_path, registry):
    """Opaque trajectory ids carry no skill-abstraction information and stay
    in provenance/mining records — never in the teacher prompt."""
    pair = make_pair(tmp_path, registry)
    text, _ = render_pair_evidence(pair)
    for forbidden in ("code_commit", "backend_version", "benchmark", "split",
                       "evaluator_id", "evaluator_version",
                       pair.success.trajectory_id, pair.failure.trajectory_id,
                       "R-success-ep", "R-failure-ep"):
        assert forbidden not in text, f"{forbidden} leaked into teacher evidence"
    # behavioral evidence and outcome labels ARE included
    assert pair.success.task in text
    assert '"success":true' in text and '"success":false' in text


def test_evidence_includes_task_family_and_outcomes(tmp_path, registry):
    pair = make_pair(tmp_path / "fam", registry, family="clean_surface")
    text, _ = render_pair_evidence(pair)
    document = json.loads(text)
    assert document["task_family"] == "clean_surface"
    assert document["kind"] == "heritagent-differential-evidence"


# render_pair_evidence of _truncated_reply_pair, computed before model_called
# carried finish_reason
EVIDENCE_SHA256_OF_A_TRUNCATED_REPLY_PAIR = \
    "8734442d09d089770ddd2e9700d13e17cd52b0b11c1b822274cb7f4252fbbf74"


def _truncated_reply_pair(tmp_path, registry):
    """A pair whose successful episode starts with a reply cut at max_tokens
    (finish_reason 'length'), recorded and normalized as in a run."""
    from runtime.loop import Budgets, ToolObservation, run_agent
    from runtime.model_adapters import ScriptedAdapter
    from runtime.seed_g0 import g0_runtime_config
    from tests.miner.conftest import final, react, run_episode
    from tests.runtime.conftest import ScriptedEnv
    from traits.miner.inputs import MiningPair
    from trajectory.normalization import load_normalized
    from trajectory.recorder.context import OutcomeAnnotation, TrajectoryContext
    from trajectory.recorder.recorder import TrajectoryRecorder

    adapter = ScriptedAdapter([])
    adapter.enqueue_text("Thought: I should heat the pla", finish_reason="length",
                         completion_tokens=64)
    adapter.enqueue_text(react("heat_object", {"object": "plate"}))
    adapter.enqueue_text(final("The plate is heated."))
    path = tmp_path / "truncated.jsonl"
    recorder = TrajectoryRecorder(TrajectoryContext(genome_id="G-0f1e2d3c", run_id="R-t"),
                                  path, trajectory_id="T-aaaa1111aaaa")
    run_agent(config=g0_runtime_config(registry), model=adapter,
              env=ScriptedEnv({"heat_object": [ToolObservation(ok=True, content="heated")]}),
              task=TASK, budgets=Budgets(max_steps=4, max_total_retries=10,
                                         max_tokens_per_request=64),
              hooks=(recorder,), sleep=lambda _: None)
    recorder.attach_outcome(OutcomeAnnotation(success=True, reward=1.0,
                                              evaluator_id="scripted-evaluator",
                                              evaluator_version="1.0"))
    recorder.finalize()
    failure = run_episode(tmp_path, registry, name="failure-ep", success=False,
                          trajectory_id="T-ffff5555ffff")
    return MiningPair(success=load_normalized(path), failure=failure,
                      task_family="heat_and_place"), path


def test_a_truncated_reply_leaves_the_teacher_evidence_byte_for_byte_unchanged(tmp_path,
                                                                                registry):
    """The model_called event's finish_reason is a record for the operator,
    never teacher input: the evidence of this scripted pair is pinned to its
    bytes before the field existed."""
    pair, _ = _truncated_reply_pair(tmp_path, registry)
    text, digest = render_pair_evidence(pair)
    assert "finish_reason" not in text
    assert digest == EVIDENCE_SHA256_OF_A_TRUNCATED_REPLY_PAIR


def test_finish_reason_is_recorded_but_leaves_the_normalized_trajectory_unchanged(tmp_path,
                                                                                    registry):
    import copy

    from trajectory.normalization import normalize_loaded, normalize_trajectory
    from trajectory.storage.jsonl import load_trajectory

    _, path = _truncated_reply_pair(tmp_path, registry)
    loaded = load_trajectory(path)
    calls = [e for e in loaded.events if e["kind"] == "model_called"]
    assert [c["data"]["finish_reason"] for c in calls] == ["length", "stop", "stop"]
    stripped = copy.deepcopy(list(loaded.events))
    for event in stripped:
        event["data"].pop("finish_reason", None)
    assert normalize_loaded(loaded) == normalize_trajectory(
        trajectory_id=loaded.trajectory_id, context=loaded.context, events=stripped,
        status=loaded.status, outcome=loaded.outcome)
