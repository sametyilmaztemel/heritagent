"""Gate hardening (prior independent audit of #12, residual items): strict
#11 preflight typing, setup failures before any episode, and sensitivity
state that cannot alias the caller or answer from a failed run."""

import copy
import dataclasses

import pytest

from genome.validation.errors import RecordConsistencyError
from inheritance.gate import EvidenceCoherenceError, GateInputError
from tests.gate.test_gate_orchestrator import _replay_with_child, build_world


@pytest.mark.parametrize("field,bad_value", [
    ("stage_pass", 1),      # True == 1: loose equality must not validate
    ("stage_pass", 1.0),
    ("seed", 11.0),         # 11 == 11.0 and the schema calls 11.0 an integer
])
def test_stage11_child_fields_are_strictly_typed(tmp_path, field, bad_value):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    replay = inputs[0].replay_evaluation
    child = copy.deepcopy(replay.child_record)
    if field == "seed":
        child["seed"] = bad_value
    else:
        child["measurements"][field] = bad_value
    bad = dataclasses.replace(inputs[0], replay_evaluation=_replay_with_child(replay, child))
    calls_before = len(runner.calls)
    with pytest.raises(EvidenceCoherenceError, match="strict"):
        orch.run([bad])
    assert len(runner.calls) == calls_before


def test_missing_token_counter_fails_before_any_episode(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    orch._token_counter = None
    with pytest.raises(GateInputError, match="TokenCounter"):
        orch.run(inputs)
    assert runner.calls == []


def test_existing_aggregate_id_fails_before_any_episode(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    orch.run(inputs)
    calls_after_first = len(runner.calls)
    with pytest.raises(GateInputError, match="already exist in the live store"):
        orch.run(inputs)
    assert len(runner.calls) == calls_after_first


def test_terminal_somatic_candidate_fails_before_any_episode(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    somatic.decide("a", 1, "rejected", ["CIG-0099"])
    with pytest.raises(RecordConsistencyError, match="no longer a somatic candidate"):
        orch.run(inputs)
    assert runner.calls == []


def test_mismatched_stored_candidate_fails_before_any_episode(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    forged = copy.deepcopy(inputs[0].candidate_envelope)
    forged["candidate"]["provenance"]["source_trajectory"] = "T-mine000bbbb"
    with pytest.raises(RecordConsistencyError, match="differs in provenance"):
        orch.run([dataclasses.replace(inputs[0], candidate_envelope=forged)])
    assert runner.calls == []


def test_sensitivity_does_not_alias_the_callers_list(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"), commit_stores=False,
                        replay_pass_for={"a": True, "b": False})
    orch, somatic, cig_store, runner, inputs, registry = world
    orch.run(inputs)
    before = orch.sensitivity_recomputation("a")
    inputs.reverse()
    inputs.clear()
    assert orch.sensitivity_recomputation("a") == before


def test_sensitivity_unavailable_after_a_failed_run(tmp_path):
    world = build_world(tmp_path, candidates=("a",), commit_stores=False)
    orch, somatic, cig_store, runner, inputs, registry = world
    orch.run(inputs)
    orch.sensitivity_recomputation("a")
    orch._token_counter = None
    with pytest.raises(GateInputError):
        orch.run(inputs)
    with pytest.raises(GateInputError, match="no primary run"):
        orch.sensitivity_recomputation("a")


def test_sensitivity_unavailable_after_a_failed_commit(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world

    def failing_commit(results, *, somatic_store, cig_record_store):
        raise RecordConsistencyError("simulated journal failure")

    orch._commit_results = failing_commit
    with pytest.raises(RecordConsistencyError, match="simulated"):
        orch.run(inputs)
    with pytest.raises(GateInputError, match="no primary run"):
        orch.sensitivity_recomputation("a")
