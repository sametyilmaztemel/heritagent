"""Gate hardening (prior independent audit of #12, residual items): strict
#11 preflight typing, setup failures before any episode, and sensitivity
state that cannot alias the caller or answer from a failed run."""

import copy
import dataclasses

import pytest

from genome.validation.errors import RecordConsistencyError
from inheritance.gate import EvidenceCoherenceError, GateInputError
from somatic.store import SomaticStore
from tests.gate.test_gate_orchestrator import _replay_with_child, build_world
from trajectory.recorder.errors import RecorderError


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


# -- review of PR #36: multi-candidate batches and remaining gaps -------------------

def test_existing_aggregate_of_the_second_candidate_fails_before_any_episode(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    orch.run(inputs[1:])  # commits b's aggregate only
    calls = len(runner.calls)
    with pytest.raises(GateInputError, match="already exist in the live store"):
        orch.run(inputs)
    assert len(runner.calls) == calls


def test_terminal_second_candidate_fails_before_any_episode(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    somatic.decide("b", 1, "rejected", ["CIG-0099"])
    with pytest.raises(RecordConsistencyError, match="'b' v1 is no longer a somatic candidate"):
        orch.run(inputs)
    assert runner.calls == []


def test_mismatched_second_candidate_fails_before_any_episode(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    forged = copy.deepcopy(inputs[1].candidate_envelope)
    forged["candidate"]["provenance"]["source_trajectory"] = "T-mine000bbbb"
    with pytest.raises(RecordConsistencyError, match="'b' v1.*differs in provenance"):
        orch.run([inputs[0], dataclasses.replace(inputs[1], candidate_envelope=forged)])
    assert runner.calls == []


@pytest.mark.parametrize("value", [1, 1.0])
def test_non_bool_on_both_sides_is_rejected(tmp_path, value):
    """result.passed and child stage_pass both 1 (or 1.0): equal, but not
    bools — strict typing on both sides, zero episodes."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    replay = inputs[0].replay_evaluation
    child = copy.deepcopy(replay.child_record)
    child["measurements"]["stage_pass"] = value
    bad_replay = dataclasses.replace(replay, _passed=value, _child_record=child)
    with pytest.raises(EvidenceCoherenceError, match="strict bool"):
        orch.run([dataclasses.replace(inputs[0], replay_evaluation=bad_replay)])
    assert runner.calls == []


def test_sensitivity_unavailable_after_a_real_journal_failure(tmp_path, monkeypatch):
    """The failure the commit docstring names: a somatic journal write error
    during the mutation phase (an OSError, not a RecordConsistencyError)."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(somatic, "decide", disk_full)
    with pytest.raises(OSError, match="disk full"):
        orch.run(inputs)
    with pytest.raises(GateInputError, match="no primary run"):
        orch.sensitivity_recomputation("a")


def test_single_store_run_is_a_dry_run_without_store_checks(tmp_path):
    """Store checks apply only when the run will commit (both stores): with
    one store the run evaluates and commits nothing, as before."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    somatic.decide("a", 1, "rejected", ["CIG-0099"])
    orch._cig_record_store = None
    outcome = orch.run(inputs)
    assert outcome.committed is False and runner.calls


def test_empty_input_needs_no_token_counter(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    orch._token_counter = None
    assert orch.run([]).results == ()


def test_commit_refuses_a_result_whose_ids_disagree_with_its_candidate(tmp_path):
    """White-box: the reference the commit checks must be the one it decides."""
    world = build_world(tmp_path, candidates=("a", "b"), commit_stores=False)
    orch, somatic, cig_store, runner, inputs, registry = world
    ra, rb = orch.run(inputs).results
    live = __import__("genome.validation.records", fromlist=["CigRecordStore"]).CigRecordStore()
    with pytest.raises(RecordConsistencyError, match="must be the same"):
        orch._commit_results([ra, dataclasses.replace(rb, _candidate_version=5)],
                             somatic_store=somatic, cig_record_store=live)
    assert somatic.state("a", 1) == "candidate" and somatic.state("b", 1) == "candidate"
    assert live.aggregates == {}


def test_terminal_first_of_two_candidates_fails_before_any_episode(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    somatic.decide("a", 1, "rejected", ["CIG-0099"])
    with pytest.raises(RecordConsistencyError, match="'a' v1 is no longer a somatic candidate"):
        orch.run(inputs)
    assert runner.calls == []


# -- review of PR #36, open item: journal state checked before any episode ----------

def test_closed_somatic_store_fails_before_any_episode(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    somatic.close()
    with pytest.raises(RecorderError, match="closed"):
        orch.run(inputs)
    assert runner.calls == []


def test_stale_somatic_handle_fails_before_any_episode(tmp_path):
    """Another handle appends to the same journal: the gate's handle is
    stale, so its step-10 appends would be refused after every episode."""
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    other = SomaticStore.open(somatic.path, registry)
    other.decide("b", 1, "rejected", ["CIG-0099"])
    with pytest.raises(RecorderError, match="stale somatic store handle"):
        orch.run(inputs[:1])
    assert runner.calls == []
    assert somatic.state("a", 1) == "candidate"


def test_dry_run_does_not_require_an_appendable_store(tmp_path):
    """One store only: nothing is committed, so a closed journal is fine."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    somatic.close()
    orch._cig_record_store = None
    assert orch.run(inputs).committed is False and runner.calls
