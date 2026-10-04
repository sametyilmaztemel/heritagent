"""Mining input contract tests (issue #9 criterion 2)."""

import pytest

from traits.miner.inputs import MinerInputError, MiningBatch
from tests.miner.conftest import FAMILIES, TASK, make_pair, run_episode


def test_valid_differential_pair_accepted(tmp_path, registry):
    pair = make_pair(tmp_path, registry)
    assert pair.success.outcome["success"] is True
    assert pair.failure.outcome["success"] is False
    assert pair.task_family == "heat_and_place"


def test_incomplete_trajectory_rejected(tmp_path, registry):
    from runtime.events import RuntimeEvent
    from trajectory.recorder.context import TrajectoryContext
    from trajectory.recorder.recorder import TrajectoryRecorder
    from traits.miner.inputs import MiningPair
    success = run_episode(tmp_path, registry, name="success-ep", success=True)
    # an aborted recorder leaves an incomplete trajectory
    path = tmp_path / "incomplete.jsonl"
    recorder = TrajectoryRecorder(TrajectoryContext(genome_id="G-0f1e2d3c"), path)
    recorder(RuntimeEvent(seq=1, kind="run_started", step=None, monotonic_s=0.25,
                          data={"task": "t"}))
    recorder.finalize_incomplete("simulated crash")
    failure = load_incomplete_normalized(path)
    with pytest.raises(MinerInputError, match="incomplete"):
        MiningPair(success=success, failure=failure, task_family="heat_and_place")


def load_incomplete_normalized(path):
    from trajectory.storage.jsonl import load_trajectory
    from trajectory.normalization import normalize_loaded
    return normalize_loaded(load_trajectory(path, allow_incomplete=True))


def test_null_outcome_rejected(tmp_path, registry):
    from traits.miner.inputs import MiningPair
    success = run_episode(tmp_path, registry, name="s", success=True)
    failure = run_episode(tmp_path, registry, name="f", success=False)
    import dataclasses
    failure_no_outcome = dataclasses.replace(failure, outcome=None)
    with pytest.raises(MinerInputError, match="no generic outcome"):
        MiningPair(success=success, failure=failure_no_outcome,
                    task_family="heat_and_place")


def test_success_success_pair_rejected(tmp_path, registry):
    success_a = run_episode(tmp_path, registry, name="s1", success=True)
    success_b = run_episode(tmp_path, registry, name="s2", success=True)
    from traits.miner.inputs import MiningPair
    with pytest.raises(MinerInputError, match="failure member"):
        MiningPair(success=success_a, failure=success_b, task_family="heat_and_place")


def test_failure_failure_pair_rejected(tmp_path, registry):
    failure_a = run_episode(tmp_path, registry, name="f1", success=False)
    failure_b = run_episode(tmp_path, registry, name="f2", success=False)
    from traits.miner.inputs import MiningPair
    with pytest.raises(MinerInputError, match="success member"):
        MiningPair(success=failure_a, failure=failure_b, task_family="heat_and_place")


def test_same_trajectory_on_both_sides_rejected(tmp_path, registry):
    episode = run_episode(tmp_path, registry, name="both", success=True)
    from traits.miner.inputs import MiningPair
    with pytest.raises(MinerInputError, match="distinct trajectories"):
        MiningPair(success=episode, failure=episode, task_family="heat_and_place")


def test_runtime_finished_is_not_benchmark_success(tmp_path, registry):
    """A finished run WITHOUT an outcome annotation must not be usable as a
    success member — the miner never invents benchmark success."""
    import dataclasses
    from traits.miner.inputs import MiningPair
    success = run_episode(tmp_path, registry, name="s", success=True)
    finished_no_outcome = dataclasses.replace(success, outcome=None)
    failure = run_episode(tmp_path, registry, name="f", success=False)
    with pytest.raises(MinerInputError, match="no generic outcome"):
        MiningPair(success=finished_no_outcome, failure=failure,
                    task_family="heat_and_place")


def test_pair_context_mismatch_rejected(tmp_path, registry):
    import dataclasses
    from traits.miner.inputs import MiningPair
    success = run_episode(tmp_path, registry, name="s", success=True)
    failure = run_episode(tmp_path, registry, name="f", success=False)
    success = dataclasses.replace(
        success, provenance={**success.provenance, "benchmark": "scripted-bench"})
    other_bench = dataclasses.replace(
        failure, provenance={**failure.provenance, "benchmark": "other-bench"})
    with pytest.raises(MinerInputError, match="benchmark"):
        MiningPair(success=success, failure=other_bench, task_family="heat_and_place")


def test_batch_requires_pairs_and_valid_families(tmp_path, registry):
    from traits.miner.inputs import MiningBatch
    pair = make_pair(tmp_path, registry)
    with pytest.raises(MinerInputError, match="at least one pair"):
        MiningBatch(pairs=(), family_universe=FAMILIES)
    with pytest.raises(MinerInputError, match="outside the frozen family universe"):
        MiningBatch(pairs=(pair,), family_universe=("unknown_family",))
    batch = MiningBatch(pairs=(pair,), family_universe=FAMILIES)
    assert batch.task_families == ("heat_and_place",)
