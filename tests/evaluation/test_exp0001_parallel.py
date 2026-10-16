"""Parallel EXP-0001 episodes (issue #13 Phase B; ADR-0008): worker
processes (spawn) play the episodes, the parent stays the ledger's only
writer, and every cap counts the episodes in flight. Scripted games and
models only."""

import json

import pytest

from evaluation.exp0001.episodes import EpisodeExecutor, RunPaused
from evaluation.exp0001.ledger import CapExceeded, RunLedger
from evaluation.exp0001.lifetime import lifetime_specs, run_lifetime, run_lifetimes
from evaluation.exp0001.protocol import CAPS, MINING_SEEDS
from runtime.seed_g0 import g0_runtime_config
from tests.conftest import registry  # noqa: F401 (fixture)
from tests.evaluation.exp0001_workers import FailOnTask, GameFactory
from tests.evaluation.test_exp0001_lifetime import (
    COMMIT,
    PATTERN,
    RUN_ID,
    Clock,
    SeededModel,
    tasks,
    wins_for,
)


def parallel_executor(tmp_path, model, *, workers=2, caps=CAPS, name="ledger.jsonl",
                      pause_when=None):
    ledger = RunLedger(tmp_path / name, run_id=RUN_ID, clock=Clock(), caps=caps)
    return EpisodeExecutor(model=model, env_factory=GameFactory(), ledger=ledger,
                           trajectories_dir=tmp_path / "trajectories", run_id=RUN_ID,
                           code_commit=COMMIT, workers=workers, sleep_between_retries=False,
                           pause_when=None if pause_when is None
                           else (lambda: pause_when(ledger))), ledger


def comparable(results):
    return [{**vars(r), "resumed": None} for r in results]


def test_parallel_episodes_equal_sequential_ones_in_order(tmp_path, registry):  # noqa: F811
    config, seed = g0_runtime_config(registry), MINING_SEEDS[0]
    sequential, _ = parallel_executor(tmp_path / "seq", SeededModel(wins=wins_for(PATTERN)),
                                      workers=1)
    expected = run_lifetime(sequential, tasks(), config, seed)
    parallel, ledger = parallel_executor(tmp_path / "par", SeededModel(wins=wins_for(PATTERN)))
    try:
        results = parallel.run_many(lifetime_specs(tasks(), config, seed))
    finally:
        parallel.close()
    assert [r.key for r in results] == [r.key for r in expected]
    strip = [{**r, "trajectory_path": None} for r in comparable(expected)]
    assert [{**r, "trajectory_path": None} for r in comparable(results)] == strip
    assert ledger.episodes == 12 and ledger.in_flight == 0 and ledger.aborted_attempts == 0
    reopened, again = parallel_executor(tmp_path / "par", SeededModel())  # resumed: no episode runs
    try:
        assert comparable(reopened.run_many(lifetime_specs(tasks(), config, seed))) == \
            comparable(results)
    finally:
        reopened.close()
    assert again.episodes == 12


def test_all_lifetimes_run_as_one_batch(tmp_path, registry):  # noqa: F811
    config = g0_runtime_config(registry)
    parallel, ledger = parallel_executor(tmp_path, SeededModel())
    try:
        by_seed = run_lifetimes(parallel, tasks(2), config)
    finally:
        parallel.close()
    assert tuple(by_seed) == MINING_SEEDS
    for seed in MINING_SEEDS:
        assert [r.key for r in by_seed[seed]] == [s.key for s in
                                                  lifetime_specs(tasks(2), config, seed)]
    assert ledger.episodes == 3 * 6


def test_a_cap_stops_the_run_only_after_the_episodes_in_flight_are_recorded(
        tmp_path, registry):  # noqa: F811
    """The episode the cap would cut is never admitted; every admitted one
    finishes and is recorded, so nothing is aborted by the stop."""
    config, seed = g0_runtime_config(registry), MINING_SEEDS[0]
    parallel, ledger = parallel_executor(tmp_path, SeededModel(),
                                         caps={**CAPS, "episodes_max": 5})
    try:
        with pytest.raises(CapExceeded, match="episodes_max 5"):
            parallel.run_many(lifetime_specs(tasks(), config, seed))
    finally:
        parallel.close()
    assert ledger.state == "incomplete" and ledger.episodes == 5
    assert ledger.aborted_attempts == 0 and ledger.in_flight == 0
    reopened = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock(),
                         caps={**CAPS, "episodes_max": 5})
    assert (reopened.state, reopened.episodes, reopened.aborted_attempts) == ("incomplete", 5, 0)


def test_a_worker_failure_is_charged_and_the_rest_are_recorded(tmp_path, registry):  # noqa: F811
    config, seed = g0_runtime_config(registry), MINING_SEEDS[0]
    failing = tasks()[1].task_id
    parallel, ledger = parallel_executor(tmp_path, FailOnTask(fail_task=failing), workers=2)
    try:
        with pytest.raises(RuntimeError, match="backend died"):
            parallel.run_many(lifetime_specs(tasks(), config, seed))
    finally:
        parallel.close()
    assert ledger.state == "running" and ledger.in_flight == 0
    assert ledger.aborted_attempts >= 1  # the failed attempt is charged now, as a reload would
    reloaded = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock())
    assert (reloaded.episodes, reloaded.aborted_attempts, reloaded.model_calls) == \
        (ledger.episodes, ledger.aborted_attempts, ledger.model_calls)
    assert not any(failing in key for key in reloaded._done)


def test_run_many_needs_distinct_keys(tmp_path, registry):  # noqa: F811
    config, seed = g0_runtime_config(registry), MINING_SEEDS[0]
    parallel, _ = parallel_executor(tmp_path, SeededModel())
    spec = lifetime_specs(tasks(1), config, seed)[0]
    with pytest.raises(ValueError, match="distinct episode keys"):
        parallel.run_many([spec, spec])
    with pytest.raises(ValueError, match="positive int"):
        parallel_executor(tmp_path / "x", SeededModel(), workers=0)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_pause_records_the_episodes_in_flight_and_the_run_resumes(tmp_path, registry,  # noqa: F811
                                                                      workers):
    config, seed = g0_runtime_config(registry), MINING_SEEDS[0]
    specs = lifetime_specs(tasks(), config, seed)
    paused, ledger = parallel_executor(tmp_path, SeededModel(), workers=workers,
                                       pause_when=lambda ledger: ledger.episodes >= 4)
    try:
        with pytest.raises(RunPaused, match="paused before"):
            paused.run_many(specs)
    finally:
        paused.close()
    assert ledger.state == "running" and ledger.in_flight == 0 and ledger.aborted_attempts == 0
    assert 4 <= ledger.episodes <= 4 + workers - 1
    resumed, again = parallel_executor(tmp_path, SeededModel(), workers=workers)
    try:
        assert len(resumed.run_many(specs)) == 12
    finally:
        resumed.close()
    assert (again.episodes, again.aborted_attempts) == (12, 0)


class SteppingClock:
    """Each reading moves the clock on, so the active-time cap is crossed
    between two readings of the same admission (review of #69)."""

    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def test_a_cap_crossed_with_episodes_in_flight_records_them_before_the_stop(
        tmp_path, registry):  # noqa: F811
    config, seed = g0_runtime_config(registry), MINING_SEEDS[0]
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=SteppingClock(400.0),
                       caps={**CAPS, "wall_clock_hours_max": 1})
    executor = EpisodeExecutor(model=SeededModel(), env_factory=GameFactory(), ledger=ledger,
                               trajectories_dir=tmp_path / "trajectories", run_id=RUN_ID,
                               code_commit=COMMIT, workers=2, sleep_between_retries=False)
    try:
        with pytest.raises(CapExceeded, match="wall_clock_hours_max"):
            executor.run_many(lifetime_specs(tasks(), config, seed))
    finally:
        executor.close()
    lines = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    attempts = [r["key"] for r in lines if r["type"] == "attempt"]
    episodes = [r["key"] for r in lines if r["type"] == "episode"]
    assert attempts and sorted(attempts) == sorted(episodes)  # every admitted episode recorded
    assert lines[-1]["type"] == "incomplete" and ledger.in_flight == 0
    assert ledger.aborted_attempts == 0


def test_a_submit_failure_charges_the_admitted_attempt(tmp_path, registry, monkeypatch):  # noqa: F811
    config, seed = g0_runtime_config(registry), MINING_SEEDS[0]
    parallel, ledger = parallel_executor(tmp_path, SeededModel())
    pool = parallel._get_pool()

    def broken(*args, **kwargs):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(pool, "submit", broken)
    try:
        with pytest.raises(RuntimeError, match="cannot schedule"):
            parallel.run_many(lifetime_specs(tasks(1), config, seed))
    finally:
        monkeypatch.undo()
        parallel.close()
    reloaded = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock())
    assert (ledger.in_flight, ledger.aborted_attempts, ledger.model_calls) == \
        (0, reloaded.aborted_attempts, reloaded.model_calls) == (0, 1, 60)
