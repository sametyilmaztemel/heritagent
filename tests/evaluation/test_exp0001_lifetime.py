"""The G0 lifetime, mining pairs and T_reg (issue #13 Phase A; ADR-0007 E1,
E3), on a scripted model and scripted ALFWorld games: no GPU, no ALFWorld."""

import pytest

from evaluation.benchmarks.alfworld.splits import ALFWORLD_TASK_TYPES
from evaluation.exp0001.episodes import (
    EVALUATOR_ID,
    EpisodeExecutor,
    EpisodeResult,
    EpisodeSpec,
    trajectory_id_for,
)
from evaluation.exp0001.ledger import CapExceeded, RunLedger
from evaluation.exp0001.lifetime import (
    LifetimeError,
    LifetimeTask,
    derive_t_reg,
    eligible_count,
    lifetime_key,
    lifetime_specs,
    mining_batch,
    mining_pairs,
    run_lifetime,
    t_mine_tasks,
    t_reg_rank,
    t_reg_record,
)
from evaluation.exp0001.protocol import CAPS, MINING_SEEDS, attempt_sampling_seed
from runtime.model_adapters import ScriptedAdapter
from runtime.model_adapters.base import GenerationResult
from runtime.policies.planner_react import STOP_SEQUENCES
from runtime.seed_g0 import g0_runtime_config
from runtime.tool_adapters.alfworld import BANNER, AlfworldEnvironment
from tests.conftest import registry  # noqa: F401 (fixture)
from tests.evaluation.test_exp0001_artifacts import load
from tests.runtime.conftest import react_action, react_final
from trajectory.normalization import load_normalized

RUN_ID = "EXP-0001-test-run"
COMMIT = "a" * 40


class Game:
    """One scripted ALFWorld game: the command ``win`` wins it."""

    def __init__(self, game_id):
        self.game_id = game_id
        self.closed = False

    def reset(self):
        return (f"{BANNER}\n\nYou are in a room.\n\nYour task is to: solve {self.game_id}.",
                {"won": False, "admissible_commands": ["look", "win"]})

    def step(self, command):
        won = command == "win"
        return ("You won." if won else "Nothing happens."), float(won), won, {
            "won": won, "admissible_commands": [] if won else ["look", "win"]}


class SeededModel(ScriptedAdapter):
    """Wins the episodes whose sampling seed is in ``wins``: it sends
    ``win``; otherwise it gives a final answer without acting (a failure,
    although the runtime status is ``finished``)."""

    def __init__(self, wins=(), fail_at_call=None):
        super().__init__()
        self.wins = set(wins)
        self.calls = 0
        self.fail_at_call = fail_at_call
        self.settings = []

    def generate(self, messages, settings):
        self.calls += 1
        self.settings.append(settings)
        if self.calls == self.fail_at_call:
            raise RuntimeError("backend died")
        text = react_action("act", {"command": "win"}) if settings.seed in self.wins \
            else react_final("I am done.")
        return GenerationResult(text=text, finish_reason="stop")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.001
        return self.now


def tasks(n=4):
    """n tasks across the families, listed in game_id order."""
    return tuple(LifetimeTask(f"g{i:02d}-{ALFWORLD_TASK_TYPES[i % 6]}/trial_T0",
                              ALFWORLD_TASK_TYPES[i % 6], f"{i:064x}") for i in range(n))


def wins_for(pattern, seed=MINING_SEEDS[0]):
    """pattern: per task, a string of S/F per attempt."""
    return {attempt_sampling_seed(seed, task.task_id, a)
            for task, attempts in zip(tasks(len(pattern)), pattern)
            for a, outcome in enumerate(attempts) if outcome == "S"}


def executor(tmp_path, model, *, caps=CAPS, games=None):
    games = games if games is not None else {}

    def env_factory(split, game_id):
        games[game_id] = Game(game_id)
        return AlfworldEnvironment(games[game_id], game_id=game_id)

    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock(), caps=caps)
    return EpisodeExecutor(model=model, env_factory=env_factory, ledger=ledger,
                           trajectories_dir=tmp_path / "trajectories", run_id=RUN_ID,
                           code_commit=COMMIT, clock=Clock(), sleep=lambda _: None), ledger


PATTERN = ("SFS", "FFS", "SSS", "FFF")


def test_the_specs_follow_the_protocol(registry):  # noqa: F811
    config = g0_runtime_config(registry)
    seed = MINING_SEEDS[1]
    specs = lifetime_specs(tuple(reversed(tasks())), config, seed)
    assert [s.key for s in specs] == [lifetime_key(seed, t.task_id, a)
                                      for t in tasks() for a in range(3)]  # game_id order
    for spec in specs:
        attempt = spec.tags["attempt"]
        assert spec.settings.temperature == 0.7
        assert spec.settings.max_tokens == 512
        assert spec.settings.stop == STOP_SEQUENCES
        assert spec.settings.seed == attempt_sampling_seed(seed, spec.task_id, attempt)
        assert spec.split == "train"
    with pytest.raises(LifetimeError, match="pre-registered mining seed"):
        lifetime_specs(tasks(), config, 7)


def test_the_committed_t_mine_gives_180_episodes_per_seed(registry):  # noqa: F811
    t_mine = t_mine_tasks(load("EXP-0001-splits.json"))
    assert len(t_mine) == 60 and [t.task_id for t in t_mine] == sorted(t.task_id for t in t_mine)
    assert len(lifetime_specs(t_mine, g0_runtime_config(registry), MINING_SEEDS[0])) == 180


def test_a_lifetime_gives_first_success_first_failure_pairs(tmp_path, registry):  # noqa: F811
    seed = MINING_SEEDS[0]
    model = SeededModel(wins=wins_for(PATTERN))
    run, ledger = executor(tmp_path, model)
    results = run_lifetime(run, tasks(), g0_runtime_config(registry), seed)
    assert [r.success for r in results] == [c == "S" for p in PATTERN for c in p]
    assert model.calls == 12 and ledger.episodes == 12 and ledger.model_calls == 12
    assert all(s.temperature == 0.7 for s in model.settings)

    pairs = mining_pairs(results, tasks(), seed)
    assert len(pairs) == 2  # SSS and FFF give none
    first, second = pairs
    t = tasks()
    assert first.success.trajectory_id == trajectory_id_for(RUN_ID, lifetime_key(seed, t[0].task_id, 0))
    assert first.failure.trajectory_id == trajectory_id_for(RUN_ID, lifetime_key(seed, t[0].task_id, 1))
    assert second.success.trajectory_id == trajectory_id_for(RUN_ID, lifetime_key(seed, t[1].task_id, 2))
    assert second.failure.trajectory_id == trajectory_id_for(RUN_ID, lifetime_key(seed, t[1].task_id, 0))
    assert first.task_family == t[0].family
    batch = mining_batch(results, tasks(), seed)
    assert batch.family_universe == tuple(sorted(ALFWORLD_TASK_TYPES))


def test_success_is_alfworld_won_not_the_runtime_status(tmp_path, registry):  # noqa: F811
    seed = MINING_SEEDS[0]
    run, _ = executor(tmp_path, SeededModel(wins=wins_for(("SF",))))
    results = run_lifetime(run, tasks(1), g0_runtime_config(registry), seed)
    won, lost = results[0], results[1]
    assert (won.status, won.success) == ("environment_terminal", True)
    assert (lost.status, lost.success) == ("finished", False)  # a final answer is not a win
    trajectory = load_normalized(lost.trajectory_path)
    assert trajectory.outcome["success"] is False
    assert trajectory.outcome["evaluator_id"] == EVALUATOR_ID
    assert trajectory.provenance["benchmark"] == "ALFWorld"
    assert trajectory.provenance["split"] == "train"


def test_a_resumed_lifetime_runs_nothing_twice(tmp_path, registry):  # noqa: F811
    seed = MINING_SEEDS[0]
    config = g0_runtime_config(registry)
    first, _ = executor(tmp_path, SeededModel(wins=wins_for(PATTERN)))
    before = run_lifetime(first, tasks(), config, seed)
    model = SeededModel(wins=wins_for(PATTERN))
    again, ledger = executor(tmp_path, model)  # the same ledger file, reopened
    after = run_lifetime(again, tasks(), config, seed)
    assert model.calls == 0 and ledger.episodes == 12
    assert all(r.resumed for r in after)
    strip = [{**vars(r), "resumed": None} for r in before]
    assert strip == [{**vars(r), "resumed": None} for r in after]
    assert mining_pairs(after, tasks(), seed) == mining_pairs(before, tasks(), seed)


def test_a_crashed_episode_is_not_counted_and_reruns_cleanly(tmp_path, registry):  # noqa: F811
    seed = MINING_SEEDS[0]
    config = g0_runtime_config(registry)
    games = {}
    run, ledger = executor(tmp_path, SeededModel(wins=wins_for(PATTERN), fail_at_call=5),
                           games=games)
    with pytest.raises(RuntimeError, match="backend died"):
        run_lifetime(run, tasks(), config, seed)
    assert ledger.episodes == 4
    rerun, ledger = executor(tmp_path, SeededModel(wins=wins_for(PATTERN)))
    results = run_lifetime(rerun, tasks(), config, seed)
    assert ledger.episodes == 12
    assert [r.resumed for r in results] == [True] * 4 + [False] * 8
    for result in results:
        assert load_normalized(result.trajectory_path).status != "incomplete"


def test_a_hard_cap_stops_the_lifetime(tmp_path, registry):  # noqa: F811
    run, ledger = executor(tmp_path, SeededModel(), caps={**CAPS, "episodes_max": 5})
    with pytest.raises(CapExceeded, match="episodes_max 5"):
        run_lifetime(run, tasks(), g0_runtime_config(registry), MINING_SEEDS[0])
    assert ledger.state == "incomplete" and ledger.episodes == 5


def test_a_lifetime_without_a_pair_fails_closed(tmp_path, registry):  # noqa: F811
    seed = MINING_SEEDS[0]
    run, _ = executor(tmp_path, SeededModel(wins=wins_for(("SSS", "FFF"))))
    results = run_lifetime(run, tasks(2), g0_runtime_config(registry), seed)
    with pytest.raises(LifetimeError, match="no mining pair"):
        mining_batch(results, tasks(2), seed)


def fake(seed, task, attempt, success, family=None):
    key = lifetime_key(seed, task.task_id, attempt)
    return EpisodeResult(key=key, task_id=task.task_id, family=family or task.family,
                         success=success, trajectory_id=trajectory_id_for(RUN_ID, key),
                         trajectory_path=f"/nowhere/{key}", model_calls=1, steps=1,
                         invalid_commands=0, status="finished", resumed=False)


def lifetimes(task_list, solved):
    """solved(seed, task_index, attempt) -> bool"""
    return {seed: [fake(seed, t, a, solved(seed, i, a)) for i, t in enumerate(task_list)
                   for a in range(3)] for seed in MINING_SEEDS}


def test_incomplete_or_inconsistent_lifetimes_fail_closed(tmp_path):
    seed = MINING_SEEDS[0]
    task_list = tasks()
    results = lifetimes(task_list, lambda s, i, a: a == 0)[seed]
    with pytest.raises(LifetimeError, match="not complete and exact"):
        mining_pairs(results[:-1], task_list, seed)
    with pytest.raises(LifetimeError, match="duplicates"):
        mining_pairs(results[:-1] + results[:1], task_list, seed)
    wrong = results[:-1] + [fake(seed, task_list[-1], 2, False, family="other")]
    with pytest.raises(LifetimeError, match="family"):
        mining_pairs(wrong, task_list, seed)


def test_t_reg_is_one_keyed_draw_with_the_earliest_success_as_baseline():
    task_list = tasks(50)
    # task i is solved only in seed MINING_SEEDS[i % 3], at attempt i % 3;
    # task 1 is also solved in the third seed's attempt 0 (a later seed)
    def solved(seed, i, attempt):
        return (seed == MINING_SEEDS[i % 3] and attempt == i % 3) or \
            (i == 1 and seed == MINING_SEEDS[2] and attempt == 0)
    allocation = derive_t_reg(lifetimes(task_list, solved), task_list)
    ids = [t.task_id for t in allocation.tasks]
    expected = sorted(sorted((t.task_id for t in task_list), key=t_reg_rank)[:40])
    assert ids == expected
    for treg in allocation.tasks:
        i = next(k for k, t in enumerate(task_list) if t.task_id == treg.task_id)
        seed, attempt = MINING_SEEDS[i % 3], i % 3  # seed order first: task 1 keeps seed 23
        assert treg.payload["baseline_seed"] == seed
        assert treg.payload["baseline_attempt"] == attempt
        assert treg.baseline_trajectory_id == trajectory_id_for(
            RUN_ID, lifetime_key(seed, treg.task_id, attempt))
    record = t_reg_record(allocation, eligible_count(lifetimes(task_list, solved)))
    assert record["eligible"] == 50 and len(record["tasks"]) == 40
    assert record == t_reg_record(derive_t_reg(lifetimes(task_list, solved), task_list), 50)


def test_fewer_than_40_solved_tasks_fail_closed():
    task_list = tasks(50)
    with pytest.raises(LifetimeError, match="only 39 T_mine tasks"):
        derive_t_reg(lifetimes(task_list, lambda s, i, a: i < 39), task_list)


def test_t_reg_needs_every_seeds_lifetime():
    task_list = tasks(50)
    every = lifetimes(task_list, lambda s, i, a: True)
    partial = {seed: every[seed] for seed in MINING_SEEDS[:2]}
    with pytest.raises(LifetimeError, match="every mining seed"):
        derive_t_reg(partial, task_list)
    with pytest.raises(LifetimeError, match="not complete and exact"):
        derive_t_reg({**every, MINING_SEEDS[2]: every[MINING_SEEDS[2]][1:]}, task_list)


def test_an_episode_key_runs_once_per_ledger(tmp_path, registry):  # noqa: F811
    model = SeededModel()
    run, ledger = executor(tmp_path, model)
    spec = EpisodeSpec(key="pilot/x", task_id=tasks(1)[0].task_id, family=tasks(1)[0].family,
                       split="train", config=g0_runtime_config(registry),
                       settings=lifetime_specs(tasks(1), g0_runtime_config(registry),
                                               MINING_SEEDS[0])[0].settings)
    first = run.run(spec)
    second = run.run(spec)
    assert model.calls == 1 and not first.resumed and second.resumed
    assert ledger.done_record("pilot/x")["outcome"]["trajectory_id"] == first.trajectory_id


class StepLimitGame(Game):
    """Ends (done) on the first step without a win: ALFWorld's step limit."""

    def step(self, command):
        return "Nothing happens.", 0.0, True, {"won": False, "admissible_commands": []}


def test_an_episode_that_ends_without_a_win_is_a_failure(tmp_path, registry):  # noqa: F811
    """ADR-0003 D6: success is ALFWorld's won, not the environment ending."""
    seed = MINING_SEEDS[0]
    model = SeededModel(wins=wins_for(("S",)))  # the model sends "win" ...
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock())
    run = EpisodeExecutor(
        model=model, env_factory=lambda split, g: AlfworldEnvironment(StepLimitGame(g), game_id=g),
        ledger=ledger, trajectories_dir=tmp_path / "t", run_id=RUN_ID, code_commit=COMMIT,
        clock=Clock(), sleep=lambda _: None)
    (result,) = run_lifetime(run, tasks(1), g0_runtime_config(registry), seed)[:1]
    assert result.status == "environment_terminal"  # ... but the game ends unwon
    assert result.success is False
    assert ledger.done_record(result.key)["outcome"]["success"] is False
    assert load_normalized(result.trajectory_path).outcome["success"] is False


def test_a_resumed_episode_must_carry_the_settings_it_ran_with(tmp_path, registry):  # noqa: F811
    from dataclasses import replace
    from evaluation.exp0001.ledger import LedgerError
    run, _ = executor(tmp_path, SeededModel())
    spec = lifetime_specs(tasks(1), g0_runtime_config(registry), MINING_SEEDS[0])[0]
    run.run(spec)
    with pytest.raises(LedgerError, match="recorded with settings"):
        run.run(replace(spec, settings=replace(spec.settings, seed=spec.settings.seed + 1)))


def test_the_baseline_is_the_earliest_attempt_within_a_seed_and_40_suffice():
    task_list = tasks(40)
    # every task: successes at attempts 0 and 2 of the second seed only
    allocation = derive_t_reg(lifetimes(task_list, lambda s, i, a: s == MINING_SEEDS[1] and a != 1),
                              task_list)
    assert len(allocation.tasks) == 40  # exactly 40 eligible: not a shortfall
    assert {t.payload["baseline_attempt"] for t in allocation.tasks} == {0}
    assert {t.payload["baseline_seed"] for t in allocation.tasks} == {MINING_SEEDS[1]}
    with pytest.raises(LifetimeError, match="new experiment revision"):
        derive_t_reg(lifetimes(task_list, lambda s, i, a: i < 39), task_list)


@pytest.mark.parametrize("field,value", [("temperature", 0.0), ("max_tokens", 256),
                                         ("stop", ("\nStop:",))])
def test_every_recorded_setting_is_checked_on_resume(tmp_path, registry, field, value):  # noqa: F811
    from dataclasses import replace
    from evaluation.exp0001.ledger import LedgerError
    run, _ = executor(tmp_path, SeededModel())
    spec = lifetime_specs(tasks(1), g0_runtime_config(registry), MINING_SEEDS[0])[0]
    run.run(spec)
    with pytest.raises(LedgerError, match="recorded with settings"):
        run.run(replace(spec, settings=replace(spec.settings, **{field: value})))


def test_a_key_admitted_twice_in_one_process_charges_the_first_attempt(tmp_path):
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock())
    assert ledger.admit("e0")  # the episode raised; the same process tries again
    assert ledger.admit("e0")
    ledger.record("e0", model_calls=2, outcome={})
    worst = 60
    assert (ledger.aborted_attempts, ledger.model_calls) == (1, worst + 2)
    reloaded = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock())
    assert (reloaded.aborted_attempts, reloaded.model_calls) == (1, worst + 2)


def test_a_retry_in_one_process_is_cap_checked_with_the_aborted_charge(tmp_path):
    from evaluation.exp0001.protocol import CAPS
    caps = {**CAPS, "model_calls_max": 100}
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock(), caps=caps)
    assert ledger.admit("e0")  # 60 reserved; the episode raises
    with pytest.raises(CapExceeded, match="model_calls_max"):
        ledger.admit("e0")  # 60 charged + 60 more > 100, as a reload decides
    assert ledger.state == "incomplete"
    reloaded = RunLedger(path, run_id=RUN_ID, clock=Clock(), caps=caps)
    assert (reloaded.state, reloaded.aborted_attempts, reloaded.model_calls) == ("incomplete", 1, 60)


def test_a_mining_retry_in_one_process_charges_the_first_batch(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock())
    assert ledger.admit_calls("mining/11", 47)  # the teacher raised
    assert ledger.admit_calls("mining/11", 47)
    ledger.record_calls("mining/11", model_calls=47)
    assert (ledger.aborted_attempts, ledger.model_calls) == (1, 94)
    reloaded = RunLedger(path, run_id=RUN_ID, clock=Clock())
    assert (reloaded.aborted_attempts, reloaded.model_calls) == (1, 94)


def test_a_mining_retry_and_an_episode_cap_see_the_aborted_charge(tmp_path):
    from evaluation.exp0001.protocol import CAPS
    caps = {**CAPS, "model_calls_max": 90}
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock(), caps=caps)
    assert ledger.admit_calls("mining/11", 47)  # the teacher raised
    with pytest.raises(CapExceeded, match="model_calls_max"):
        ledger.admit_calls("mining/11", 47)  # 47 charged + 47 > 90, as a reload decides
    reloaded = RunLedger(path, run_id=RUN_ID, clock=Clock(), caps=caps)
    assert (ledger.aborted_attempts, ledger.model_calls) == (1, 47) == \
        (reloaded.aborted_attempts, reloaded.model_calls)

    caps = {**CAPS, "episodes_max": 1}
    path = tmp_path / "episodes.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock(), caps=caps)
    assert ledger.admit("e0")  # raises: the executor aborts the attempt
    ledger.abort("e0")
    assert ledger.admit("e1")
    ledger.record("e1", model_calls=3, outcome={})
    with pytest.raises(CapExceeded, match="episodes_max"):
        ledger.admit("e0")
    reloaded = RunLedger(path, run_id=RUN_ID, clock=Clock(), caps=caps)
    assert (ledger.aborted_attempts, ledger.model_calls) == (1, 63) == \
        (reloaded.aborted_attempts, reloaded.model_calls)
