"""Stage-4A regression + stage-4B overhead tests (issue #12 sections 5-6)."""

import pytest

from genome.validation.registry import TraitRegistry
from inheritance.regression import (
    RegressionAllocation,
    RegressionEvaluator,
    TRegTask,
    drop_within,
    measure_expression_overhead,
    paired_drop_within,
    run_regression_baseline,
)
from inheritance.runner import RunnerContractError
from runtime.expression import RuntimeSkill
from runtime.loop import Budgets
from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope

GENE_ID = "look_before_heat_v1"
MINING_SEED = 11


class ScriptedRegRunner:
    """Stage-4A episode runner: with the trait the first ``successes`` tasks
    succeed; without it (the shared G0 run) the first ``without``."""

    def __init__(self, successes: int, without: int = 40):
        self.successes = successes
        self.without = without
        self.calls: list = []
        self.received_settings: list = []

    def run_episode(self, task, config, budgets, condition, settings=None):
        from inheritance.runner import EpisodeOutcome
        self.calls.append((task.task_id, condition))
        self.received_settings.append(settings)
        index = int(task.task_id.rsplit("-", 1)[-1])
        success = index < (self.successes if condition == "with_trait" else self.without)
        return EpisodeOutcome(
            task_id=task.task_id, condition=condition, success=success,
            expressed_gene_ids=(GENE_ID,) if condition == "with_trait" else (),
            status="finished",
            trajectory_id="T-" + format(index, "04d") + ("bbbb" if condition == "with_trait"
                                                          else "cccc"),
            effective_settings=settings)


def make_treg(count=40):
    return RegressionAllocation(tasks=tuple(
        TRegTask(task_id="treg-" + str(i), family="heat_and_place",
                  baseline_trajectory_id="T-" + format(i, "04d") + "aaaa")
        for i in range(count)))


BUDGETS = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)


def make_evaluator(tmp_path, registry, base_config, envelope, runner,
                    mining_seed=MINING_SEED):
    return RegressionEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        runner=runner, budgets=BUDGETS, parent_cig_id="CIG-0007", mining_seed=mining_seed)


def make_baseline(base_config, runner, allocation=None, mining_seed=MINING_SEED):
    return run_regression_baseline(base_config=base_config, runner=runner, budgets=BUDGETS,
                                   mining_seed=mining_seed,
                                   allocation=allocation or make_treg())


def evaluate(tmp_path, registry, base_config, envelope, successes, without=40):
    runner = ScriptedRegRunner(successes=successes, without=without)
    baseline = make_baseline(base_config, runner)
    result = make_evaluator(tmp_path, registry, base_config, envelope,
                            runner).evaluate(make_treg(), baseline)
    return result, runner


def test_regression_39_of_40_passes(tmp_path, registry, base_config, envelope):
    result, _ = evaluate(tmp_path, registry, base_config, envelope, 39)
    assert (result.successes_with, result.successes_without) == (39, 40)
    assert result.regression_drop == pytest.approx(0.025)
    assert result.passed is True  # 0.025 <= tau_r 0.03


def test_regression_38_of_40_fails(tmp_path, registry, base_config, envelope):
    result, _ = evaluate(tmp_path, registry, base_config, envelope, 38)
    assert result.regression_drop == pytest.approx(0.05)
    assert result.passed is False  # 0.05 > tau_r 0.03


@pytest.mark.parametrize("with_trait,without,passed", [
    (12, 13, True),    # a net loss of one task
    (11, 13, False),   # two
    (20, 10, True),    # a net gain passes
    (0, 0, True),      # G0 solves nothing at temperature 0: no loss is possible
    (40, 40, True),
])
def test_regression_is_paired_against_the_shared_g0_run(tmp_path, registry, base_config,
                                                        envelope, with_trait, without, passed):
    """ADR-0009 D1: the drop is G0's successes minus the trait's, over 40."""
    result, _ = evaluate(tmp_path, registry, base_config, envelope, with_trait, without)
    assert result.passed is passed
    assert result.regression_drop == pytest.approx((without - with_trait) / 40)
    assert result.baseline["successes_without"] == without
    assert result.baseline["shared_across_candidates"] is True


def test_regression_drop_is_exact_at_the_boundary(tmp_path, registry, base_config,
                                                  envelope):
    """2/40 is exactly 0.05: the persisted drop is the nearest float (not
    1 - 38/40 = 0.050000000000000044) and the denominator is recorded."""
    result, _ = evaluate(tmp_path, registry, base_config, envelope, 38)
    assert result.task_count == 40
    assert result.regression_drop == 0.05


@pytest.mark.parametrize("successes,tau,expected", [
    (40, 0.02, True), (39, 0.02, False),
    (39, 0.03, True), (38, 0.03, False),
    (38, 0.05, True), (37, 0.05, False),
])
def test_drop_within_is_exact(successes, tau, expected):
    assert drop_within(successes, 40, tau) is expected
    assert paired_drop_within(successes, 40, 40, tau) is expected


@pytest.mark.parametrize("with_trait,without,tau,expected", [
    (17, 19, 0.05, True), (16, 19, 0.05, False), (5, 6, 0.03, True), (4, 6, 0.03, False),
])
def test_paired_drop_within_is_exact(with_trait, without, tau, expected):
    assert paired_drop_within(with_trait, without, 40, tau) is expected


def test_forty_calls_per_candidate_and_one_shared_g0_run(tmp_path, registry, base_config,
                                                         envelope):
    runner = ScriptedRegRunner(successes=40)
    baseline = make_baseline(base_config, runner)
    for _ in range(2):  # two candidates read the same G0 run
        make_evaluator(tmp_path, registry, base_config, envelope,
                       runner).evaluate(make_treg(), baseline)
    conditions = [c[1] for c in runner.calls]
    assert conditions.count("without_trait") == 40 and conditions.count("with_trait") == 80
    assert conditions[:40] == ["without_trait"] * 40


def test_locked_settings_passed_to_runner(tmp_path, registry, base_config, envelope):
    _, runner = evaluate(tmp_path, registry, base_config, envelope, 40)
    assert len({(s.temperature, s.max_tokens, s.seed, tuple(s.stop))
                for s in runner.received_settings}) == 1  # both conditions alike
    settings = runner.received_settings[0]
    assert settings.temperature == 0.0
    assert settings.max_tokens == 64  # Budgets.max_tokens_per_request
    assert settings.seed == MINING_SEED  # derived from candidate mining seed


def test_a_g0_run_for_other_tasks_or_settings_fails_closed(tmp_path, registry, base_config,
                                                           envelope):
    runner = ScriptedRegRunner(successes=40)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope, runner)
    other_tasks = RegressionAllocation(tasks=tuple(
        TRegTask(task_id="other-" + str(i), family="heat_and_place",
                 baseline_trajectory_id="T-" + format(i, "04d") + "aaaa") for i in range(40)))
    with pytest.raises(RunnerContractError, match="other T_reg tasks"):
        evaluator.evaluate(make_treg(), make_baseline(base_config, runner, other_tasks))
    with pytest.raises(RunnerContractError, match="used settings"):
        evaluator.evaluate(make_treg(), make_baseline(base_config, runner, mining_seed=23))


def test_baseline_provenance_required():
    with pytest.raises(ValueError, match="G0 trajectory id"):
        TRegTask(task_id="treg-1", family="heat_and_place",
                  baseline_trajectory_id=None)
    with pytest.raises(ValueError, match="G0 trajectory id"):
        TRegTask(task_id="treg-1", family="heat_and_place",
                  baseline_trajectory_id="not-a-trajectory")


def test_overhead_512_passes_513_fails():
    from inheritance.regression import measure_expression_overhead
    skill = RuntimeSkill(
        gene_id=GENE_ID, name=GENE_ID,
        principle="Inspect affordances before acting.",
        when_to_apply="Heat tasks.", procedure=(), condition=None)
    at_limit = measure_expression_overhead(skill, _FixedCounter(512))
    assert at_limit.tokens == 512 and at_limit.passed is True
    over = measure_expression_overhead(skill, _FixedCounter(513))
    assert over.tokens == 513 and over.passed is False


class _SneakyCounter:
    """Probe-valid counter whose ACTUAL count for a non-empty rendered
    block is invalid (the construction-time probe only exercises "")."""

    identity = "sneaky-counter-v1"

    def __init__(self, actual):
        self._actual = actual

    def count_tokens(self, text: str):
        return 0 if text == "" else self._actual


@pytest.mark.parametrize("bad_actual", (1.5, -3, True))
def test_overhead_rejects_invalid_actual_counts(bad_actual):
    """The ACTUAL token count returned for the real rendered block is
    fail-closed validated: non-int, negative, and bool counts abort the
    measurement so no invalid evidence is ever written."""
    from inheritance.regression import measure_expression_overhead
    from inheritance.runner import RunnerContractError
    from runtime.skills import skill_block
    skill = RuntimeSkill(
        gene_id=GENE_ID, name=GENE_ID, principle="p", when_to_apply="w",
        procedure=(), condition=None)
    assert skill_block([skill])  # non-empty rendered block -> actual path
    counter = _SneakyCounter(bad_actual)
    from inheritance.regression import validate_token_counter
    validate_token_counter(counter)  # probe passes by construction
    with pytest.raises(RunnerContractError, match="invalid actual token count"):
        measure_expression_overhead(skill, counter)


def test_overhead_counter_identity_and_hash_recorded():
    from inheritance.regression import measure_expression_overhead
    from trajectory.storage.canonical import sha256_hex
    skill = RuntimeSkill(
        gene_id=GENE_ID, name=GENE_ID, principle="p", when_to_apply="w",
        procedure=(), condition=None)
    result = measure_expression_overhead(
        skill, _FixedCounter(tokens=100, identity="tiktoken-cl100k-v1"))
    assert result.counter_identity == "tiktoken-cl100k-v1"
    assert len(result.rendered_sha256) == 64


def test_overhead_no_evaluation_only_metadata_in_rendered():
    from inheritance.regression import measure_expression_overhead
    poisoned = RuntimeSkill(
        gene_id=GENE_ID, name=GENE_ID,
        principle="applicability quantum task_families leaked",
        when_to_apply="w", procedure=(), condition=None)
    with pytest.raises(Exception, match="evaluation-only"):
        measure_expression_overhead(poisoned, _FixedCounter(tokens=1))


class _FixedCounter:
    def __init__(self, tokens, identity="fixed-counter-v1"):
        self._tokens = tokens
        self._identity = identity

    @property
    def identity(self):
        return self._identity

    def count_tokens(self, text: str) -> int:
        return self._tokens
