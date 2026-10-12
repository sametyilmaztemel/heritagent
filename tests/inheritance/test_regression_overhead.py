"""Stage-4A regression + stage-4B overhead tests (issue #12 sections 5-6)."""

import pytest

from genome.validation.registry import TraitRegistry
from inheritance.regression import (
    RegressionAllocation,
    RegressionEvaluator,
    TRegTask,
    measure_expression_overhead,
)
from runtime.expression import RuntimeSkill
from runtime.loop import Budgets
from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope

GENE_ID = "look_before_heat_v1"
MINING_SEED = 11


class ScriptedRegRunner:
    """With-trait episode runner: the first `successes` tasks succeed."""

    def __init__(self, successes: int):
        self.successes = successes
        self.calls: list = []
        self.received_settings: list = []

    def run_episode(self, task, config, budgets, condition, settings=None):
        from inheritance.runner import EpisodeOutcome
        self.calls.append((task.task_id, condition))
        self.received_settings.append(settings)
        index = int(task.task_id.rsplit("-", 1)[-1])
        success = index < self.successes
        return EpisodeOutcome(
            task_id=task.task_id, condition=condition, success=success,
            expressed_gene_ids=(GENE_ID,), status="finished",
            trajectory_id="T-" + format(index, "04d") + "bbbb",
            effective_settings=settings)


def make_treg(count=40):
    return RegressionAllocation(tasks=tuple(
        TRegTask(task_id="treg-" + str(i), family="heat_and_place",
                  baseline_trajectory_id="T-" + format(i, "04d") + "aaaa")
        for i in range(count)))


def make_evaluator(tmp_path, registry, base_config, envelope, runner,
                    mining_seed=MINING_SEED):
    return RegressionEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        runner=runner, budgets=Budgets(max_steps=4, max_total_retries=10,
                                        max_tokens_per_request=64),
        parent_cig_id="CIG-0007", mining_seed=mining_seed)


def test_regression_39_of_40_passes(tmp_path, registry, base_config, envelope):
    runner = ScriptedRegRunner(successes=39)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope, runner)
    result = evaluator.evaluate(make_treg())
    assert result.successes_with == 39
    assert result.regression_drop == pytest.approx(0.025)
    assert result.passed is True  # 0.025 <= tau_r 0.03


def test_regression_38_of_40_fails(tmp_path, registry, base_config, envelope):
    runner = ScriptedRegRunner(successes=38)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope, runner)
    result = evaluator.evaluate(make_treg())
    assert result.regression_drop == pytest.approx(0.05)
    assert result.passed is False  # 0.05 > tau_r 0.03


def test_exactly_forty_calls_no_baseline_rerun(tmp_path, registry, base_config,
                                                envelope):
    runner = ScriptedRegRunner(successes=40)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope, runner)
    result = evaluator.evaluate(make_treg())
    with_calls = [c for c in runner.calls if c[1] == "with_trait"]
    assert len(with_calls) == 40
    assert all(c[1] == "with_trait" for c in runner.calls)  # no baseline rerun


def test_locked_settings_passed_to_runner(tmp_path, registry, base_config, envelope):
    runner = ScriptedRegRunner(successes=40)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope, runner)
    evaluator.evaluate(make_treg())
    settings = runner.received_settings[0]
    assert settings.temperature == 0.0
    assert settings.max_tokens == 64  # Budgets.max_tokens_per_request
    assert settings.seed == MINING_SEED  # derived from candidate mining seed


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
