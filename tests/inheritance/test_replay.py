"""Replay evaluator tests (issue #11 criteria 5-6, 13)."""

import json

import pytest

from inheritance.runner import EvaluationTask, RunnerContractError
from inheritance.replay import (
    ReplayAllocation,
    ReplayEvaluator,
    REPLAY_INSTANCE_COUNT,
)
from runtime.loop import Budgets
from tests.inheritance.conftest import GENE_ID, MINING_SEED, PARENT_CIG
from tests.inheritance.scripted_runner import ScriptedRunner

SUCCESS_GENES = (GENE_ID,)  # tuple of gene-id strings
NO_GENES = ()


def make_allocation(tmp_path, registry, *, bank_id="replay_bank[heat_and_place]"):
    """Source + 4 same-family bank variants (distinct ids/bank provenance)."""
    source = EvaluationTask(task_id="src-1", family="heat_and_place",
                             payload={"goal": "heat"}, bank_id=None)
    variants = tuple(
        EvaluationTask(task_id=f"bank-{i}", family="heat_and_place",
                        payload={"goal": "heat"}, bank_id=bank_id)
        for i in range(1, 5))
    return ReplayAllocation(source_task=source, bank_variants=variants,
                             source_family="heat_and_place",
                             replay_bank_id=bank_id)


def make_evaluator(tmp_path, registry, base_config, envelope, budgets=None):
    return ReplayEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        runner=ScriptedRunner(),
        budgets=budgets or Budgets(max_steps=4, max_total_retries=10,
                                    max_tokens_per_request=64),
        parent_cig_id=PARENT_CIG, mining_seed=MINING_SEED)


def outcomes_for(per_task_with, per_task_without, gene=GENE_ID):
    """(task_id, condition) -> (success, expressed, trajectory_id) map for
    5 tasks, with deterministic fake T-... trajectory provenance."""
    outcomes = {}
    for index, (w_success, wo_success) in enumerate(zip(per_task_with,
                                                         per_task_without)):
        task_id = "src-1" if index == 0 else f"bank-{index}"
        outcomes[(task_id, "with_trait")] = (
            w_success, SUCCESS_GENES if w_success else NO_GENES,
            f"T-replay{index:02d}w")
        outcomes[(task_id, "without_trait")] = (
            wo_success, NO_GENES, f"T-replay{index:02d}o")
    return outcomes


def test_exact_five_tasks_family_and_distinctness(tmp_path, registry, base_config, envelope):
    allocation = make_allocation(tmp_path, registry)
    assert len(allocation.tasks) == REPLAY_INSTANCE_COUNT
    with pytest.raises(ValueError, match="exactly 4"):
        ReplayAllocation(source_task=allocation.source_task,
                          bank_variants=allocation.bank_variants[:2],
                          source_family="heat_and_place",
                          replay_bank_id=allocation.replay_bank_id)
    with pytest.raises(ValueError, match="distinct"):
        ReplayAllocation(source_task=allocation.source_task,
                          bank_variants=(allocation.bank_variants[0],
                                          allocation.bank_variants[0],
                                          allocation.bank_variants[1],
                                          allocation.bank_variants[2]),
                          source_family="heat_and_place",
                          replay_bank_id=allocation.replay_bank_id)
    with pytest.raises(ValueError, match="!= source family"):
        ReplayAllocation(source_task=allocation.source_task,
                          bank_variants=tuple(
                              EvaluationTask(task_id=f"x{i}", family="other",
                                              payload={}, bank_id="b")
                              for i in range(4)),
                          source_family="heat_and_place",
                          replay_bank_id=allocation.replay_bank_id)
    with pytest.raises(ValueError, match="bank_id.*does not match|does not match the declared"):
        ReplayAllocation(source_task=allocation.source_task,
                          bank_variants=tuple(
                              EvaluationTask(task_id=f"x{i}", family="heat_and_place",
                                              payload={}, bank_id="replay_bank[other]")
                              for i in range(4)),
                          source_family="heat_and_place",
                          replay_bank_id="replay_bank[heat_and_place]")


def test_boundary_three_of_five_with_plus_two_delta_passes(tmp_path, registry, base_config,
                                                             envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True, True, True, False, False],
        per_task_without=[False, False, True, False, False])
    result = evaluator.evaluate(allocation)
    assert result.successes_with == 3
    assert result.delta_count == 2
    assert result.delta_success == pytest.approx(0.4)
    assert result.passed is True  # expression passes everywhere
    assert len(result.episodes) == 10  # 5 tasks x 2 conditions


def test_two_of_five_fails(tmp_path, registry, base_config, envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True, True, False, False, False],
        per_task_without=[False] * 5)
    result = evaluator.evaluate(allocation)
    assert result.passed is False


def test_delta_plus_one_of_five_fails(tmp_path, registry, base_config, envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True, False, False, False, False],
        per_task_without=[False] * 5)
    result = evaluator.evaluate(allocation)
    assert result.delta_count == 1
    assert result.passed is False


def test_successful_with_episode_without_expression_fails(tmp_path, registry, base_config,
                                                            envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    outcomes = outcomes_for(per_task_with=[True] * 5, per_task_without=[False] * 5)
    # one successful with-trait episode never expressed the target gene
    outcomes[("src-1", "with_trait")] = (True, NO_GENES)
    evaluator._runner.outcomes = outcomes
    result = evaluator.evaluate(allocation)
    assert result.expression_ok is False
    assert result.passed is False


def test_without_trait_expression_leak_is_integrity_error(tmp_path, registry, base_config,
                                                           envelope):
    """Contaminated without-trait expression is INVALID evidence, not a
    candidate failure: explicit integrity error, stage abort, NO completed
    child record."""
    from inheritance.runner import EvaluationIntegrityError
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    outcomes = outcomes_for(per_task_with=[True] * 5, per_task_without=[False] * 5)
    outcomes[("bank-1", "without_trait")] = (False, SUCCESS_GENES)  # leak
    evaluator._runner.outcomes = outcomes
    with pytest.raises(EvaluationIntegrityError, match="contaminated"):
        evaluator.evaluate(allocation)


def test_regulated_target_may_not_fire_in_failed_episodes(tmp_path, registry, base_config,
                                                           envelope):
    """Failed with-trait episodes legitimately contribute no expression."""
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True, True, True, False, False],
        per_task_without=[False] * 5)
    # failed with-trait episodes without expression: allowed
    result = evaluator.evaluate(allocation)
    assert result.expression_ok is True
    assert result.passed is True


def test_exactly_ten_calls_one_per_task_condition(tmp_path, registry, base_config, envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True] * 5, per_task_without=[False] * 5)
    evaluator.evaluate(allocation)
    calls = evaluator._runner.calls
    assert len(calls) == 10
    assert len(set(calls)) == 10  # one execution per (task instance, condition)


def test_child_record_schema_valid_and_coherent(tmp_path, registry, base_config, envelope):
    from genome.validation.loader import load_cig
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True] * 5, per_task_without=[False] * 5)
    result = evaluator.evaluate(allocation)
    child = result.child_record
    load_cig(child)  # schema-valid
    assert child["cig_id"] == f"{PARENT_CIG}/S{MINING_SEED}-replay"
    assert child["parent_cig_id"] == PARENT_CIG
    assert child["seed"] == MINING_SEED
    assert child["stage"] == "replay"
    m = child["measurements"]
    assert m["successes_with"] == 5 and m["delta_count"] == 5
    assert m["stage_pass"] is True
    assert m["deterministic_execution_policy"]["runs_per_task_condition"] == 1
    assert m["allocation"]["source_family"] == "heat_and_place"


def test_runner_contract_violations_fail_closed(tmp_path, registry, base_config, envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True] * 5, per_task_without=[False] * 5)

    from runtime.model_adapters import GenerationSettings as GS
    from inheritance.runner import EpisodeOutcome, RunnerContractError
    real_run = evaluator._runner.run_episode

    def mismatched(task, config, budgets, condition, settings):
        outcome = real_run(task, config, budgets, condition, settings)
        return EpisodeOutcome(task_id="wrong-id", condition=condition,
                               success=outcome.success,
                               expressed_gene_ids=outcome.expressed_gene_ids,
                               status=outcome.status,
                               effective_settings=settings)
    evaluator._runner.run_episode = mismatched
    with pytest.raises(RunnerContractError, match="task_id"):
        evaluator.evaluate(allocation)


def test_no_somatic_decision_no_germline_mutation(tmp_path, registry, base_config, envelope):
    """#11 never writes a somatic terminal decision or a germline ref."""
    from somatic.store import SomaticStore
    path = tmp_path / "somatic.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    store.close()
    assert store.state("look_before_heat_v1", 1) == "candidate"  # untouched

    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True] * 5, per_task_without=[False] * 5)
    result = evaluator.evaluate(allocation)
    # even a PASSING replay stage does not decide the somatic fate
    assert result.passed is True
    assert store.state("look_before_heat_v1", 1) == "candidate"
    assert "cig_record" not in store.get("look_before_heat_v1", 1)["candidate"]["provenance"]
    store.close()


def test_preflight_invalid_parent_zero_runner_calls(tmp_path, registry, base_config,
                                                     envelope):
    """Invalid parent CIG id => typed error BEFORE any episode."""
    from trajectory.recorder.errors import RecorderError
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    with pytest.raises(RunnerContractError, match="parent CIG id"):
        evaluator.__init__(  # preflight fails before any episode
            candidate_envelope=envelope, registry=registry,
            base_config=base_config, runner=evaluator._runner,
            budgets=Budgets(max_steps=4, max_total_retries=10,
                             max_tokens_per_request=64),
            parent_cig_id="GATE-1", mining_seed=MINING_SEED)
    assert evaluator._runner.calls == []


def test_preflight_invalid_seed_zero_runner_calls(tmp_path, registry, base_config,
                                                   envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    with pytest.raises(RunnerContractError, match="non-negative integer"):
        evaluator.__init__(
            candidate_envelope=envelope, registry=registry,
            base_config=base_config, runner=evaluator._runner,
            budgets=Budgets(max_steps=4, max_total_retries=10,
                             max_tokens_per_request=64),
            parent_cig_id=PARENT_CIG, mining_seed=-1)
    assert evaluator._runner.calls == []


def test_missing_trajectory_rejected_no_child(tmp_path, registry, base_config, envelope):
    """Missing trajectory provenance fails closed: no completed child."""
    from inheritance.runner import EpisodeOutcome, RunnerContractError
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True] * 5, per_task_without=[False] * 5)
    real_run = evaluator._runner.run_episode

    def no_trajectory(task, config, budgets, condition, settings=None):
        outcome = real_run(task, config, budgets, condition, settings)
        return EpisodeOutcome(
            task_id=outcome.task_id, condition=outcome.condition,
            success=outcome.success,
            expressed_gene_ids=outcome.expressed_gene_ids,
            status=outcome.status, trajectory_id=None,  # missing
            effective_settings=outcome.effective_settings)
    evaluator._runner.run_episode = no_trajectory
    with pytest.raises(RunnerContractError, match="trajectory_id"):
        evaluator.evaluate(allocation)


def test_runner_temperature_violation_rejected_no_child(tmp_path, registry, base_config,
                                                          envelope):
    from inheritance.runner import EpisodeOutcome as _EO  # noqa: F401
    """A runner executing/reporting temperature 0.7 is rejected before any
    completed child evidence is emitted."""
    from inheritance.runner import EpisodeOutcome, RunnerContractError
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True] * 5, per_task_without=[False] * 5)
    evaluator._runner.acknowledge_settings = False
    evaluator._runner.reported_temperature = 0.7
    with pytest.raises(RunnerContractError, match="temperature"):
        evaluator.evaluate(allocation)
    # no child record was emitted
    assert not hasattr(evaluator, "_child_record")


def test_both_conditions_receive_identical_locked_settings(tmp_path, registry,
                                                             base_config, envelope):
    """Every with/without pair receives the IDENTICAL locked settings; the
    settings acknowledge temperature 0 and max_tokens == the fixed Budgets
    contract; both conditions see exactly the same object values."""
    budgets = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    runner = evaluator._runner
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True] * 5, per_task_without=[False] * 5)
    received = []  # (task_id, condition, settings)
    real_run = runner.run_episode

    def recording_run(task, config, budgets, condition, settings):
        received.append((task.task_id, condition, settings))
        return real_run(task, config, budgets, condition, settings)
    runner.run_episode = recording_run

    evaluator.evaluate(allocation)
    assert len(received) == 10
    locked = evaluator._locked_settings
    for task_id, condition, settings in received:
        assert settings == locked  # identical locked settings object values
        assert settings.temperature == 0.0
        assert settings.max_tokens == budgets.max_tokens_per_request
    # per pair: with and without received identical settings
    by_task = {}
    for task_id, condition, settings in received:
        by_task.setdefault(task_id, []).append(settings)
    assert all(len({json.dumps(s.__dict__, sort_keys=True)
                     for s in settings_list}) == 1
                for settings_list in by_task.values())


def test_runner_acks_temp0_but_changes_max_tokens_rejected(tmp_path, registry,
                                                             base_config, envelope):
    """A runner acknowledging temperature 0 while changing max_tokens/seed/
    stop is rejected: exact settings divergence, no completed child."""
    from inheritance.runner import EpisodeOutcome, RunnerContractError
    from runtime.model_adapters import GenerationSettings
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator._runner.outcomes = outcomes_for(
        per_task_with=[True] * 5, per_task_without=[False] * 5)
    real_run = evaluator._runner.run_episode

    def diverging(task, config, budgets, condition, settings):
        outcome = real_run(task, config, budgets, condition, settings)
        diverged = GenerationSettings(temperature=0.0,   # acks temp 0...
                                       max_tokens=9999,   # ...but changed budget
                                       seed=12345,        # ...changed seed
                                       stop=("END",))     # ...changed stop
        return EpisodeOutcome(
            task_id=outcome.task_id, condition=outcome.condition,
            success=outcome.success,
            expressed_gene_ids=outcome.expressed_gene_ids,
            status=outcome.status, trajectory_id=outcome.trajectory_id,
            effective_settings=diverged)
    evaluator._runner.run_episode = diverging
    with pytest.raises(RunnerContractError, match="effective settings differ"):
        evaluator.evaluate(allocation)  # no completed child is produced


def test_locked_settings_derived_from_budgets(tmp_path, registry, base_config, envelope):
    budgets = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope,
                                budgets=budgets)
    assert evaluator._locked_settings.temperature == 0.0
    assert evaluator._locked_settings.max_tokens == budgets.max_tokens_per_request
    assert evaluator._locked_settings.seed == MINING_SEED  # mining seed as fixed seed
    assert evaluator._locked_settings.stop == ()
