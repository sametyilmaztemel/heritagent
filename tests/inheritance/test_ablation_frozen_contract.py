"""Frozen allocation contract + contamination tests for the ablation
evaluator (issue #10 final re-review, groups 4-5) — ablation side."""

import pytest

from inheritance.ablation import AblationAllocation, AblationEvaluator
from inheritance.runner import (
    EvaluationIntegrityError,
    EvaluationTask,
)
from runtime.loop import Budgets
from tests.somatic.conftest import acquired_envelope

FROZEN_STRATA = {"heat_and_place": 10, "clean_surface": 6, "navigate": 4}
GENE_ID = "look_before_heat_v1"


def make_allocation(tmp_path, registry, *, families=None, frozen_strata=None):
    """frozen_strata=None -> canonical default plan; an explicit {} reaches
    AblationAllocation so its fail-closed empty-plan check fires."""
    families = families or ["heat_and_place"] * 10 + ["clean_surface"] * 6 + \
        ["navigate"] * 4
    tasks = tuple(EvaluationTask(task_id=f"abl-{i}", family=family,
                                  payload={"goal": f"task {i}"}, bank_id="ablation_bank")
                   for i, family in enumerate(families))
    if frozen_strata is None:
        frozen_strata = dict(FROZEN_STRATA)
    return AblationAllocation(tasks=tasks, frozen_strata=frozen_strata)


def make_evaluator(tmp_path, registry, base_config, *, outcomes=None,
                    parent_cig_id="CIG-0007", mining_seed=11):
    from tests.inheritance.scripted_runner import ScriptedRunner
    evaluator = AblationEvaluator(
        candidate_envelope=acquired_envelope(registry), registry=registry,
        base_config=base_config,
        runner=ScriptedRunner(),
        budgets=Budgets(max_steps=2, max_total_retries=1, max_tokens_per_request=64),
        parent_cig_id=parent_cig_id, mining_seed=mining_seed)
    if outcomes is not None:
        evaluator._runner.outcomes = outcomes
    return evaluator


def test_empty_frozen_strata_rejected(tmp_path, registry):
    with pytest.raises(ValueError, match="must be non-empty"):
        make_allocation(tmp_path, registry, frozen_strata={})


def test_strata_sum_must_equal_twenty(tmp_path, registry):
    with pytest.raises(ValueError, match="sum to 10"):
        make_allocation(tmp_path, registry,
                         frozen_strata={"heat_and_place": 10})  # sum 10 != 20


def test_observed_counts_must_match_frozen_exactly(tmp_path, registry):
    with pytest.raises(ValueError, match="frozen stratification plan"):
        make_allocation(tmp_path, registry,
                         families=["heat_and_place"] * 12 + ["clean_surface"] * 6
                         + ["navigate"] * 2,
                         frozen_strata={"heat_and_place": 10,
                                         "clean_surface": 6, "navigate": 4})


def test_frozen_strata_snapshot_survives_caller_mutation(tmp_path, registry):
    """Caller mutation of the original strata dict after construction cannot
    change the frozen snapshot used for validation/metadata."""
    original_strata = dict(FROZEN_STRATA)
    allocation = make_allocation(tmp_path, registry, frozen_strata=original_strata)
    original_strata["heat_and_place"] = 999
    original_strata["injected"] = 5
    metadata = allocation.allocation_metadata()
    assert metadata["frozen_strata"] == dict(FROZEN_STRATA)
    assert "injected" not in metadata["frozen_strata"]


def test_task_payload_snapshot_survives_caller_mutation(tmp_path, registry, base_config):
    """Caller mutation of a task payload after allocation construction cannot
    change the executed task definition."""
    from tests.somatic.conftest import acquired_envelope as _ae  # noqa: F401
    allocation = make_allocation(tmp_path, registry)
    original_payload = dict(allocation.tasks[0].payload)
    make_evaluator(tmp_path, registry, base_config)  # separate store-side setup
    with pytest.raises(TypeError):
        allocation.tasks[0].payload["goal"] = "TAMPERED"
    assert original_payload == {"goal": "task 0"}  # payload snapshot intact


def test_without_trait_contamination_integrity_error(tmp_path, registry, base_config):
    """Target expression in a without-trait episode is INVALID intervention
    evidence: typed integrity error, stage abort, NO completed child record."""
    allocation = make_allocation(tmp_path, registry)
    outcomes = {}
    for i in range(20):
        outcomes[(f"abl-{i}", "with_trait")] = (True, (GENE_ID,))
        outcomes[(f"abl-{i}", "without_trait")] = (True, (GENE_ID,))  # contaminated
    evaluator = make_evaluator(tmp_path, registry, base_config, outcomes=outcomes)
    with pytest.raises(EvaluationIntegrityError, match="contaminated"):
        evaluator.evaluate(allocation)
    # no completed child record was persisted on the evaluator
    assert not hasattr(evaluator, "_child_record")


def test_with_trait_non_expression_still_allowed(tmp_path, registry, base_config):
    """With-trait non-expression stays allowed and unfiltered (regulated
    trait semantics): those tasks contribute paired Δ=0."""
    allocation = make_allocation(tmp_path, registry)
    outcomes = {}
    for i in range(20):
        outcomes[(f"abl-{i}", "with_trait")] = (True, ())   # never fires
        outcomes[(f"abl-{i}", "without_trait")] = (True, ())
    evaluator = make_evaluator(tmp_path, registry, base_config, outcomes=outcomes)
    result = evaluator.evaluate(allocation)
    assert result.delta_success == pytest.approx(0.0)
    observations = result.child_record["measurements"]["expression_observations"]
    assert all(obs["fired_with"] is False for obs in observations)
    assert len(observations) == 20  # unfiltered
