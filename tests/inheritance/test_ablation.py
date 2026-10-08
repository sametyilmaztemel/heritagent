"""Ablation evaluator tests (issue #11 criteria 7-11)."""

import json

import pytest

from inheritance.ablation import AblationAllocation, AblationEvaluator, TAU_C
from inheritance.runner import EvaluationTask
from runtime.loop import Budgets
from tests.inheritance.conftest import GENE_ID, MINING_SEED, PARENT_CIG
from tests.inheritance.scripted_runner import ScriptedRunner

FROZEN_STRATA = {"heat_and_place": 10, "clean_surface": 6, "navigate": 4}


def make_allocation(tmp_path, registry, *, families=None):
    families = families or ["heat_and_place"] * 10 + ["clean_surface"] * 6 + \
        ["navigate"] * 4
    tasks = tuple(EvaluationTask(task_id=f"abl-{i}", family=family,
                                  payload={}, bank_id="ablation_bank")
                   for i, family in enumerate(families))
    return AblationAllocation(tasks=tasks, frozen_strata=dict(FROZEN_STRATA))


def make_evaluator(tmp_path, registry, base_config, envelope, **kwargs):
    return AblationEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        runner=ScriptedRunner(), budgets=Budgets(max_steps=4, max_total_retries=10,
                                                  max_tokens_per_request=64),
        parent_cig_id=PARENT_CIG, mining_seed=MINING_SEED, **kwargs)


def with_gene(express: bool):
    return ((GENE_ID,) if express else ())


def test_allocation_validation(tmp_path, registry):
    with pytest.raises(ValueError, match="exactly 20"):
        make_allocation(tmp_path, registry,
                         families=["heat_and_place"] * 19)
    from inheritance.runner import EvaluationTask
    with pytest.raises(ValueError, match="frozen stratification"):
        make_allocation(tmp_path, registry,
                         families=["heat_and_place"] * 20)  # unique ids, wrong strata
    # duplicate ids with matching strata -> distinct rejection
    tasks = tuple(EvaluationTask(task_id="dup", family="heat_and_place", payload={})
                   for _ in range(20))
    with pytest.raises(ValueError, match="distinct"):
        AblationAllocation(tasks=tasks, frozen_strata={"heat_and_place": 20})


def test_exactly_forty_calls_paired(tmp_path, registry, base_config, envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    outcomes = {}
    for i in range(20):
        outcomes[(f"abl-{i}", "with_trait")] = (True, with_gene(True))
        outcomes[(f"abl-{i}", "without_trait")] = (True, ())  # no contamination
    evaluator._runner.outcomes = outcomes
    result = evaluator.evaluate(allocation)
    assert len(evaluator._runner.calls) == 40  # 20 tasks x 2 conditions
    assert len(set(evaluator._runner.calls)) == 40
    assert len(result.episodes) == 40


def test_paired_delta_correct(tmp_path, registry, base_config, envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    outcomes = {}
    for i in range(20):
        task_id = f"abl-{i}"
        outcomes[(task_id, "with_trait")] = (True, with_gene(True))
        outcomes[(task_id, "without_trait")] = (i % 2 == 0, ())  # even: success without
    evaluator._runner.outcomes = outcomes
    result = evaluator.evaluate(allocation)
    # with: 20/20 success; without: 10/20 -> point estimate 0.5
    assert result.delta_success == pytest.approx(0.5)
    assert list(result.deltas) == [1.0 if i % 2 == 1 else 0.0 for i in range(20)]
    assert result.passed is True


def test_regulated_trait_not_firing_contributes_zero(tmp_path, registry, base_config,
                                                       envelope):
    """No post-filtering: tasks where the regulated trait never fired
    legitimately contribute paired Δ=0 (expression recorded as observation)."""
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    # trait never fires anywhere (regulation never triggers): all Δ=0
    evaluator._runner.outcomes = {
        (f"abl-{i}", "with_trait"): (True, ()) for i in range(20)
    } | {
        (f"abl-{i}", "without_trait"): (True, ()) for i in range(20)
    }
    result = evaluator.evaluate(allocation)
    assert result.delta_success == pytest.approx(0.0)
    assert all(obs["fired_with"] is False
                for obs in result.child_record["measurements"]["expression_observations"])
    assert len(result.deltas) == 20  # unfiltered


def test_bootstrap_deterministic_and_recorded(tmp_path, registry, base_config, envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    outcomes = {}
    for i in range(20):
        outcomes[(f"abl-{i}", "with_trait")] = (True, ())
        outcomes[(f"abl-{i}", "without_trait")] = (i < 10, ())
    evaluator._runner.outcomes = outcomes
    result = evaluator.evaluate(allocation)
    b = result.bootstrap
    assert b["n_resamples"] == 10_000
    assert b["seed"] == 0
    assert b["method"] == "percentile-paired-cluster-bootstrap"
    assert b["lower_bound"] <= b["upper_bound"]

    # identical evidence + seed -> identical bootstrap
    evaluator2 = make_evaluator(tmp_path, registry, base_config, envelope)
    evaluator2._runner.outcomes = outcomes
    result2 = evaluator2.evaluate(allocation)
    assert result2.bootstrap == result.bootstrap


def test_lower_bound_and_tau_c_boundaries(tmp_path, registry, base_config, envelope):
    """Boundary: all-20 positive pairs -> LB>0 -> pass; exactly tau_c point
    with LB==0 -> fail."""
    # strong effect: with always, without never -> every resample mean is 1.0
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    outcomes = {}
    for i in range(20):
        outcomes[(f"abl-{i}", "with_trait")] = (True, ())
        outcomes[(f"abl-{i}", "without_trait")] = (False, ())
    evaluator._runner.outcomes = outcomes
    result = evaluator.evaluate(allocation)
    assert result.passed is True
    assert result.bootstrap["lower_bound"] == pytest.approx(1.0)
    assert result.delta_success == pytest.approx(1.0)
    assert TAU_C < result.delta_success

    # real tau_c boundary at evaluator level: exactly ONE paired +1 delta
    # among 20 (point 0.05 == tau_c) and 19 paired zeros -> every bootstrap
    # resample mean is k/20; the 2.5th percentile lands on 0.0 -> LB == 0
    # -> FAIL despite the point estimate meeting tau_c
    allocation2 = make_allocation(tmp_path, registry)
    evaluator2 = make_evaluator(tmp_path, registry, base_config, envelope,
                                 bootstrap_seed=12345)
    outcomes2 = {}
    for i in range(20):
        outcomes2[(f"abl-{i}", "with_trait")] = (i == 0, with_gene(True))
        outcomes2[(f"abl-{i}", "without_trait")] = (False, ())
    evaluator2._runner.outcomes = outcomes2
    result2 = evaluator2.evaluate(allocation2)
    assert result2.delta_success == pytest.approx(0.05)
    assert result2.bootstrap["lower_bound"] == 0.0
    assert result2.passed is False  # LB must be strictly > 0


def test_child_record_schema_valid(tmp_path, registry, base_config, envelope):
    from genome.validation.loader import load_cig
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    outcomes = {}
    for i in range(20):
        outcomes[(f"abl-{i}", "with_trait")] = (True, with_gene(True))
        outcomes[(f"abl-{i}", "without_trait")] = (False, ())
    evaluator._runner.outcomes = outcomes
    result = evaluator.evaluate(allocation)
    child = result.child_record
    load_cig(child)
    assert child["cig_id"] == f"{PARENT_CIG}/S{MINING_SEED}-ablation"
    assert child["stage"] == "ablation"
    m = child["measurements"]
    assert m["allocation"]["task_count"] == 20
    assert m["tau_c"] == 0.05
    assert m["bootstrap"]["n_resamples"] == 10_000
    assert m["stage_pass"] is True
    assert "cig_record" not in json.dumps(child["measurements"])


def test_immutable_result_evidence(tmp_path, registry, base_config, envelope):
    allocation = make_allocation(tmp_path, registry)
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    outcomes = {}
    for i in range(20):
        outcomes[(f"abl-{i}", "with_trait")] = (True, with_gene(True))
        outcomes[(f"abl-{i}", "without_trait")] = (False, ())
    evaluator._runner.outcomes = outcomes
    result = evaluator.evaluate(allocation)

    result.episodes[0]["success"] = False
    deltas_copy = result.deltas
    deltas_copy[0] = -1.0
    result.child_record["measurements"]["stage_pass"] = False
    bootstrap_copy = result.bootstrap
    bootstrap_copy["lower_bound"] = -5.0

    assert all(d >= 0.0 for d in result.deltas)
    assert result.bootstrap["lower_bound"] > 0.0
    assert result.passed is True
    assert result.child_record["measurements"]["stage_pass"] is True
