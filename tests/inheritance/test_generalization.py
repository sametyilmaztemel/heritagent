"""Stage-3 generalization evaluator tests (issue #12 criterion 16) —
applicability from artifact, locked thresholds, leakage."""

import pytest

from genome.validation.errors import RecordConsistencyError
from inheritance.generalization import (
    GeneralizationAllocation,
    GeneralizationEvaluator,
)
from inheritance.runner import EvaluationTask
from runtime.loop import Budgets

from tests.gate.conftest import ScriptedGateRunner

GENE_ID = "look_before_heat_v1"


def make_envelope(registry, *, applicability=("heat_and_place",)):
    payload = {
        "name": GENE_ID,
        "principle": "Inspect affordances before acting.",
        "when_to_apply": "Heat tasks.",
        "applicability": {"task_families": list(applicability)},
        "procedure": [{"id": "S1", "text": "probe heater"}],
    }
    uri = registry.put("skill", GENE_ID, 1, payload)
    return {
        "candidate": {
            "gene_id": GENE_ID, "version": 1, "type": "skill",
            "origin": "acquired", "artifact": uri,
            "provenance": {"born_generation": 1,
                            "source_trajectory": "T-aaaa1111aaaa"},
        },
        "validation": {"state": "candidate"},
    }


def make_allocation(registry, envelope, *_ignored, in_families=None,
                     out_families=None, declared=None, plans=True):
    """Frozen plans derived from the observed families (so plan checks pass
    and applicability-vs-allocation checks fire). plans=False leaves the
    plans empty for the empty-plan rejection test. declared overrides the
    caller-supplied applicability label (which the evaluator ignores)."""
    in_families = in_families or ["heat_and_place"] * 10
    out_families = out_families or ["clean_surface"] * 10
    in_tasks = tuple(EvaluationTask(
        task_id=f"in-{i}", family=f, payload={"goal": "g"},
        bank_id=f"generalization_bank[{f}]") for i, f in enumerate(in_families))
    out_tasks = tuple(EvaluationTask(
        task_id=f"out-{i}", family=f, payload={"goal": "g"},
        bank_id=f"generalization_bank[{f}]") for i, f in enumerate(out_families))

    def plan_for(families):
        plan = {}
        for f in families:
            plan[f] = plan.get(f, 0) + 1
        return plan

    declared = declared if declared is not None else ["heat_and_place"]
    return GeneralizationAllocation(
        in_scope_tasks=in_tasks, out_scope_tasks=out_tasks,
        frozen_in_scope_plan=plan_for(in_families) if plans else {},
        frozen_out_scope_plan=plan_for(out_families) if plans else {},
        declared_frozen_applicability=tuple(declared))


def make_evaluator(tmp_path, registry, base_config, envelope):
    return GeneralizationEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        runner=ScriptedGateRunner(), budgets=Budgets(max_steps=4,
                                                      max_total_retries=10,
                                                      max_tokens_per_request=64),
        parent_cig_id="CIG-0007", mining_seed=11)


def test_applicability_read_from_artifact_not_caller(tmp_path, registry, base_config,
                                                      envelope):
    """The frozen applicability comes from the STORED artifact; the caller's
    declared list is ignored (the evaluator resolves families itself)."""
    from inheritance.generalization import resolve_frozen_applicability
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    alloc = make_allocation(tmp_path, registry, envelope,
                             declared=["quantum_chamber"])  # caller label
    result = evaluator.evaluate(alloc, candidate_envelope=envelope)
    # applicability used = artifact families, not the caller's declaration
    assert result.passed is True
    # metadata echoes the declared label (audit trail), but the enforced
    # families came from the artifact: in-scope tasks are heat_and_place
    assert result.allocation["frozen_applicability"] == ["quantum_chamber"]


def test_in_scope_family_outside_artifact_rejected(tmp_path, registry, base_config,
                                                    envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    alloc = make_allocation(tmp_path, registry, envelope,
                             in_families=["quantum_chamber"] * 10,
                             declared=["quantum_chamber"])
    with pytest.raises(ValueError, match="frozen applicability"):
        evaluator.evaluate(alloc, candidate_envelope=envelope)


def test_out_scope_family_inside_artifact_rejected(tmp_path, registry, base_config,
                                                    envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    alloc = make_allocation(tmp_path, registry, envelope,
                             out_families=["heat_and_place"] * 10)
    with pytest.raises(ValueError, match="frozen applicability"):
        evaluator.evaluate(alloc, candidate_envelope=envelope)


def test_frozen_family_plans_enforced(tmp_path, registry, base_config, envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    with pytest.raises(ValueError, match="non-empty"):
        alloc = make_allocation(tmp_path, registry, envelope, plans=False)
        evaluator.evaluate(alloc, candidate_envelope=envelope)


def test_exactly_40_calls_all_ids_distinct(tmp_path, registry, base_config, envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    runner = evaluator._runner
    alloc = make_allocation(tmp_path, registry, envelope)
    result = evaluator.evaluate(alloc, candidate_envelope=envelope)
    gen_calls = [c for c in runner.calls if c[0].startswith(("in-", "out-"))]
    assert len(gen_calls) == 40  # 20 tasks x 2 conditions
    assert len(set(gen_calls)) == 40
    ids = ([t.task_id for t in alloc.in_scope_tasks]
            + [t.task_id for t in alloc.out_scope_tasks])
    assert len(set(ids)) == 20


def test_in_scope_zero_delta_passes(tmp_path, registry, base_config, envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    runner = evaluator._runner
    for i in range(10):
        runner.set(f"in-{i}", "with_trait", True, ())
        runner.set(f"in-{i}", "without_trait", True, ())
    result = evaluator.evaluate(make_allocation(tmp_path, registry, envelope),
                                 candidate_envelope=envelope)
    assert result.delta_in == 0.0
    assert result.in_scope_passed is True  # delta 0 >= 0


def test_in_scope_negative_delta_fails(tmp_path, registry, base_config, envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    runner = evaluator._runner
    for i in range(10):
        runner.set(f"in-{i}", "with_trait", False, ())
        runner.set(f"in-{i}", "without_trait", True, ())
    result = evaluator.evaluate(make_allocation(tmp_path, registry, envelope),
                                 candidate_envelope=envelope)
    assert result.delta_in == -1.0
    assert result.in_scope_passed is False


def test_out_scope_below_minus_threshold_fails(tmp_path, registry, base_config,
                                                envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    runner = evaluator._runner
    for i in range(10):
        runner.set(f"out-{i}", "with_trait", i < 4, ())   # 4 with-successes
        runner.set(f"out-{i}", "without_trait", i < 9, ())  # 9 without-successes
    alloc = make_allocation(tmp_path, registry, envelope,
                             out_families=["clean_surface"] * 10)
    result = evaluator.evaluate(alloc, candidate_envelope=envelope)
    assert result.delta_out == pytest.approx(-0.5)
    assert result.out_scope_passed is False


def test_out_scope_at_boundary_passes(tmp_path, registry, base_config, envelope):
    """delta exactly 0.0 passes; the -0.05 rule is a lower BOUND, so any
    delta >= -0.05 passes (deltas on 10 binary tasks are multiples of 0.1,
    so 0.0 is the adjacent boundary below 0)."""
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    runner = evaluator._runner
    for i in range(10):
        runner.set(f"out-{i}", "with_trait", True, ())
        runner.set(f"out-{i}", "without_trait", True, ())
    result = evaluator.evaluate(make_allocation(tmp_path, registry, envelope),
                                 candidate_envelope=envelope)
    assert result.delta_out == 0.0
    assert result.out_scope_passed is True


def test_control_target_expression_contamination_fails_closed(tmp_path, registry,
                                                                base_config, envelope):
    from inheritance.runner import EvaluationIntegrityError
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    runner = evaluator._runner
    for i in range(10):
        runner.set(f"in-{i}", "with_trait", True, ())
        runner.set(f"in-{i}", "without_trait", True, (GENE_ID,))  # contamination
    with pytest.raises(EvaluationIntegrityError, match="contaminated"):
        evaluator.evaluate(make_allocation(tmp_path, registry, envelope),
                            candidate_envelope=envelope)


def test_trajectory_and_settings_mismatch_fail_closed(tmp_path, registry, base_config,
                                                        envelope):
    from inheritance.runner import EpisodeOutcome, RunnerContractError
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    runner = evaluator._runner
    for i in range(10):
        runner.set(f"in-{i}", "with_trait", True, ())
    real_run = runner.run_episode

    def no_trajectory(task, config, budgets, condition, settings=None):
        outcome = real_run(task, config, budgets, condition, settings)
        return EpisodeOutcome(
            task_id=outcome.task_id, condition=outcome.condition,
            success=outcome.success,
            expressed_gene_ids=outcome.expressed_gene_ids,
            status=outcome.status, trajectory_id=None,  # missing
            effective_settings=outcome.effective_settings)
    runner.run_episode = no_trajectory
    with pytest.raises(RunnerContractError, match="trajectory_id"):
        evaluator.evaluate(make_allocation(tmp_path, registry, envelope),
                            candidate_envelope=envelope)


def test_both_children_schema_valid(tmp_path, registry, base_config, envelope):
    from genome.validation.loader import load_cig
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    result = evaluator.evaluate(make_allocation(tmp_path, registry, envelope),
                                 candidate_envelope=envelope)
    for child in result.child_records:
        load_cig(child)
    assert result.in_scope_child["cig_id"].endswith("-generalization_in_scope")
    assert result.out_scope_child["cig_id"].endswith(
        "-generalization_out_of_scope")
