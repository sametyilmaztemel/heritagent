"""Stage-3 generalization evaluator tests (issue #12 criterion 16;
recalibrated by ADR-0012 D5 and D7): the in-scope child by reference to
stage 2, the 24-game out-of-scope check in R5 against R5's cells, the
vacuous all-family pass (ADR-0007 E6) and the not-run child."""

from fractions import Fraction

import pytest

from genome.validation.loader import load_cig
from inheritance.generalization import (
    NOT_RUN_THRESHOLD_TEXT,
    OUT_SCOPE_COUNT,
    OUT_SCOPE_ROUND,
    VACUOUS_OUT_OF_SCOPE_THRESHOLD_TEXT,
    GeneralizationEvaluator,
    OutOfScopeAllocation,
    paired_outcomes_sha256,
)
from inheritance.runner import EvaluationIntegrityError, EvaluationTask, RunnerContractError
from tests.gate.helpers import (
    BANK_SHA,
    BUDGETS,
    FAMILY,
    MINING_SEED,
    PARENT_CIG,
    UNIVERSE,
    ScriptedGateRunner,
    mint_envelope,
    out_of_scope_allocation,
    stage12_evidence,
)

GENE_ID = "look_before_heat_v1"


def make_evaluator(registry, base_config, envelope):
    return GeneralizationEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        budgets=BUDGETS, parent_cig_id=PARENT_CIG, mining_seed=MINING_SEED)


def stage2(registry, base_config, envelope, shape="pass", families=(FAMILY,)):
    _, ablation = stage12_evidence(registry, base_config, envelope, stage2=shape,
                                   families=families)
    return ablation


def script_out_of_scope(runner, allocation, *, losses=0, rescues=0, cell_expresses=False):
    """The first ``losses`` games: the cell solves, the with-trait execution
    fails; the next ``rescues`` games the reverse; the rest both fail."""
    for index, task in enumerate(allocation.tasks):
        with_success = losses <= index < losses + rescues
        cell_success = index < losses
        runner.set(task.task_id, "with_trait", with_success, (GENE_ID,) if with_success else ())
        runner.set(task.task_id, "without_trait", cell_success,
                   (GENE_ID,) if cell_expresses else ())


@pytest.fixture
def envelope(registry):
    return mint_envelope(registry, GENE_ID)


# -- in scope (by reference) ------------------------------------------------------------
def test_the_in_scope_child_records_stage_2_by_reference(registry, base_config, envelope):
    ablation = stage2(registry, base_config, envelope)
    runner = ScriptedGateRunner()
    allocation = out_of_scope_allocation(GENE_ID)
    script_out_of_scope(runner, allocation)
    result = make_evaluator(registry, base_config, envelope).evaluate(ablation, allocation, runner)
    child = result.in_scope_child
    load_cig(child)
    assert child["cig_id"] == f"{PARENT_CIG}/S{MINING_SEED}-generalization_in_scope"
    m = child["measurements"]
    outcomes = ablation.paired_outcomes
    assert m["by_reference"] == {
        "stage": "ablation", "child_id": ablation.child_record["cig_id"],
        "paired_outcomes_sha256": paired_outcomes_sha256(outcomes),
        "games": 48, "n_plus": 24, "n_minus": 0, "net": 24}
    assert m["episodes_run"] == 0 and m["delta"]["exact"] == "1/2"
    assert result.in_scope_passed is True and result.delta_in == Fraction(1, 2)
    # the in-scope side plays nothing: every call is an out-of-scope game
    assert {task_id for task_id, _ in runner.calls} == {t.task_id for t in allocation.tasks}


def test_the_in_scope_child_must_be_the_candidates_own_stage_2(registry, base_config, envelope):
    other = mint_envelope(registry, "someone_else_v1")
    ablation = stage2(registry, base_config, other)
    with pytest.raises(RunnerContractError, match="stage-2 record"):
        make_evaluator(registry, base_config, envelope).evaluate(
            ablation, out_of_scope_allocation(GENE_ID), ScriptedGateRunner())


# -- out of scope --------------------------------------------------------------------------
@pytest.mark.parametrize("losses,rescues,passed,delta", [
    (0, 0, True, "0/1"),       # the record always writes numerator/denominator
    (1, 0, True, "-1/24"),     # a net loss of 1 of 24 is within -0.05
    (2, 0, False, "-1/12"),    # 2 of 24 is not
    (2, 1, True, "-1/24"),     # the net decides, not the losses
    (0, 3, True, "1/8"),
])
def test_the_out_of_scope_rule_in_exact_arithmetic(registry, base_config, envelope, losses,
                                                   rescues, passed, delta):
    ablation = stage2(registry, base_config, envelope)
    runner = ScriptedGateRunner()
    allocation = out_of_scope_allocation(GENE_ID)
    script_out_of_scope(runner, allocation, losses=losses, rescues=rescues)
    result = make_evaluator(registry, base_config, envelope).evaluate(ablation, allocation, runner)
    m = result.out_scope_child["measurements"]
    assert result.out_scope_passed is passed and m["stage_pass"] is passed
    assert m["delta"]["exact"] == delta and result.delta_out == Fraction(delta)
    assert m["summary"]["net_loss"] == losses - rescues
    assert result.passed is passed  # the in-scope side passes


def test_the_out_of_scope_check_plays_24_games_in_r5(registry, base_config, envelope):
    ablation = stage2(registry, base_config, envelope)
    runner = ScriptedGateRunner()
    allocation = out_of_scope_allocation(GENE_ID)
    script_out_of_scope(runner, allocation)
    result = make_evaluator(registry, base_config, envelope).evaluate(ablation, allocation, runner)
    assert runner.calls == [(t.task_id, condition) for t in allocation.tasks
                            for condition in ("with_trait", "without_trait")]
    child = result.out_scope_child
    load_cig(child)
    m = child["measurements"]
    assert m["round"] == OUT_SCOPE_ROUND == 5 and m["episodes_run"] == OUT_SCOPE_COUNT == 24
    assert all(o["round"] == o["cell"]["round"] == 5 and o["cell"]["seed"] == MINING_SEED
               for o in m["paired_outcomes"])
    assert m["allocation"]["omitted_families"] == ["clean_surface", "navigate"]
    assert m["allocation"]["family_counts"] == {"clean_surface": 12, "navigate": 12}
    assert m["allocation"]["bank_sha256"] == BANK_SHA
    assert "vacuous" not in m and "not_run" not in m


def test_a_cell_that_expresses_the_target_is_an_integrity_error(registry, base_config,
                                                                envelope):
    ablation = stage2(registry, base_config, envelope)
    runner = ScriptedGateRunner()
    allocation = out_of_scope_allocation(GENE_ID)
    script_out_of_scope(runner, allocation, cell_expresses=True)
    with pytest.raises(EvaluationIntegrityError, match="contaminated"):
        make_evaluator(registry, base_config, envelope).evaluate(ablation, allocation, runner)


def test_a_runner_reporting_another_task_is_rejected(registry, base_config, envelope):
    from inheritance.runner import EpisodeOutcome
    ablation = stage2(registry, base_config, envelope)
    runner = ScriptedGateRunner()
    real = runner.run_episode

    def mismatched(task, config, budgets, condition, settings=None):
        outcome = real(task, config, budgets, condition, settings)
        return EpisodeOutcome(task_id="wrong", condition=condition, success=outcome.success,
                              expressed_gene_ids=outcome.expressed_gene_ids,
                              status=outcome.status, trajectory_id=outcome.trajectory_id,
                              effective_settings=outcome.effective_settings)
    runner.run_episode = mismatched
    with pytest.raises(RunnerContractError, match="task_id"):
        make_evaluator(registry, base_config, envelope).evaluate(
            ablation, out_of_scope_allocation(GENE_ID), runner)


def test_the_out_of_scope_check_needs_the_r5_runner(registry, base_config, envelope):
    ablation = stage2(registry, base_config, envelope)
    with pytest.raises(RunnerContractError, match="R5"):
        make_evaluator(registry, base_config, envelope).evaluate(
            ablation, out_of_scope_allocation(GENE_ID), None)


# -- not run (ADR-0012 D7) ------------------------------------------------------------------
@pytest.mark.parametrize("families", [(FAMILY,), UNIVERSE])
def test_a_candidate_that_never_met_alpha_gets_a_not_run_child(registry, base_config,
                                                              families):
    envelope = mint_envelope(registry, GENE_ID, families=families)
    ablation = stage2(registry, base_config, envelope, shape="futility", families=families)
    assert ablation.alpha_met is False
    runner = ScriptedGateRunner()
    allocation = out_of_scope_allocation(GENE_ID, applicability=families)
    result = make_evaluator(registry, base_config, envelope).evaluate(ablation, allocation, runner)
    assert runner.calls == []
    child = result.out_scope_child
    load_cig(child)
    m = child["measurements"]
    assert m["not_run"] is True and m["stage_pass"] is False and m["episodes_run"] == 0
    assert m["all_family_applicability"] is (families == UNIVERSE)
    assert m["locked_threshold"] == {"min_delta": None, "rule": NOT_RUN_THRESHOLD_TEXT}
    assert "ADR-0012 D7" in m["not_run_reason"]
    assert result.out_scope_state == "not_run" and result.delta_out is None
    assert result.passed is False
    # the in-scope child is still written, by reference; delta 0 passes the locked
    # in-scope rule (delta >= 0)
    in_scope = result.in_scope_child["measurements"]
    assert in_scope["by_reference"]["games"] == 48 and in_scope["delta"]["exact"] == "0/1"
    assert in_scope["stage_pass"] is True and result.in_scope_passed is True


# -- vacuous (ADR-0007 E6) ----------------------------------------------------------------
def test_an_all_family_applicability_passes_vacuously(registry, base_config):
    envelope = mint_envelope(registry, GENE_ID, families=UNIVERSE)
    ablation = stage2(registry, base_config, envelope, families=UNIVERSE)
    runner = ScriptedGateRunner()
    allocation = out_of_scope_allocation(GENE_ID, applicability=UNIVERSE)
    assert allocation.vacuous and allocation.tasks == ()
    result = make_evaluator(registry, base_config, envelope).evaluate(ablation, allocation, runner)
    assert runner.calls == []
    m = result.out_scope_child["measurements"]
    load_cig(result.out_scope_child)
    assert m["vacuous"] is True and m["stage_pass"] is True and m["episodes_run"] == 0
    assert m["locked_threshold"] == {"min_delta": None,
                                     "rule": VACUOUS_OUT_OF_SCOPE_THRESHOLD_TEXT}
    assert m["allocation"]["out_scope_vacuous"] is True
    assert m["allocation"]["family_universe"] == list(UNIVERSE)
    assert result.out_scope_state == "vacuous" and result.passed is True


def test_the_allocation_must_be_for_the_stored_applicability(registry, base_config):
    """The caller's allocation says every family; the stored artifact names
    one: the stage is not vacuous, and nothing runs."""
    envelope = mint_envelope(registry, GENE_ID)
    ablation = stage2(registry, base_config, envelope)
    runner = ScriptedGateRunner()
    with pytest.raises(ValueError, match="frozen applicability"):
        make_evaluator(registry, base_config, envelope).evaluate(
            ablation, out_of_scope_allocation(GENE_ID, applicability=UNIVERSE), runner)
    assert runner.calls == []


# -- the allocation ------------------------------------------------------------------------
def _tasks(families, count=OUT_SCOPE_COUNT, bank=None):
    return tuple(EvaluationTask(task_id=f"out-{j:02d}", family=families[j % len(families)],
                                payload={"goal": "g"},
                                bank_id=bank or f"benefit_bank[{families[j % len(families)]}]")
                 for j in range(count))


@pytest.mark.parametrize("kwargs,message", [
    ({"tasks": _tasks(("clean_surface", "navigate"), 23)}, "exactly 24"),
    ({"tasks": _tasks(("navigate", "clean_surface"))}, "slot belongs to"),
    ({"tasks": _tasks(("clean_surface", "navigate"), bank="t_mine")}, "bank provenance"),
    ({"tasks": _tasks(("clean_surface", "navigate"), 23)
      + _tasks(("clean_surface",), 1)}, "distinct"),
    ({"applicability": ("cool_and_place",)}, "non-empty subset"),
    ({"applicability": ()}, "non-empty subset"),
    ({"family_universe": ()}, "family universe"),
    ({"bank_sha256": "b" * 63}, "sha256"),
    ({"applicability": UNIVERSE}, "vacuous out-of-scope stage has no games"),
])
def test_the_allocation_is_checked_at_construction(kwargs, message):
    args = {"tasks": _tasks(("clean_surface", "navigate")), "applicability": (FAMILY,),
            "family_universe": UNIVERSE, "bank_sha256": BANK_SHA}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        OutOfScopeAllocation(**args)


@pytest.mark.parametrize("omitted,counts", [
    (1, [24]), (2, [12, 12]), (3, [8, 8, 8]), (4, [6, 6, 6, 6]), (5, [5, 5, 5, 5, 4])])
def test_the_round_robin_family_counts(omitted, counts):
    """ADR-0012 D5: 24, 12, 8, 6, or 5/5/5/5/4 games per omitted family."""
    universe = tuple(f"fam{i}" for i in range(omitted + 1))
    applicability = (universe[-1],)
    allocation = out_of_scope_allocation("g", applicability=applicability, universe=universe)
    metadata = allocation.allocation_metadata()
    assert [metadata["family_counts"][f] for f in universe[:-1]] == counts
