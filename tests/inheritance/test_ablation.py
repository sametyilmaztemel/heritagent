"""Stage-2 evaluator tests (ADR-0012 D4): the allocation, the looks, the
exact sign-test rule and its records."""

import copy
import re
from fractions import Fraction

import pytest

from inheritance.ablation import (
    ALLOCATION_SIZE,
    ALPHAS,
    LOOKS,
    TAU_C,
    AblationAllocation,
    AblationEvaluator,
    look_decision,
    sequential_path,
    stage2_evidence_issues,
    tau_c_sensitivity,
)
from inheritance.bootstrap import sign_test_p
from inheritance.runner import EvaluationIntegrityError, EvaluationTask, RunnerContractError
from runtime.loop import Budgets
from tests.inheritance.conftest import GENE_ID, MINING_SEED, PARENT_CIG
from tests.inheritance.scripted_runner import ScriptedRunner

FAMILY = "heat_and_place"  # the test candidate's frozen applicability
BANK_SHA = "b" * 64


def make_allocation(families=(FAMILY,), count=ALLOCATION_SIZE) -> AblationAllocation:
    m = len(families)
    tasks = tuple(EvaluationTask(task_id=f"bank-{families[j % m]}-{j // m:03d}",
                                 family=families[j % m], payload={"goal": f"game {j}"},
                                 bank_id=f"benefit_bank[{families[j % m]}]")
                  for j in range(count))
    return AblationAllocation(tasks=tasks, families=tuple(families), bank_sha256=BANK_SHA)


def make_evaluator(tmp_path, registry, base_config, envelope, *, allocation=None, **kwargs):
    evaluator = AblationEvaluator(
        candidate_envelope=envelope, registry=registry, base_config=base_config,
        budgets=Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64),
        parent_cig_id=PARENT_CIG, mining_seed=MINING_SEED,
        allocation=allocation or make_allocation(), **kwargs)
    evaluator.runner = ScriptedRunner()
    return evaluator


def script(runner, allocation, pairs, *, fired=True):
    """(with-trait success, cell success) for the allocation's first games."""
    for task, (with_success, cell_success) in zip(allocation.tasks, pairs):
        runner.outcomes[(task.task_id, "with_trait")] = (
            with_success, (GENE_ID,) if fired and with_success else (), f"T-w{task.task_id[-3:]}")
        runner.outcomes[(task.task_id, "without_trait")] = (
            cell_success, (), f"T-c{task.task_id[-3:]}")


def play(evaluator):
    while not evaluator.stopped:
        evaluator.play_look(evaluator.next_look, evaluator.runner)
    return evaluator.evaluation()


# -- the rule ------------------------------------------------------------------------
def test_the_decision_table_has_the_properties_adr_0012_states():
    """35,284 (n+, n-) cells over the four looks (D4, 'Tables checked'): only
    (9, 0) at look 4 meets alpha without the point rule (D7), meeting alpha
    implies net > 0 (D5: the in-scope child adds no test), efficacy and
    futility never coincide, and the smallest net meeting alpha per look."""
    cells, alpha_without_point, smallest = 0, [], {}
    for look, n_k in enumerate(LOOKS, start=1):
        for n_plus in range(n_k + 1):
            for n_minus in range(n_k + 1 - n_plus):
                cells += 1
                d = look_decision(look, n_plus, n_minus)
                if d["alpha_met"]:
                    assert d["net"] > 0
                    smallest[look] = min(smallest.get(look, n_k), d["net"])
                    if d["point"] < TAU_C:
                        alpha_without_point.append((look, n_plus, n_minus))
                if d["decision"] == "fail" and d["reason"] == "futility":
                    assert not (d["alpha_met"] and d["point"] >= TAU_C)
    assert cells == 35_284
    assert alpha_without_point == [(4, 9, 0)]
    assert smallest == {1: 11, 2: 11, 3: 10, 4: 9}


def test_the_sign_test_is_exact():
    assert sign_test_p(0, 0) == 1
    assert sign_test_p(9, 0) == Fraction(1, 512)
    assert sign_test_p(3, 1) == Fraction(5, 16)
    assert sign_test_p(0, 5) == 1
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError):
            sign_test_p(bad, 0)


@pytest.mark.parametrize("look, n_plus, n_minus, decision, reason", [
    (1, 11, 0, "pass", "efficacy"),          # p = 2^-11 <= 1/2000 and 11/48 >= 0.05
    (1, 1, 0, "fail", "futility"),           # net 1 <= floor(48/40)
    (1, 2, 0, "continue", None),             # net 2 > 1, alpha not met
    (2, 2, 0, "fail", "futility"),           # floor(96/40) = 2
    (3, 3, 0, "fail", "futility"),           # floor(144/40) = 3
    (3, 4, 0, "continue", None),
    (4, 9, 0, "fail", "no efficacy at the last look"),   # alpha met, point 9/192 < 0.05
    (4, 10, 0, "pass", "efficacy"),
    (4, 0, 0, "fail", "no efficacy at the last look"),   # no futility rule at look 4
])
def test_look_decisions(look, n_plus, n_minus, decision, reason):
    d = look_decision(look, n_plus, n_minus)
    assert (d["decision"], d["reason"]) == (decision, reason)
    assert d["alpha"] == ALPHAS[look - 1]


def _null_pass_probability(discordance: float) -> float:
    """P(stage 2 passes) for a no-effect candidate: each game is discordant
    with probability ``discordance``, then n+ or n- with 1/2 each (exact DP
    over the counts, the stopping rule applied at each look)."""
    states = {(0, 0): 1.0}
    passed, games = 0.0, 0
    for look, n_k in enumerate(LOOKS, start=1):
        while games < n_k:
            new: dict = {}
            for (a, b), prob in states.items():
                for key, weight in (((a + 1, b), discordance / 2), ((a, b + 1), discordance / 2),
                                    ((a, b), 1 - discordance)):
                    if weight:
                        new[key] = new.get(key, 0.0) + prob * weight
            states, games = new, games + 1
        survivors = {}
        for (a, b), prob in states.items():
            d = look_decision(look, a, b)
            if d["decision"] == "pass":
                passed += prob
            elif d["decision"] == "continue":
                survivors[(a, b)] = prob
        states = survivors
    return passed


@pytest.mark.parametrize("discordance", [0.1, 1.0])
def test_a_no_effect_candidate_passes_within_the_alpha_budget(discordance):
    """D4: at most alpha_1 + ... + alpha_4 = 0.005 at any noise level (the
    ADR's exact DP puts the largest size near 0.0032, M)."""
    assert sum(ALPHAS) == Fraction(1, 200)
    assert _null_pass_probability(discordance) <= 0.005


def test_the_path_stops_where_the_rule_decides():
    pairs = [(True, False)] * 11 + [(False, False)] * 37  # (11, 0) at look 1
    looks, stop = sequential_path(pairs)
    assert [look["decision"] for look in looks] == ["pass"] and stop["games"] == 48
    looks, stop = sequential_path(pairs[:40])
    assert looks == [] and stop is None  # fewer games than the first look


@pytest.mark.parametrize("pairs, tau, identifiable, passes", [
    # efficacy at look 3 with (10, 0): tau 0.08 would need 11.52, so it would have
    # continued past look 3 (D7, not identifiable)
    ([(True, False)] * 10 + [(False, False)] * 134, Fraction(2, 25), False, None),
    ([(True, False)] * 10 + [(False, False)] * 134, Fraction(3, 100), True, True),
    # (9, 0) at look 4 fails at 0.05 and passes at 0.03 (identifiable: no later look)
    ([(True, False)] * 9 + [(False, False)] * 183, Fraction(3, 100), True, True),
    ([(True, False)] * 9 + [(False, False)] * 183, Fraction(2, 25), True, False),
])
def test_tau_c_sensitivity_on_the_recorded_path(pairs, tau, identifiable, passes):
    # the recorded path stops where tau_c = 0.05 stops
    assert sequential_path(pairs)[1]["games"] == len(pairs)
    result = tau_c_sensitivity(pairs, tau)
    assert result["identifiable"] is identifiable
    assert result["pass"] is passes


# -- the allocation ---------------------------------------------------------------------
def test_the_allocation_interleaves_the_families_in_name_order():
    allocation = make_allocation(families=("a_family", "b_family"))
    assert [t.family for t in allocation.tasks[:4]] == ["a_family", "b_family"] * 2
    assert allocation.look_tasks(2) == allocation.tasks[48:96]
    assert allocation.allocation_metadata()["family_counts"] == {"a_family": 96, "b_family": 96}
    with pytest.raises(ValueError, match="exactly 192"):
        make_allocation(count=191)
    with pytest.raises(ValueError, match="name order"):
        make_allocation(families=("b_family", "a_family"))
    tasks = list(make_allocation(families=("a_family", "b_family")).tasks)
    tasks[0], tasks[1] = tasks[1], tasks[0]
    with pytest.raises(ValueError, match="slot belongs to"):
        AblationAllocation(tasks=tuple(tasks), families=("a_family", "b_family"),
                           bank_sha256=BANK_SHA)
    bad = [EvaluationTask(task_id=t.task_id, family=t.family, payload={},
                          bank_id="generalization_bank[heat_and_place]")
           for t in make_allocation(families=("a_family", "b_family")).tasks]
    with pytest.raises(ValueError, match="bank provenance"):
        AblationAllocation(tasks=tuple(bad), families=("a_family", "b_family"),
                           bank_sha256=BANK_SHA)
    with pytest.raises(ValueError, match="distinct"):
        AblationAllocation(tasks=(make_allocation().tasks[0],) * 192, families=(FAMILY,),
                           bank_sha256=BANK_SHA)


def test_the_allocation_must_cover_the_frozen_applicability(tmp_path, registry, base_config,
                                                            envelope):
    from inheritance.overlay import EvaluationSetupError
    with pytest.raises(EvaluationSetupError, match="frozen applicability"):
        make_evaluator(tmp_path, registry, base_config, envelope,
                       allocation=make_allocation(families=("navigate",)))


@pytest.mark.parametrize("seed", [None, True, 1.0, "0", 1])
def test_invalid_bootstrap_seed_rejected_before_any_episode(tmp_path, registry, base_config,
                                                            envelope, seed):
    from inheritance.overlay import EvaluationSetupError
    with pytest.raises(EvaluationSetupError, match="bootstrap seed"):
        make_evaluator(tmp_path, registry, base_config, envelope, bootstrap_seed=seed)


# -- the looks ----------------------------------------------------------------------------
def test_efficacy_at_look_one_stops_after_48_games(tmp_path, registry, base_config, envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    allocation = make_allocation()
    script(evaluator.runner, allocation, [(True, False)] * 11 + [(False, False)] * 181)
    result = play(evaluator)
    assert result.passed is True and result.alpha_met is True
    assert result.stop == {"look": 1, "round": 1, "games": 48, "decision": "pass",
                           "reason": "efficacy"}
    assert result.contribution == Fraction(11, 48)
    # one with-trait execution and one cell read per game, never past the stop
    calls = evaluator.runner.calls
    assert len(calls) == 96 and len(set(calls)) == 96
    assert {task for task, _ in calls} == {t.task_id for t in allocation.tasks[:48]}
    with pytest.raises(RunnerContractError, match="never after its stop"):
        evaluator.play_look(2, evaluator.runner)


def test_futility_stops_and_the_looks_play_in_order(tmp_path, registry, base_config, envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    with pytest.raises(RunnerContractError, match="look 2 requested"):
        evaluator.play_look(2, evaluator.runner)
    allocation = make_allocation()
    # look 1: net 2 (continue); look 2: still net 2 <= floor(96/40) (futility)
    script(evaluator.runner, allocation, [(True, False)] * 2 + [(False, False)] * 190)
    assert evaluator.play_look(1, evaluator.runner)["decision"] == "continue"
    assert evaluator.play_look(2, evaluator.runner)["decision"] == "fail"
    result = evaluator.evaluation()
    assert (result.passed, result.alpha_met) == (False, False)
    assert result.stop["reason"] == "futility" and result.stop["games"] == 96
    with pytest.raises(RunnerContractError, match="not decided yet"):
        make_evaluator(tmp_path, registry, base_config, envelope).evaluation()


def test_nine_to_nothing_at_look_four_meets_alpha_but_fails(tmp_path, registry, base_config,
                                                            envelope):
    """(9, 0) at look 4: p = 1/512 <= 0.003, point 9/192 < 0.05: stage 2
    fails, yet alpha was met, so stage 3 and 4A must run (D7)."""
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    # net 3 at look 1 (above futility), 9 by look 4, discordance spread over the looks
    pairs = [(False, False)] * 192
    for index in (0, 1, 2, 50, 60, 100, 110, 150, 160):
        pairs[index] = (True, False)
    script(evaluator.runner, make_allocation(), pairs)
    result = play(evaluator)
    assert [look["decision"] for look in result.looks] == ["continue"] * 3 + ["fail"]
    assert (result.passed, result.alpha_met) == (False, True)
    assert result.contribution == Fraction(9, 192)


def test_games_are_never_filtered_by_expression(tmp_path, registry, base_config, envelope):
    """A regulated trait that does not fire still contributes its paired
    outcome; expression is recorded, not used."""
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    script(evaluator.runner, make_allocation(), [(True, False)] * 11 + [(False, False)] * 181,
           fired=False)
    result = play(evaluator)
    assert result.passed is True
    assert all(o["fired"] is False for o in result.paired_outcomes)


def test_a_cell_that_expresses_the_target_is_an_integrity_error(tmp_path, registry,
                                                               base_config, envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    allocation = make_allocation()
    script(evaluator.runner, allocation, [(True, False)] * 192)
    first = allocation.tasks[0].task_id
    evaluator.runner.outcomes[(first, "without_trait")] = (False, (GENE_ID,), "T-c000")
    with pytest.raises(EvaluationIntegrityError, match="contaminated"):
        evaluator.play_look(1, evaluator.runner)


def test_the_runner_must_acknowledge_the_locked_settings(tmp_path, registry, base_config,
                                                         envelope):
    from inheritance.runner import EpisodeOutcome
    from runtime.model_adapters import GenerationSettings
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    script(evaluator.runner, make_allocation(), [(True, False)] * 192)
    real_run = evaluator.runner.run_episode

    def diverging(task, config, budgets, condition, settings):
        outcome = real_run(task, config, budgets, condition, settings)
        return EpisodeOutcome(
            task_id=outcome.task_id, condition=outcome.condition, success=outcome.success,
            expressed_gene_ids=outcome.expressed_gene_ids, status=outcome.status,
            trajectory_id=outcome.trajectory_id,
            effective_settings=GenerationSettings(temperature=0.0, max_tokens=9999,
                                                  seed=12345, stop=("END",)))
    evaluator.runner.run_episode = diverging
    with pytest.raises(RunnerContractError, match="effective settings differ"):
        evaluator.play_look(1, evaluator.runner)


# -- the record ----------------------------------------------------------------------------
@pytest.fixture
def passed_record(tmp_path, registry, base_config, envelope):
    evaluator = make_evaluator(tmp_path, registry, base_config, envelope)
    pairs = [(True, False)] * 6 + [(False, True)] + [(False, False)] * 41 + \
        [(True, False)] * 10 + [(False, False)] * 134
    script(evaluator.runner, make_allocation(), pairs)
    return play(evaluator)


def test_the_child_record_carries_the_path_and_recomputes(passed_record):
    from genome.validation.loader import load_cig
    child = passed_record.child_record
    load_cig(child)
    assert child["cig_id"] == f"{PARENT_CIG}/S{MINING_SEED}-ablation"
    m = child["measurements"]
    assert [look["decision"] for look in m["looks"]] == ["continue", "pass"]
    assert m["looks"][1]["n_plus"] == 16 and m["looks"][1]["n_minus"] == 1
    assert m["looks"][1]["p"]["exact"] == str(sign_test_p(16, 1))
    assert m["stop"]["round"] == 2 and m["contribution"]["exact"] == "5/32"
    assert len(m["paired_outcomes"]) == 96
    assert m["paired_outcomes"][50]["round"] == m["paired_outcomes"][50]["cell"]["round"] == 2
    assert m["bootstrap"]["descriptive"] is True and m["bootstrap"]["seed"] == 0
    assert m["paired_delta_task_ids"] == sorted(o["task_id"] for o in m["paired_outcomes"])
    assert m["stage_pass"] is True
    assert stage2_evidence_issues(m, mining_seed=MINING_SEED) == []


@pytest.mark.parametrize("tamper, message", [
    (lambda m: m["looks"][1].__setitem__("n_plus", 17), "recorded looks"),
    (lambda m: m.__setitem__("stage_pass", False), "stage_pass"),
    (lambda m: m.__setitem__("alpha_met", False), "alpha_met"),
    (lambda m: m["paired_outcomes"][0]["with_trait"].__setitem__("success", False),
     "recorded looks|stop"),
    (lambda m: m["paired_outcomes"][50]["cell"].__setitem__("round", 1), "round"),
    (lambda m: m["paired_outcomes"][3].__setitem__("task_id", "bank-x"), "allocation's game"),
    (lambda m: m["paired_outcomes"].pop(), "does not stop"),
    (lambda m: m["bootstrap"].__setitem__("seed", 1), "bootstrap.seed"),
    (lambda m: m["bootstrap"].__setitem__("lower_bound", 0.5), "lower_bound"),
    (lambda m: m.__setitem__("contribution", {"exact": "1/2", "float": 0.5}), "contribution"),
])
def test_a_tampered_record_does_not_recompute(passed_record, tamper, message):
    m = copy.deepcopy(passed_record.child_record["measurements"])
    tamper(m)
    issues = stage2_evidence_issues(m, mining_seed=MINING_SEED)
    assert issues and any(re.search(message, i) for i in issues), issues


def test_a_cell_of_another_seed_is_flagged(passed_record):
    m = passed_record.child_record["measurements"]
    assert stage2_evidence_issues(m, mining_seed=MINING_SEED + 1)


def test_immutable_result_evidence(passed_record):
    looks = passed_record.looks
    looks[0]["decision"] = "pass"
    assert passed_record.looks[0]["decision"] == "continue"
    record = passed_record.child_record
    record["measurements"]["stage_pass"] = False
    assert passed_record.child_record["measurements"]["stage_pass"] is True
