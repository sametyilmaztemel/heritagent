"""Gate test gaps found by mutation testing in the prior independent audit
of #12 (section 0.2: M1, M4, M5, T1, T2). Each test fails against the
mutant the audit reported as surviving the whole suite."""

import copy
import dataclasses

import pytest

from inheritance.gate import EvidenceCoherenceError
from runtime.regulation import Condition, Predicate
from tests.gate.test_gate_orchestrator import build_world


def test_higher_contribution_on_the_later_rank_is_retained(tmp_path):
    """M1: the harmful-pair rule keeps the HIGHER ablation contribution even
    when it sits on the later frozen rank (rank decides only exact ties)."""
    world = build_world(tmp_path, candidates=("a", "b"), ranks={"a": 0, "b": 1},
                        ablation_deltas={"a": 0.1, "b": 0.9},
                        interaction={"both": (False, ()), "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    by_gene = {r.gene_id: r for r in outcome.results}
    assert by_gene["a"].eliminated_by == "b" and by_gene["a"].verdict == "reject"
    assert by_gene["b"].eliminated_by is None and by_gene["b"].verdict == "promote"


class _OverBudgetCounter:
    identity = "scripted-over-budget-counter-v1"

    def count_tokens(self, text: str) -> int:
        return 600


def test_over_threshold_expression_overhead_rejects(tmp_path):
    """M4: tau_k = 512 tokens is enforced: a 600-token skill block fails
    stage 4 and the candidate is rejected."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    orch._token_counter = _OverBudgetCounter()
    result = orch.run(inputs).results[0]
    overhead = result.stage4_child["measurements"]["expression_overhead"]
    assert overhead["tokens"] == 600 and overhead["overhead_pass"] is False
    assert result.stage_results["overhead"] is False
    assert result.verdict == "reject"
    assert somatic.state("a", 1) == "rejected"


def test_generalization_failure_makes_a_candidate_ineligible(tmp_path):
    """M5: a candidate that fails stage 3 (its out-of-scope check, ADR-0012
    D5) is not preliminary-eligible, so it can neither be paired nor
    eliminate an eligible candidate."""
    world = build_world(tmp_path, candidates=("a", "b"),
                        ablation_deltas={"a": 0.9, "b": 0.1},
                        interaction={"both": (False, ()), "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    for task in inputs[0].generalization_allocation.tasks:  # a loses 24 of 24 out of scope
        runner.set(task.task_id, "with_trait", False, ("a",))
    outcome = orch.run(inputs)
    by_gene = {r.gene_id: r for r in outcome.results}
    assert by_gene["a"].stage_results["generalization_out_of_scope"] is False
    assert by_gene["a"].verdict == "reject"
    # b, despite its lower contribution, is not eliminated by an ineligible a
    assert by_gene["b"].eliminated_by is None and by_gene["b"].verdict == "promote"
    interaction = by_gene["b"].stage4_child["measurements"]["interaction"]
    assert interaction["pair_count"] == 0 and interaction["vacuous"] is True
    assert by_gene["a"].stage4_child["measurements"]["interaction"]["reason"].startswith(
        "not evaluated")
    assert not any(cond in ("both", "neither") for _, cond in runner.calls)


def _ablation_with_child(ablation, child):
    return dataclasses.replace(ablation, _child_record=child)


@pytest.mark.parametrize("field,bad_value", [
    ("temperature", 0.7),
    ("seed", 11.0),
    ("stop", ["x"]),
    ("stop", []),                            # the pre-A2 empty stop (review of #62)
    ("stop", ["\nObservation:", "END"]),     # a superset of the protocol stop
    ("stop", "\nObservation:"),              # the right text, but not a list
])
def test_corrupted_ablation_child_rejected_before_any_episode(tmp_path, field, bad_value):
    """T1: the #11 preflight covers the ablation child, not only replay."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    ablation = inputs[0].ablation_evaluation
    child = copy.deepcopy(ablation.child_record)
    child["measurements"]["eval_generation_settings"][field] = bad_value
    bad = dataclasses.replace(inputs[0],
                              ablation_evaluation=_ablation_with_child(ablation, child))
    with pytest.raises(EvidenceCoherenceError, match=f"ablation.*{field}"):
        orch.run([bad])
    assert runner.calls == []


def test_target_regulation_reaches_every_with_trait_episode(tmp_path):
    """T2: a frozen express_when condition supplied for a candidate is used
    in its stage-3 and stage-4 with-trait configs (not run constitutively)."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    condition = Condition(all_of=(Predicate("consecutive_failures", ">=", 1.0),))
    orch._target_regulation_by_gene = {"a": condition}
    seen = []
    original = runner.run_episode

    def spy(task, config, budgets, cond, settings=None):
        seen.append((task.task_id, cond, config.regulation.get("a")))
        return original(task, config, budgets, cond, settings)

    runner.run_episode = spy
    orch.run(inputs)
    with_trait = [(tid, reg) for tid, cond, reg in seen if cond == "with_trait"]
    assert with_trait, "no with-trait episodes ran"
    assert all(reg == condition for _, reg in with_trait), with_trait
    assert all(reg is None for _, cond, reg in seen if cond == "without_trait")


@pytest.mark.parametrize("change,message", [
    (lambda m: m["bootstrap"].update(lower_bound=m["bootstrap"]["lower_bound"] + 0.01),
     "lower_bound .* != recomputed"),
    (lambda m: m["bootstrap"].update(upper_bound=1.0), "upper_bound .* != recomputed"),
    (lambda m: m.update(point_estimate_delta_success=0.6), "contribution"),
    (lambda m: m["paired_outcomes"][0]["with_trait"].update(success=False),
     "recorded looks|stop"),
    (lambda m: m.update(paired_outcomes=m["paired_outcomes"][:47]), "does not stop"),
    (lambda m: m["allocation"].update(task_ids=["other"] * 192), "allocation's game"),
    (lambda m: m["looks"][0].update(p={"exact": "1/2", "float": 0.5}), "recorded looks"),
    (lambda m: m["bootstrap"].pop("confidence"), "confidence None"),
    (lambda m: m["bootstrap"].update(confidence=0.9), "confidence 0.9"),
    (lambda m: m["bootstrap"].update(method="bca"), "method 'bca'"),
    (lambda m: m["bootstrap"].update(descriptive=False), "descriptive False"),
])
def test_the_gate_recomputes_the_stage_2_decision(tmp_path, change, message):
    """ADR-0012 D4 (as ADR-0009 D5 did for the bootstrap): the gate
    recomputes the sequential decision and the descriptive bootstrap from
    the child; a record that does not recompute is no evidence, and no R5
    episode runs."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    ablation = inputs[0].ablation_evaluation
    child = copy.deepcopy(ablation.child_record)
    change(child["measurements"])
    bad = dataclasses.replace(inputs[0],
                              ablation_evaluation=_ablation_with_child(ablation, child))
    with pytest.raises(EvidenceCoherenceError, match=message):
        orch.run([bad])
    assert runner.calls == []


def test_a_stage_pass_the_recomputed_decision_does_not_give_is_rejected(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    ablation = inputs[0].ablation_evaluation
    child = copy.deepcopy(ablation.child_record)
    child["measurements"]["stage_pass"] = False
    bad = dataclasses.replace(inputs[0], ablation_evaluation=dataclasses.replace(
        _ablation_with_child(ablation, child), _passed=False))
    with pytest.raises(EvidenceCoherenceError, match="stage_pass False != the rule's True"):
        orch.run([bad])
    assert runner.calls == []


def test_an_evaluation_that_disagrees_with_its_record_is_rejected(tmp_path):
    """The evaluation the gate consumes must carry its record's alpha_met,
    contribution, looks, stop and paired outcomes, not only its pass: the
    in-scope child and the sensitivity grid read them (ADR-0012 D4, D5, D7)."""
    from fractions import Fraction
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    ablation = inputs[0].ablation_evaluation
    outcomes = copy.deepcopy(ablation.paired_outcomes)
    for outcome in outcomes:
        outcome["with_trait"]["success"] = False
    looks = copy.deepcopy(ablation.looks)
    looks[0]["n_plus"] = 23
    for change in ({"_passed": False}, {"_alpha_met": False},
                   {"_contribution": Fraction(1, 3)},  # the record: 1/2
                   {"_paired_outcomes": tuple(outcomes)}, {"_looks": tuple(looks)},
                   {"_stop": {**ablation.stop, "reason": "futility"}}):
        bad = dataclasses.replace(inputs[0],
                                  ablation_evaluation=dataclasses.replace(ablation, **change))
        with pytest.raises(EvidenceCoherenceError,
                           match="pass, alpha_met, contribution, looks, stop or paired outcomes"):
            orch.run([bad])
    assert runner.calls == []


def _harmful():
    return {"both": (False, ()), "neither": (True, ())}


def _pair_calls(runner):
    return [c for c in runner.calls if c[1] in ("both", "neither")]


def test_a_candidate_failing_stage_1_is_not_eligible_and_eliminates_nobody(tmp_path):
    """Eligibility needs every stage (design stage 4): a fails stage 1 with the
    higher contribution, so it enters no pair and cannot eliminate b."""
    world = build_world(tmp_path, candidates=("a", "b"), replay_pass_for={"a": False},
                        ablation_deltas={"a": 0.9, "b": 0.1}, interaction=_harmful())
    orch, somatic, cig_store, runner, inputs, registry = world
    by_gene = {r.gene_id: r for r in orch.run(inputs).results}
    assert by_gene["a"].stage_results["replay"] is False and by_gene["a"].verdict == "reject"
    assert by_gene["b"].verdict == "promote" and by_gene["b"].eliminated_by is None
    assert _pair_calls(runner) == []


class _OverheadOfA:
    identity = "scripted-overhead-of-a-v1"

    def count_tokens(self, text: str) -> int:
        return 600 if "principle of a" in text else 10


def test_a_candidate_failing_the_overhead_is_not_eligible_and_eliminates_nobody(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"), ablation_deltas={"a": 0.9, "b": 0.1},
                        interaction=_harmful())
    orch, somatic, cig_store, runner, inputs, registry = world
    orch._token_counter = _OverheadOfA()
    by_gene = {r.gene_id: r for r in orch.run(inputs).results}
    assert by_gene["a"].stage_results["overhead"] is False and by_gene["a"].verdict == "reject"
    assert by_gene["b"].stage_results["overhead"] is True
    assert by_gene["b"].verdict == "promote" and by_gene["b"].eliminated_by is None
    assert _pair_calls(runner) == []


def test_a_candidate_failing_4a_is_not_eligible_and_eliminates_nobody(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"), ablation_deltas={"a": 0.9, "b": 0.1},
                        interaction=_harmful())
    orch, somatic, cig_store, runner, inputs, registry = world
    for i in range(10):  # a loses 10 of its 40 T_reg tasks: drop 0.25 > tau_r
        runner.set(f"treg-a-{i}", "with_trait", False, ("a",))
    by_gene = {r.gene_id: r for r in orch.run(inputs).results}
    assert by_gene["a"].stage_results["regression"] is False and by_gene["a"].verdict == "reject"
    assert by_gene["b"].verdict == "promote" and by_gene["b"].eliminated_by is None
    assert _pair_calls(runner) == []


@pytest.mark.parametrize("change,message", [
    (lambda m: (m["allocation"].update(families=["navigate"]),
                [o.update(family="navigate") for o in m["paired_outcomes"]]),
     "frozen applicability"),
    (lambda m: m["paired_outcomes"][0]["cell"].update(seed=99), "a cell of seed 99"),
])
def test_the_gate_binds_stage_2_to_the_applicability_and_the_seed(tmp_path, change, message):
    """The gate passes the candidate's frozen applicability and its mining
    seed into the stage-2 recompute (ADR-0012 D2, D4)."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    ablation = inputs[0].ablation_evaluation
    child = copy.deepcopy(ablation.child_record)
    change(child["measurements"])
    bad = dataclasses.replace(inputs[0],
                              ablation_evaluation=_ablation_with_child(ablation, child))
    with pytest.raises(EvidenceCoherenceError, match=message):
        orch.run([bad])
    assert runner.calls == []


def test_a_failed_unexpressed_source_episode_passes_stage_1(tmp_path):
    """ADR-0012 D3: a failed episode passes, nothing is compared; the gate's
    recompute of the rule accepts it and the candidate is promoted."""
    world = build_world(tmp_path, candidates=("a",), source_episode={"a": (False, False)})
    orch, somatic, cig_store, runner, inputs, registry = world
    [episode] = inputs[0].replay_evaluation.child_record["measurements"]["episodes"]
    assert (episode["success"], episode["expressed_gene_ids"]) == (False, [])
    result = orch.run(inputs).results[0]
    assert result.stage_results["replay"] is True and result.verdict == "promote"


def test_a_one_game_net_loss_in_a_pair_is_harmful(tmp_path):
    """ADR-0002 A1 (kept by ADR-0012 D6): delta_pair = -1/10 < -0.05, so the
    lower-contribution member is eliminated."""
    world = build_world(tmp_path, candidates=("a", "b"), ablation_deltas={"a": 0.9, "b": 0.1})
    orch, somatic, cig_store, runner, inputs, registry = world
    runner.set("ix-0", "both", False, ("a", "b"))
    by_gene = {r.gene_id: r for r in orch.run(inputs).results}
    assert by_gene["b"].eliminated_by == "a" and by_gene["b"].verdict == "reject"
    assert by_gene["a"].verdict == "promote"


@pytest.mark.parametrize("change", [
    lambda m: m["episodes"][0].update(round=True),
    lambda m: m["episodes"][0].update(round=1.0),
    lambda m: m["episodes"][0].update(round=2),
    lambda m: m["episodes"][0].update(condition="without_trait"),
    lambda m: m["episodes"][0].update(task_id="src-other"),
    lambda m: (m["allocation"].pop("source_task_id"), m["episodes"][0].pop("task_id")),
])
def test_a_stage_1_record_off_its_source_task_or_round_is_rejected(tmp_path, change):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    replay = inputs[0].replay_evaluation
    child = copy.deepcopy(replay.child_record)
    change(child["measurements"])
    bad = dataclasses.replace(inputs[0], replay_evaluation=dataclasses.replace(
        replay, _child_record=child))
    with pytest.raises(EvidenceCoherenceError, match="not the source task with the trait"):
        orch.run([bad])
    assert runner.calls == []


def test_the_primary_thresholds_are_adr_0012_d10s():
    """The aggregate's thresholds_used states the rule the gate applies
    (ADR-0012 D10): pinned against the ADR's literal values."""
    from inheritance.gate import (
        ABLATION_STAGE_TEXT,
        IN_SCOPE_STAGE_TEXT,
        OUT_SCOPE_STAGE_TEXT,
        PRIMARY_THRESHOLDS,
        THRESHOLD_VERSION,
    )
    assert THRESHOLD_VERSION == "EXP-0001-CIG-M0-v2"
    expected = {
        "ablation_looks": [48, 96, 144, 192],
        "ablation_alphas": ["1/2000", "1/2000", "1/1000", "3/1000"],
        "tau_c": 0.05, "ablation_futility": "net <= floor(n_k/40) at looks 1-3",
        "in_scope_min_delta": 0.0, "out_scope_games": 24, "out_scope_min_delta": -0.05,
        "tau_r": 0.03, "harmful_pair_delta": -0.05, "tau_k_tokens": 512,
        "bootstrap_confidence": 0.95, "bootstrap_resamples": 10_000, "bootstrap_seed": 0,
        "bootstrap_cluster_order": "task_id"}
    assert {k: PRIMARY_THRESHOLDS[k] for k in expected} == expected
    assert set(PRIMARY_THRESHOLDS) == set(expected) | {"replay_rule", "in_scope_rule", "rounds"}
    assert "(1/2000, 1/2000, 1/1000, 3/1000) and point >= 0.05" in ABLATION_STAGE_TEXT
    assert "floor(n/40)" in ABLATION_STAGE_TEXT
    assert IN_SCOPE_STAGE_TEXT.startswith("delta_in >= 0.0 ")
    assert OUT_SCOPE_STAGE_TEXT.startswith("delta_out >= -0.05 over 24 games, exact: net loss <= 1")
    # the prose every record carries, pinned literally
    from inheritance.ablation import ABLATION_THRESHOLD_TEXT, ALLOCATION_RULE, thresholds_record
    from inheritance.gate import STAGE4_NOT_RUN_TEXT, STAGE4_TEXT
    from inheritance.generalization import OUT_SCOPE_RULE
    assert (PRIMARY_THRESHOLDS["replay_rule"], PRIMARY_THRESHOLDS["in_scope_rule"],
            PRIMARY_THRESHOLDS["rounds"]) == (
        "source task once with the trait: gene expressed or episode failed (D3)",
        "by reference to stage 2's games (D5)",
        "R1 stage 1 and look 1; R2-R4 looks 2-4; R5 out-of-scope and 4A; R6 4C; one G0 cell "
        "per (seed, round, game), read only in its round (D2)")
    assert ABLATION_THRESHOLD_TEXT == (
        "exact one-sided paired sign test at looks 48/96/144/192: pass at look k iff p_k <= "
        "alpha_k (1/2000, 1/2000, 1/1000, 3/1000) and net/n_k >= 0.05; at looks 1-3 stop and "
        "fail iff net <= floor(n_k/40); no efficacy at look 4 fails (ADR-0012 D4)")
    assert thresholds_record()["futility"] == "net <= floor(n_k / 40) at looks 1-3"
    assert ALLOCATION_RULE == (
        "192 benefit_bank games over the frozen applicability families in name order: game j "
        "is the floor(j/m)-th game, in bank order, of the (j mod m)-th family")
    assert OUT_SCOPE_RULE == (
        "24 benefit_bank games over the omitted families in name order, round-robin: game j "
        "is the floor(j/m)-th game, in bank order, of the (j mod m)-th omitted family")
    assert STAGE4_TEXT == ("paired regression drop <= 0.03 against the R5 cells (ADR-0009 D1; "
                           "ADR-0012 D2), overhead <= 512 tokens, not eliminated")
    assert STAGE4_NOT_RUN_TEXT == ("4A and 4C not run: stage 2 never met alpha_k (ADR-0012 D7); "
                                   "overhead <= 512 tokens")


def test_a_replay_record_off_the_expression_rule_is_rejected(tmp_path):
    """ADR-0012 D3 recomputed: a successful source episode that did not
    express the gene cannot pass stage 1."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    replay = inputs[0].replay_evaluation
    child = copy.deepcopy(replay.child_record)
    child["measurements"]["episodes"][0]["expressed_gene_ids"] = []
    bad = dataclasses.replace(inputs[0], replay_evaluation=dataclasses.replace(
        replay, _child_record=child))
    with pytest.raises(EvidenceCoherenceError, match="expression rule"):
        orch.run([bad])
    assert runner.calls == []


def test_the_t_reg_cells_are_round_5s_and_must_not_express_the_target(tmp_path):
    """4A pairs T_reg with the seed's R5 cells (ADR-0012 D2, D6); a cell that
    expressed the target gene is a contaminated intervention."""
    from inheritance.runner import EvaluationIntegrityError
    orch, somatic, cig_store, runner, inputs, registry = build_world(tmp_path / "clean",
                                                                     candidates=("a",))
    result = orch.run(inputs).results[0]
    episodes = result.stage4_child["measurements"]["regression"]["paired_g0_run"]["episodes"]
    assert len(episodes) == 40 and {e["round"] for e in episodes} == {5}
    orch, somatic, cig_store, runner, inputs, registry = build_world(tmp_path / "leak",
                                                                     candidates=("a",))
    runner.set("treg-a-3", "without_trait", True, ("a",))
    with pytest.raises(EvaluationIntegrityError, match="expressed the target gene"):
        orch.run(inputs)
