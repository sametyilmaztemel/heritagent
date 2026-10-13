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
    """M5: a candidate that fails stage 3 is not preliminary-eligible, so it
    can neither be paired nor eliminate an eligible candidate."""
    world = build_world(tmp_path, candidates=("a", "b"),
                        ablation_deltas={"a": 0.9, "b": 0.1},
                        interaction={"both": (False, ()), "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    for i in range(10):  # a's in-scope with-trait episodes fail: delta_in < 0
        runner.set(f"in-a-{i}", "with_trait", False, ("a",))
    outcome = orch.run(inputs)
    by_gene = {r.gene_id: r for r in outcome.results}
    assert by_gene["a"].stage_results["generalization_in_scope"] is False
    assert by_gene["a"].verdict == "reject"
    # b, despite its lower contribution, is not eliminated by an ineligible a
    assert by_gene["b"].eliminated_by is None and by_gene["b"].verdict == "promote"
    interaction = by_gene["b"].stage4_child["measurements"]["interaction"]
    assert interaction["pair_count"] == 0 and interaction["vacuous"] is True
    assert not any(cond in ("both", "neither") for _, cond in runner.calls)


def _ablation_with_child(ablation, child):
    return dataclasses.replace(ablation, _child_record=child)


@pytest.mark.parametrize("field,bad_value", [
    ("temperature", 0.7),
    ("seed", 11.0),
    ("stop", ["x"]),
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
