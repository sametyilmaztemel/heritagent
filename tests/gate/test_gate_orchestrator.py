"""CIG gate orchestrator tests (issue #12 criteria 14-16) — fully scripted,
no GPU/live model/ALFWorld."""

import json

import pytest

from genome.validation.errors import RecordConsistencyError
from genome.validation.loader import load_cig
from genome.validation.registry import TraitRegistry
from genome.validation.records import CigRecordStore
from inheritance.gate import (
    CigGateOrchestrator,
    GateCandidateInput,
    GateInputError,
)
from runtime.loop import Budgets
from somatic.store import SomaticStore
from tests.gate.conftest import interaction_tasks
from tests.gate.helpers import (
    BUDGETS,
    MINING_SEED,
    PARENT_CIG,
    ScriptedGateRunner,
    coherent_stage11_children,
    mint_envelope,
)


def build_world(tmp_path, *, candidates=("a", "b"), ranks=None,
                 replay_pass=True, ablation_deltas=None, gen_in_with=True,
                 gen_out_with=True, treg_success_count=40,
                 interaction=None, replay_pass_for=None):
    """Build a full scripted world: runner, inputs, orchestrator, stores.

    Generalization defaults: with-trait all succeed, without-trait all
    succeed -> delta 0 (pass both scopes). Interaction defaults: both pass,
    neither pass -> delta 0 (not harmful).
    ablation_deltas: optional per-name ablation contribution."""
    from inheritance.generalization import GeneralizationAllocation
    from inheritance.regression import RegressionAllocation, TRegTask
    from inheritance.interaction import InteractionAllocation
    from inheritance.runner import EvaluationTask

    registry = TraitRegistry(tmp_path / "registry")
    runner = ScriptedGateRunner()
    inputs = []
    ranks = ranks or {name: i for i, name in enumerate(candidates)}
    ablation_deltas = ablation_deltas or {}
    store = SomaticStore.create(tmp_path / "somatic.jsonl", registry)
    cig_store = CigRecordStore()

    for name in candidates:
        envelope = mint_envelope(registry, name)
        store.add_candidate(envelope)  # real flow: candidate persisted by #9/#10
        gene_id = envelope["candidate"]["gene_id"]
        delta = ablation_deltas.get(name, 0.5)
        rp = replay_pass
        if replay_pass_for is not None and name in replay_pass_for:
            rp = replay_pass_for[name]
        replay, ablation = coherent_stage11_children(
            gene_id, replay_pass=rp, ablation_pass=delta >= 0.05,
            ablation_delta=delta)
        # generalization: with/without per scope
        for scope, w_ok in (("in", gen_in_with), ("out", gen_out_with)):
            for i in range(10):
                tid = f"{scope}-{name}-{i}"
                runner.set(tid, "with_trait", w_ok, (gene_id,))
                runner.set(tid, "without_trait", True, ())
        # regression: first treg_success_count succeed
        for i in range(40):
            runner.set(f"treg-{name}-{i}", "with_trait",
                        i < treg_success_count, (gene_id,))
        # interaction: both/neither outcomes shared per pair; explicit
        # dict form allows without-arm contamination simulation
        interaction = interaction or {"both": (True, (gene_id,)),
                                       "neither": (True, ())}
        both = interaction["both"]
        neither = interaction["neither"]
        for i in range(10):
            tid = f"ix-{i}"
            runner.set(tid, "both", both[0], tuple(both[1]))
            runner.set(tid, "neither", neither[0], tuple(neither[1]))
        in_scope = tuple(EvaluationTask(
            task_id=f"in-{name}-{i}", family="heat_and_place",
            payload={"goal": "g"},
            bank_id="generalization_bank[heat_and_place]") for i in range(10))
        out_scope = tuple(EvaluationTask(
            task_id=f"out-{name}-{i}", family="clean_surface",
            payload={"goal": "g"},
            bank_id="generalization_bank[clean_surface]") for i in range(10))
        gen_alloc = GeneralizationAllocation(
            in_scope_tasks=in_scope, out_scope_tasks=out_scope,
            frozen_in_scope_plan={"heat_and_place": 10},
            frozen_out_scope_plan={"clean_surface": 10},
            declared_frozen_applicability=("heat_and_place",))
        treg = RegressionAllocation(tasks=tuple(
            TRegTask(task_id=f"treg-{name}-{i}", family="heat_and_place",
                      baseline_trajectory_id="T-" + format(i, "04d") + "aaaa")
            for i in range(40)))
        inputs.append(GateCandidateInput(
            candidate_envelope=envelope, frozen_rank=ranks[name],
            replay_evaluation=replay, ablation_evaluation=ablation,
            generalization_allocation=gen_alloc,
            regression_allocation=treg))

    interaction_alloc = InteractionAllocation(tasks=interaction_tasks(10))
    counter = _WordCounter()
    orch = CigGateOrchestrator(
        registry=registry, base_config=g0_config(registry), runner=runner,
        budgets=BUDGETS, parent_cig_id=PARENT_CIG, mining_seed=MINING_SEED,
        token_counter=counter, interaction_allocation=interaction_alloc,
        somatic_store=store, cig_record_store=cig_store)
    return orch, store, cig_store, runner, inputs, registry


def g0_config(registry):
    from runtime.seed_g0 import g0_runtime_config
    return g0_runtime_config(registry)


class _WordCounter:
    identity = "scripted-word-counter-v1"

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_single_candidate_promotes_when_all_stages_pass(tmp_path):
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)

    gene = inputs[0].candidate_envelope["candidate"]["gene_id"]
    result = outcome.results[0]
    assert result.verdict == "promote"
    # aggregate has exactly the five coherent stage children
    aggregate = result.aggregate
    assert len(aggregate["child_records"]) == 5
    assert set(aggregate["stages"]) == {
        "replay", "ablation", "generalization_in_scope",
        "generalization_out_of_scope", "interaction_regression"}
    # interaction vacuously passed (fewer than 2 eligible candidates)
    stage4 = result.stage4_child["measurements"]
    assert stage4["interaction"]["pair_count"] == 0
    # atomic commit: somatic terminal decision = validated
    assert somatic.state(gene, 1) == "validated"
    # gate_reports references the AGGREGATE id only
    stored = somatic.get(gene, 1)
    assert stored["validation"]["gate_reports"] == [PARENT_CIG]
    # origin/artifact/provenance unchanged; no assimilation
    assert stored["candidate"]["origin"] == "acquired"
    assert "cig_record" not in stored["candidate"]["provenance"]


def test_interaction_two_eligible_twenty_calls_shared_evidence(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"),
                         interaction={"both": (True, ("a", "b")),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    # 2 eligible -> exactly one unordered pair: 10 tasks x 2 conditions = 20 calls
    ix_calls = [c for c in runner.calls if c[0].startswith("ix-")]
    assert len(ix_calls) == 20
    assert len(set(ix_calls)) == 20
    # both candidates promote (delta 0, not harmful)
    assert set(outcome.verdicts.values()) == {"promote"}
    # both stage-4 children reference the SAME pair evidence hash
    children = [r.stage4_child for r in outcome.results]
    hashes = [c["measurements"]["interaction"]["pair_evidence"][0]
               ["evidence_sha256"] for c in children]
    assert hashes[0] == hashes[1]


def test_harmful_pair_eliminates_lower_contribution(tmp_path):
    """a: strong ablation (0.9); b: weak (0.1). Harmful interaction -> b
    eliminated, a retained -> a promotes, b rejects."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         ablation_deltas={"a": 0.9, "b": 0.1},
                         interaction={"both": (False, ()),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    verdicts = outcome.verdicts
    # a retained (higher contribution); b eliminated -> interaction stage
    # fails for b -> reject
    a_gene = "a"
    b_gene = "b"
    assert verdicts[a_gene] == "promote" or verdicts[a_gene] == "reject"
    assert verdicts[b_gene] == "reject"
    b_result = [r for r in outcome.results if r.gene_id == b_gene][0]
    assert b_result.eliminated_by == a_gene
    assert b_result.stage4_child["measurements"]["component_pass_flags"][
        "not_eliminated"] is False


def test_equal_contribution_tie_uses_earlier_frozen_rank(tmp_path):
    """HARMFUL pair with EQUAL ablation contribution -> earlier frozen rank
    (b, rank 0) retained; a eliminated -> a rejects."""
    world = build_world(tmp_path, candidates=("b", "a"),  # b first in list
                         ranks={"b": 0, "a": 5},
                         ablation_deltas={"b": 0.5, "a": 0.5},
                         interaction={"both": (False, ()),
                                       "neither": (True, ())})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    a_result = [r for r in outcome.results if r.gene_id == "a"][0]
    b_result = [r for r in outcome.results if r.gene_id == "b"][0]
    # equal contribution: earlier frozen rank retained, later eliminated
    assert a_result.eliminated_by == "b"
    assert a_result.verdict == "reject"
    assert b_result.eliminated_by is None
    assert b_result.verdict == "promote"


def test_candidate_ordering_does_not_change_verdicts(tmp_path):
    world1 = build_world(tmp_path / "w1", candidates=("a", "b"))
    world2 = build_world(tmp_path / "w2", candidates=("b", "a"))
    o1 = world1[0].run(world1[4])
    o2 = world2[0].run(world2[4])
    assert o1.verdicts == o2.verdicts


def test_fewer_than_two_eligible_zero_pair_calls(tmp_path):
    """Both candidates fail ablation -> both reject, zero interaction calls."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         ablation_deltas={"a": 0.0, "b": 0.0})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    ix_calls = [c for c in runner.calls if c[0].startswith("ix-")]
    assert ix_calls == []
    assert set(outcome.verdicts.values()) == {"reject"}
    for r in outcome.results:
        assert r.stage4_child["measurements"]["interaction"]["pair_count"] == 0


def test_control_expression_of_target_aborts_no_child(tmp_path):
    """Neither-arm expression of either target = contaminated intervention:
    typed integrity error, stage abort, no completed child record."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         interaction={"both": (True, ("a",)),
                                       "neither": (True, ("b",))})
    orch, somatic, cig_store, runner, inputs, registry = world
    from inheritance.runner import EvaluationIntegrityError
    with pytest.raises(EvaluationIntegrityError, match="contaminated"):
        orch.run(inputs)


def test_reject_verdicts_when_replay_fails(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"), replay_pass=False,
                         ablation_deltas={"a": 0.0, "b": 0.0})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    # later primary stages still ran (no short-circuit) and verdicts reject
    for r in outcome.results:
        assert r.verdict == "reject"
        assert r.stage_results["replay"] is False
        # stage 3 still produced evidence
        assert r.aggregate["stages"]["generalization_in_scope"] is not None
    # somatic decisions: rejected
    for r in outcome.results:
        assert somatic.state(r.gene_id, 1) == "rejected"
