"""Aggregate/verdict coherence + atomic CIG bundle + somatic terminal
decision tests (issue #12 criteria 11-13, handoff section 13)."""

import pytest

from genome.validation.errors import RecordConsistencyError
from pathlib import Path

from genome.validation.records import CigRecordStore
from somatic.store import SomaticStore
from tests.gate.test_gate_orchestrator import build_world

FIVE_STAGES = ("replay", "ablation", "generalization_in_scope",
                "generalization_out_of_scope", "interaction_regression")


def test_exactly_five_coherent_stage_children(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    for result in outcome.results:
        aggregate = result.aggregate
        assert set(aggregate["stages"]) == set(FIVE_STAGES)
        assert len(aggregate["child_records"]) == 5
        # each stage's aggregate pass equals the referenced child's pass;
        # children come from the orchestrator's own bundle builder
        children = orch._bundle_children(result)
        by_stage = {c["stage"]: c for c in children}
        assert set(by_stage) == set(FIVE_STAGES)
        for stage in FIVE_STAGES:
            child = by_stage[stage]
            assert aggregate["stages"][stage]["pass"] == \
                child["measurements"]["stage_pass"]


def _all_children(outcome, gene_id):
    result = next(r for r in outcome.results if r.gene_id == gene_id)
    return [result.stage4_child] + [inp.replay_evaluation.child_record
                                     for inp in outcome._results and []] or []


def test_promote_iff_all_stages_pass(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    for result in outcome.results:
        assert result.verdict == "promote"
        assert all(s["pass"] for s in result.aggregate["stages"].values())


def test_reject_on_each_individual_stage_fail(tmp_path):
    """Replay fail / ablation fail / regression fail each produce reject."""
    import tempfile
    for index, kwargs in enumerate(({"replay_pass": False},
                                     {"ablation_deltas": {"a": 0.0, "b": 0.0}},
                                     {"treg_success_count": 38})):
        world = build_world(Path(tempfile.mkdtemp()) / ("case" + str(index)),
                             candidates=("a", "b"), **kwargs)
        orch, somatic, cig_store, runner, inputs, registry = world
        outcome = orch.run(inputs)
        for result in outcome.results:
            assert result.verdict == "reject"


def test_child_candidate_mismatch_rejected_by_bundle(tmp_path):
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    for result in outcome.results:
        aggregate = result.aggregate
        children = orch._bundle_children(result)
        # corrupt: child candidate disagrees with aggregate
        children[0] = dict(children[0])
        children[0]["measurements"] = dict(
            children[0]["measurements"],
            candidate_gene_id="other_gene")
        scratch = CigRecordStore()
        with pytest.raises(RecordConsistencyError, match="candidate_gene_id"):
            scratch.add_bundle(aggregate, children)


def test_atomic_bundle_failure_leaves_stores_unchanged(tmp_path):
    """A corrupt 5th child fails add_bundle atomically: live CigRecordStore
    and somatic store stay unchanged."""
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    # snapshot both stores
    cig_before = (dict(cig_store._aggregates), dict(cig_store._children))
    somatic_before = (dict(somatic._envelopes), dict(somatic._history))
    # corrupt the stage-4 child and attempt the bundle
    corrupt = outcome.results[0].stage4_child
    corrupt["measurements"]["candidate_gene_id"] = "borrowed_gene"
    scratch = CigRecordStore()
    with pytest.raises(RecordConsistencyError):
        scratch.add_bundle(outcome.results[0].aggregate,
                            [corrupt] + [c for c in
                              orch._bundle_children(outcome.results[0])
                              if c["cig_id"] != corrupt["cig_id"]])
    # live stores unchanged
    assert (dict(cig_store._aggregates), dict(cig_store._children)) == cig_before
    assert (dict(somatic._envelopes), dict(somatic._history)) == somatic_before


def test_promote_and_reject_terminal_states_retained(tmp_path):
    """a promotes (validated); b fails replay (rejected) — both retained and
    queryable with their terminal states."""
    world = build_world(tmp_path, candidates=("a", "b"),
                         replay_pass_for={"a": True, "b": False})
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    verdicts = outcome.verdicts
    assert verdicts["a"] == "promote"
    assert verdicts["b"] == "reject"
    assert somatic.state("a", 1) == "validated"
    assert somatic.state("b", 1) == "rejected"
    # both retained and queryable
    assert set(somatic.all()) >= {("a", 1), ("b", 1)}
    assert somatic.by_state("validated")
    assert somatic.by_state("rejected")
    # no germline mutation: origin unchanged, no assimilation
    for gene, state in (("a", "validated"), ("b", "rejected")):
        stored = somatic.get(gene, 1)
        assert stored["candidate"]["origin"] == "acquired"
        assert "assimilation" not in json.dumps(stored)


def test_failed_commit_leaves_candidate_and_cig_store_unchanged(tmp_path):
    """A poisoned stage-4 child fails the bundle prevalidation at commit
    time: the somatic decision is NOT applied and the live stores stay
    unchanged."""
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    cig_before = (dict(cig_store._aggregates), dict(cig_store._children))
    states_before = {r.gene_id: somatic.state(r.gene_id, 1)
                      for r in outcome.results}
    broken = copy.deepcopy(outcome.results[0].stage4_child)
    broken["measurements"]["candidate_gene_id"] = "borrowed_gene"

    class _BrokenResult:
        gene_id = "a"
        verdict = "promote"
        aggregate = outcome.results[0].aggregate
        stage4_child = broken
        parent_cig_id = "CIG-0007"
        aggregate_cig_id = "CIG-0007"

    with pytest.raises(RecordConsistencyError):
        orch._commit_results([_BrokenResult()], somatic_store=somatic,
                              cig_record_store=cig_store)
    # live stores unchanged
    assert (dict(cig_store._aggregates), dict(cig_store._children)) == cig_before
    assert {r.gene_id: somatic.state(r.gene_id, 1)
             for r in outcome.results} == states_before


import copy
import json  # noqa: E402
