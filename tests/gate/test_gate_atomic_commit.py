"""Aggregate/verdict coherence + atomic CIG bundle + somatic terminal
decision tests (issue #12 criteria 11-13, handoff section 13)."""

import copy
import json

import pytest

from genome.validation.errors import RecordConsistencyError
from pathlib import Path

from genome.validation.records import CigRecordStore
from somatic.store import SomaticStore
from tests.gate.helpers import PARENT_CIG
from tests.gate.test_gate_orchestrator import (
    assert_live_bundle_exposed,
    build_world,
)

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
        # per-candidate aggregate ids are unique and all live
        assert_live_bundle_exposed(cig_store, result)


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


def test_bundle_rejects_incoherent_verdict(tmp_path):
    """verdict = promote iff every stage pass is true: an aggregate whose
    verdict disagrees with its stage passes is rejected by add_bundle and
    flagged by verify() parity."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    result = outcome.results[0]
    aggregate = result.aggregate
    assert aggregate["verdict"] == "promote"
    broken = copy.deepcopy(aggregate)
    broken["verdict"] = "reject"  # all stage passes are True
    scratch = CigRecordStore()
    with pytest.raises(RecordConsistencyError, match="verdict"):
        scratch.add_bundle(broken, orch._bundle_children(result))
    broken2 = copy.deepcopy(aggregate)
    broken2["verdict"] = "promote"
    for stage in broken2["stages"]:
        broken2["stages"][stage]["pass"] = False
    with pytest.raises(RecordConsistencyError, match="verdict"):
        scratch.add_bundle(broken2, orch._bundle_children(result))


def test_bundle_rejects_inexact_child_ids_and_incoherent_seed(tmp_path):
    """M0 bundle invariants: exact <parent>/S<seed>-<stage> child ids,
    exactly one child per stage, one coherent bundle seed."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    result = outcome.results[0]
    scratch = CigRecordStore()

    # wrong exact id: relabel the replay child with a run-label suffix
    children = orch._bundle_children(result)
    relabeled = copy.deepcopy(children)
    for child in relabeled:
        if child["stage"] == "replay":
            child["cig_id"] = child["cig_id"] + "-run2"
    aggregate = copy.deepcopy(result.aggregate)
    aggregate["child_records"] = [c["cig_id"] for c in relabeled]
    with pytest.raises(RecordConsistencyError, match="exact M0 form"):
        scratch.add_bundle(aggregate, relabeled)

    # incoherent bundle seed: one child claims a different seed
    children = orch._bundle_children(result)
    reseeded = copy.deepcopy(children)
    for child in reseeded:
        if child["stage"] == "ablation":
            child["seed"] = 23
            child["cig_id"] = (child["parent_cig_id"] + "/S23-"
                                + child["stage"])
    aggregate = copy.deepcopy(result.aggregate)
    aggregate["child_records"] = [c["cig_id"] for c in reseeded]
    with pytest.raises(RecordConsistencyError,
                        match="incoherent seeds"):
        scratch.add_bundle(aggregate, reseeded)


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


def test_successful_commit_exposes_live_bundle_and_consistency(tmp_path):
    """The atomic commit exposes each aggregate + its five children on the
    LIVE CigRecordStore, somatic gate_reports reference the aggregate, the
    version comes from the evaluated envelope, and require_consistent()
    passes over the whole store."""
    world = build_world(tmp_path, candidates=("a", "b"))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    assert outcome.committed is True
    aggregate_ids = set()
    for result in outcome.results:
        assert_live_bundle_exposed(cig_store, result)
        aggregate_ids.add(result.aggregate_cig_id)
        stored = somatic.get(result.gene_id, result.candidate_version)
        assert stored["validation"]["gate_reports"] == \
            [result.aggregate_cig_id]
    # per-candidate aggregates: distinct ids, both live
    assert len(aggregate_ids) == 2
    cig_store.require_consistent()


def test_commit_prevalidates_against_current_live_store(tmp_path):
    """The staged prevalidation runs against a snapshot of the CURRENT live
    store: committing the same run twice fails (aggregate already exists)
    and leaves the live store unchanged."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    result = outcome.results[0]
    cig_before = (dict(cig_store._aggregates), dict(cig_store._children))
    somatic_before = (dict(somatic._envelopes), dict(somatic._history))
    with pytest.raises(RecordConsistencyError, match="already exists"):
        orch._commit_results([result], somatic_store=somatic,
                              cig_record_store=cig_store)
    assert (dict(cig_store._aggregates), dict(cig_store._children)) == cig_before
    assert (dict(somatic._envelopes), dict(somatic._history)) == somatic_before


def test_commit_rejects_terminal_somatic_candidate(tmp_path):
    """The somatic prevalidation runs BEFORE any live mutation: a bundle
    whose candidate is already terminal is refused ('no longer a somatic
    candidate') and neither store changes."""
    world = build_world(tmp_path, candidates=("a",), commit_stores=False)
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    result = outcome.results[0]
    orch._commit_results([result], somatic_store=somatic,
                          cig_record_store=cig_store)
    # retag the SAME evidence under a fresh aggregate id so the bundle
    # prevalidation passes and the somatic preflight is what fires
    new_parent = "CIG-9999"
    retagged_children = []
    for child in orch._bundle_children(result):
        child = copy.deepcopy(child)
        child["parent_cig_id"] = new_parent
        child["cig_id"] = new_parent + child["cig_id"][len(PARENT_CIG):]
        retagged_children.append(child)
    retagged_agg = copy.deepcopy(result.aggregate)
    retagged_agg["cig_id"] = new_parent
    retagged_agg["child_records"] = [c["cig_id"] for c in retagged_children]

    class _RetaggedResult:
        gene_id = result.gene_id
        candidate_version = result.candidate_version
        candidate = result.candidate
        verdict = result.verdict
        aggregate = retagged_agg
        stage4_child = next(c for c in retagged_children
                             if c["stage"] == "interaction_regression")
        parent_cig_id = new_parent
        aggregate_cig_id = new_parent

    # sanity: the retagged bundle itself is coherent
    scratch = CigRecordStore()
    scratch.add_bundle(retagged_agg, retagged_children)
    cig_before = (dict(cig_store._aggregates), dict(cig_store._children))
    somatic_before = (dict(somatic._envelopes), dict(somatic._history))
    real_builder = orch._bundle_children
    orch._bundle_children = lambda res: retagged_children
    try:
        with pytest.raises(RecordConsistencyError,
                            match="no longer a somatic"):
            orch._commit_results([_RetaggedResult()], somatic_store=somatic,
                                  cig_record_store=cig_store)
    finally:
        orch._bundle_children = real_builder
    assert (dict(cig_store._aggregates), dict(cig_store._children)) == cig_before
    assert (dict(somatic._envelopes), dict(somatic._history)) == somatic_before


def test_batch_commit_is_all_or_nothing_across_candidates(tmp_path):
    """Batch atomicity: when the SECOND candidate's bundle is poisoned, the
    staged prevalidation fails before ANY mutation — the first candidate's
    somatic decision is NOT applied."""
    world = build_world(tmp_path, candidates=("a", "b"), commit_stores=False)
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    a_result, b_result = outcome.results
    broken_b = copy.deepcopy(b_result.stage4_child)
    broken_b["measurements"]["candidate_gene_id"] = "borrowed_gene"

    class _BrokenB:
        gene_id = b_result.gene_id
        candidate_version = b_result.candidate_version
        verdict = b_result.verdict
        aggregate = b_result.aggregate
        stage4_child = broken_b
        parent_cig_id = b_result.parent_cig_id
        aggregate_cig_id = b_result.aggregate_cig_id

    # orch._bundle_children reads the run state; patch just the poisoned
    # child by committing through a shim that swaps the builder output
    real_builder = orch._bundle_children

    def poisoned_builder(res):
        children = real_builder(res)
        if res.gene_id == b_result.gene_id:
            children = [broken_b if c["stage"] == "interaction_regression"
                         else c for c in children]
        return children

    cig_before = (dict(cig_store._aggregates), dict(cig_store._children))
    somatic_before = (dict(somatic._envelopes), dict(somatic._history))
    orch._bundle_children = poisoned_builder
    try:
        with pytest.raises(RecordConsistencyError):
            orch._commit_results([a_result, _BrokenB()],
                                  somatic_store=somatic,
                                  cig_record_store=cig_store)
    finally:
        orch._bundle_children = real_builder
    # NEITHER store was mutated: no partial commit
    assert cig_store._aggregates == {} and cig_store._children == {}
    assert (dict(cig_store._aggregates), dict(cig_store._children)) == cig_before
    assert somatic.state(a_result.gene_id, 1) == "candidate"
    assert somatic.state(b_result.gene_id, 1) == "candidate"
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
    # both live CIG bundles exposed
    for result in outcome.results:
        assert_live_bundle_exposed(cig_store, result)


def test_failed_commit_leaves_candidate_and_cig_store_unchanged(tmp_path):
    """A poisoned stage-4 child fails the bundle prevalidation at commit
    time (staged against the CURRENT live store): the somatic decision is
    NOT applied and the live stores stay unchanged."""
    world = build_world(tmp_path, candidates=("a",), commit_stores=False)
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    result = outcome.results[0]
    cig_before = (dict(cig_store._aggregates), dict(cig_store._children))
    somatic_before = (dict(somatic._envelopes), dict(somatic._history))
    broken = copy.deepcopy(result.stage4_child)
    broken["measurements"]["candidate_gene_id"] = "borrowed_gene"

    class _BrokenResult:
        gene_id = result.gene_id
        candidate_version = result.candidate_version
        verdict = "promote"
        aggregate = result.aggregate
        stage4_child = broken
        parent_cig_id = result.parent_cig_id
        aggregate_cig_id = result.aggregate_cig_id

    real_builder = orch._bundle_children

    def poisoned_builder(res):
        children = real_builder(res)
        return [broken if c["stage"] == "interaction_regression"
                 else c for c in children]

    orch._bundle_children = poisoned_builder
    try:
        with pytest.raises(RecordConsistencyError,
                            match="borrowed|candidate_gene_id"):
            orch._commit_results([_BrokenResult()], somatic_store=somatic,
                                  cig_record_store=cig_store)
    finally:
        orch._bundle_children = real_builder
    # live stores unchanged
    assert cig_store._aggregates == {} and cig_store._children == {}
    assert (dict(cig_store._aggregates), dict(cig_store._children)) == cig_before
    assert somatic.state(result.gene_id, result.candidate_version) == \
        "candidate"
    assert (dict(somatic._envelopes), dict(somatic._history)) == somatic_before


def test_aggregate_records_the_evaluated_artifact(tmp_path):
    """The aggregate names the exact candidate reference its verdict is
    about (gene_id, version, content-addressed artifact URI)."""
    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    outcome = orch.run(inputs)
    result = outcome.results[0]
    evaluated = inputs[0].candidate_envelope["candidate"]
    expected = {"gene_id": evaluated["gene_id"], "version": evaluated["version"],
                "artifact": evaluated["artifact"]}
    assert result.aggregate["artifacts"] == {"candidate": expected}
    assert cig_store._aggregates[result.aggregate_cig_id]["artifacts"] == {
        "candidate": expected}
    assert result.candidate == evaluated


def test_commit_refuses_decision_for_a_different_stored_candidate(tmp_path):
    """The gate evaluated an envelope whose reference differs from the
    somatic candidate stored under the same (gene_id, version): the verdict
    must not be bound to the stored candidate, and neither store changes."""
    import dataclasses

    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    forged = copy.deepcopy(inputs[0].candidate_envelope)
    forged["candidate"]["provenance"]["source_trajectory"] = "T-mine000bbbb"
    inputs = [dataclasses.replace(inputs[0], candidate_envelope=forged)]
    cig_before = (dict(cig_store._aggregates), dict(cig_store._children))
    somatic_before = (dict(somatic._envelopes), dict(somatic._history))
    with pytest.raises(RecordConsistencyError, match="not the evaluated one"):
        orch.run(inputs)
    assert (dict(cig_store._aggregates), dict(cig_store._children)) == cig_before
    assert (dict(somatic._envelopes), dict(somatic._history)) == somatic_before
    assert somatic.state("a", 1) == "candidate"


def test_commit_refuses_a_stored_candidate_with_a_different_artifact(tmp_path):
    """Same gene and version, different content-addressed artifact (a store
    bound to another registry): the verdict must not be bound to it."""
    from genome.validation.registry import TraitRegistry
    from tests.gate.helpers import proposal

    world = build_world(tmp_path, candidates=("a",), commit_stores=False)
    orch, somatic, cig_store, runner, inputs, registry = world
    result = orch.run(inputs).results[0]
    other_registry = TraitRegistry(tmp_path / "other-registry")
    other_payload = dict(proposal("a"), principle="a different principle")
    other_uri = other_registry.put("skill", "a", 1, other_payload)
    assert other_uri != result.candidate["artifact"]
    other_store = SomaticStore.create(tmp_path / "other.jsonl", other_registry)
    envelope = copy.deepcopy(inputs[0].candidate_envelope)
    envelope["candidate"]["artifact"] = other_uri
    other_store.add_candidate(envelope)
    live = CigRecordStore()
    with pytest.raises(RecordConsistencyError, match="differs in artifact"):
        orch._commit_results([result], somatic_store=other_store, cig_record_store=live)
    assert other_store.state("a", 1) == "candidate"
    assert live._aggregates == {} and live._children == {}


def test_aggregate_artifacts_carry_the_evaluated_version(tmp_path):
    """A version-2 candidate: the aggregate names version 2, and only v2 is
    decided in the somatic store."""
    import dataclasses
    from tests.gate.helpers import proposal

    world = build_world(tmp_path, candidates=("a",))
    orch, somatic, cig_store, runner, inputs, registry = world
    v2 = copy.deepcopy(inputs[0].candidate_envelope)
    v2["candidate"]["version"] = 2
    v2["candidate"]["artifact"] = registry.put("skill", "a", 2, proposal("a"))
    somatic.add_candidate(v2)
    outcome = orch.run([dataclasses.replace(inputs[0], candidate_envelope=v2)])
    result = outcome.results[0]
    assert result.aggregate["artifacts"]["candidate"]["version"] == 2
    assert result.aggregate["artifacts"]["candidate"]["artifact"] == v2["candidate"]["artifact"]
    assert somatic.state("a", 2) in ("validated", "rejected")
    assert somatic.state("a", 1) == "candidate"
