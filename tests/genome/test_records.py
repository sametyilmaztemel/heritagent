"""Somatic lifecycle + two-level CIG record stores (criterion 4)."""

import pytest

from genome.validation.errors import GenomeValidationError, RecordConsistencyError
from genome.validation.records import CigRecordStore, SomaticStore
from tests.conftest import fake_uri, minimal_genome

STAGES = ("replay", "ablation", "generalization_in_scope",
          "generalization_out_of_scope", "interaction_regression")
SEED = 11


def child_ids(cig_id="CIG-0042", seed=SEED):
    """The five M0 children of one CIG run: exact <parent>/S<seed>-<stage>
    ids, one per stage."""
    return [f"{cig_id}/S{seed}-{stage}" for stage in STAGES]


def aggregate_record(cig_id="CIG-0042", children=None):
    return {
        "cig_id": cig_id,
        "candidate_gene_id": "look_before_heat_v1",
        "genome_id": "G-0f1e2d3c",
        "child_records": list(children) if children is not None else child_ids(cig_id),
        "stages": {
            "replay": {"measurements_ref": "children", "threshold": ">=0.6 and +0.4", "pass": True},
            "ablation": {"measurements_ref": "children", "threshold": "tau_c=+0.05", "pass": True},
            "generalization_in_scope": {"measurements_ref": "children", "threshold": ">=0", "pass": True},
            "generalization_out_of_scope": {"measurements_ref": "children", "threshold": ">=-0.05", "pass": True},
            "interaction_regression": {"measurements_ref": "children", "threshold": "<=0.03", "pass": True},
        },
        "verdict": "promote",
        "thresholds_used": {"version": "exp-0001-v2", "values": {"tau_c": 0.05, "tau_r": 0.03}},
        "artifacts": {},
    }


def child_record(cig_id="CIG-0042/S11-replay", stage="replay", seed=11, parent_cig_id="CIG-0042"):
    return {
        "cig_id": cig_id,
        "parent_cig_id": parent_cig_id,
        "seed": seed,
        "stage": stage,
        "measurements": {"successes_with": 3, "successes_without": 1, "n": 5},
    }


def m0_children(cig_id="CIG-0042", seed=SEED):
    """Bundle-ready M0 children: correct parent, coherent seed, and the
    aggregate's candidate_gene_id in every child."""
    children = []
    for stage in STAGES:
        child = child_record(cig_id=f"{cig_id}/S{seed}-{stage}", stage=stage,
                             seed=seed, parent_cig_id=cig_id)
        child["measurements"]["candidate_gene_id"] = "look_before_heat_v1"
        children.append(child)
    return children


def full_store(cig_id="CIG-0042"):
    """One M0 gate run: aggregate + the five per-stage children (one
    coherent seed), verified consistent."""
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(cig_id))
    for stage in STAGES:
        store.add_child(child_record(cig_id=f"{cig_id}/S{SEED}-{stage}",
                                     stage=stage, seed=SEED))
    store.require_consistent()
    return store


def somatic_envelope(state="candidate", gate_reports=None, registry=None):
    """Acquired somatic candidate fixture (issue #10 lifecycle contract)."""
    candidate = {
        "gene_id": "look_before_heat_v1", "version": 1, "type": "skill",
        "origin": "acquired",
        "artifact": fake_uri("skills", "look_before_heat_v1", 1),
        "provenance": {"born_generation": 1,
                        "source_trajectory": "T-aaaa1111aaaa"},
    }
    envelope = {"candidate": candidate, "validation": {"state": state}}
    if gate_reports is not None:
        envelope["validation"]["gate_reports"] = gate_reports
    return envelope


def test_two_level_cig_records_are_consistent():
    full_store()


def test_aggregate_referencing_missing_child_is_inconsistent():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record())  # references 5 children, none added
    issues = store.verify()
    assert any("missing child record" in i for i in issues)
    with pytest.raises(RecordConsistencyError):
        store.require_consistent()


def test_child_with_foreign_parent_prefix_rejected():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record())
    with pytest.raises(RecordConsistencyError, match="not a child"):
        store.add_child(child_record(cig_id="CIG-9999/S11-replay"))


def test_child_stage_must_exist_in_aggregate():
    broken = aggregate_record(cig_id="CIG-0043", children=("CIG-0043/S11",))
    broken["stages"] = {"replay": broken["stages"]["replay"]}
    store = CigRecordStore()
    store.add_aggregate(broken)
    store.add_child(child_record(cig_id="CIG-0043/S11", stage="ablation", parent_cig_id="CIG-0043"))
    assert any("not present in aggregate" in i for i in store.verify())


def test_duplicate_records_rejected():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record())
    with pytest.raises(RecordConsistencyError, match="already exists"):
        store.add_aggregate(aggregate_record())
    store.add_child(child_record())
    with pytest.raises(RecordConsistencyError, match="append-only"):
        store.add_child(child_record())


def test_child_requires_existing_parent():
    store = CigRecordStore()
    with pytest.raises(RecordConsistencyError, match="unknown aggregate"):
        store.add_child(child_record())


def test_cig_schema_rejects_placeholder_verdict():
    bad = aggregate_record()
    bad["verdict"] = "reject|promote"  # placeholder text from the design doc must not validate
    store = CigRecordStore()
    with pytest.raises(GenomeValidationError):
        store.add_aggregate(bad)


def test_duplicate_child_records_rejected_by_schema():
    bad = aggregate_record(children=("CIG-0042/S11-replay",) * 2)
    store = CigRecordStore()
    with pytest.raises(GenomeValidationError):
        store.add_aggregate(bad)


def test_undeclared_child_rejected_at_insert():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(children=("CIG-0042/S11-replay",)))
    with pytest.raises(RecordConsistencyError, match="not declared"):
        store.add_child(child_record(cig_id="CIG-0042/S23-ablation",
                                     stage="ablation", seed=23))


def test_orphan_child_detected_by_verify():
    store = full_store()
    ghost = child_record(cig_id="CIG-0042/S99-ghost", stage="ablation", seed=99,
                         parent_cig_id="CIG-0042")
    # white-box injection simulates externally corrupted evidence that
    # bypassed add_child; verify() must still catch it
    store._children[ghost["cig_id"]] = ghost
    issues = store.verify()
    assert any("orphan" in i for i in issues)
    with pytest.raises(RecordConsistencyError):
        store.require_consistent()


def test_seed_id_mismatch_rejected_at_insert():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(children=("CIG-0042/S11-replay",)))
    with pytest.raises(RecordConsistencyError, match="seed"):
        store.add_child(child_record(cig_id="CIG-0042/S11-replay", seed=23))


def test_seed_mismatch_detected_by_verify():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(children=("CIG-0042/S11-replay",)))
    store.add_child(child_record(cig_id="CIG-0042/S11-replay", seed=11))
    store._children["CIG-0042/S11-replay"]["seed"] = 23  # simulate post-insert corruption
    assert any("seed" in i for i in store.verify())


# -- M0 bundle coherence: add_bundle rejections + verify() parity ------------


def test_bundle_rejects_inexact_m0_child_id():
    """add_bundle enforces the exact <parent>/S<seed>-<stage> child id."""
    store = CigRecordStore()
    agg = aggregate_record()
    children = m0_children()
    wrong = copy_children(children)
    for child in wrong:
        if child["stage"] == "replay":
            child["cig_id"] = "CIG-0042/S11-replay-run2"
    agg["child_records"] = [c["cig_id"] for c in wrong]
    with pytest.raises(RecordConsistencyError, match="exact M0 form"):
        store.add_bundle(agg, wrong)


def test_bundle_rejects_duplicate_stage_child():
    """Exactly one child per M0 stage."""
    store = CigRecordStore()
    agg = aggregate_record()
    children = m0_children()
    ghost = child_record(cig_id="CIG-0042/S11-replay-mirror",
                          stage="replay", seed=SEED)
    ghost["measurements"]["candidate_gene_id"] = "look_before_heat_v1"
    agg["child_records"] = child_ids()[:1] + [ghost["cig_id"]] + child_ids()[1:]
    with pytest.raises(RecordConsistencyError, match="exactly one child"):
        store.add_bundle(agg, children + [ghost])


def test_bundle_rejects_incoherent_seed():
    """One CIG run = one coherent bundle seed."""
    store = CigRecordStore()
    agg = aggregate_record()
    children = m0_children()
    for child in children:
        if child["stage"] == "ablation":
            child["seed"] = 23
            child["cig_id"] = f"CIG-0042/S23-{child['stage']}"
    agg["child_records"] = [c["cig_id"] for c in children]
    with pytest.raises(RecordConsistencyError, match="incoherent seeds"):
        store.add_bundle(agg, children)


def test_bundle_rejects_incoherent_verdict():
    """verdict = promote iff every stage pass is true."""
    store = CigRecordStore()
    children = m0_children()
    agg = aggregate_record()
    agg["verdict"] = "reject"  # all stage passes True
    with pytest.raises(RecordConsistencyError, match="verdict"):
        store.add_bundle(agg, children)
    agg2 = aggregate_record()
    agg2["verdict"] = "promote"
    for stage in agg2["stages"]:
        agg2["stages"][stage]["pass"] = False
    with pytest.raises(RecordConsistencyError, match="verdict"):
        store.add_bundle(agg2, children)


def test_bundle_rejects_stage_pass_disagreement():
    store = CigRecordStore()
    agg = aggregate_record()
    children = m0_children()
    children[0]["measurements"]["stage_pass"] = False  # aggregate says pass
    with pytest.raises(RecordConsistencyError, match="stage_pass"):
        store.add_bundle(agg, children)


def test_verify_parity_for_m0_shape_corruptions():
    """Whatever add_bundle rejects must also be DETECTABLE by verify() when
    white-box injected into the stored state."""
    # duplicate stage child
    store = full_store()
    ghost = child_record(cig_id="CIG-0042/S23-replay", stage="replay", seed=23)
    store._children[ghost["cig_id"]] = ghost
    store._aggregates["CIG-0042"]["child_records"].append(ghost["cig_id"])
    issues = store.verify()
    assert any("expected exactly 1" in i for i in issues)
    # incoherent bundle seed
    store = full_store()
    store._children["CIG-0042/S11-ablation"]["seed"] = 23
    assert any("incoherent" in i for i in store.verify())
    # incoherent verdict
    store = full_store()
    store._aggregates["CIG-0042"]["verdict"] = "reject"
    assert any("verdict" in i for i in store.verify())
    # stage_pass disagreement
    store = full_store()
    store._children["CIG-0042/S11-replay"]["measurements"]["stage_pass"] = False
    assert any("stage_pass" in i for i in store.verify())
    # inexact M0 child id
    store = full_store()
    child = store._children.pop("CIG-0042/S11-replay")
    child["cig_id"] = "CIG-0042/S11-replay-x"
    store._children["CIG-0042/S11-replay-x"] = child
    store._aggregates["CIG-0042"]["child_records"][0] = "CIG-0042/S11-replay-x"
    assert any("exact M0 form" in i for i in store.verify())


def copy_children(children):
    import copy
    return copy.deepcopy(children)


def test_mutating_input_after_add_does_not_change_stored_aggregate():
    store = CigRecordStore()
    agg = aggregate_record(children=("CIG-0042/S11-replay",))
    store.add_aggregate(agg)
    agg["verdict"] = "reject"
    agg["child_records"].append("CIG-0042/S99")
    agg["stages"]["replay"]["pass"] = False
    stored = store.aggregates["CIG-0042"]
    assert stored["verdict"] == "promote"
    assert stored["child_records"] == ["CIG-0042/S11-replay"]
    assert stored["stages"]["replay"]["pass"] is True


def test_mutating_input_after_add_does_not_change_stored_child():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(children=("CIG-0042/S11-replay",)))
    child = child_record(cig_id="CIG-0042/S11-replay", seed=11)
    store.add_child(child)
    child["measurements"]["successes_with"] = 999
    child["seed"] = 41
    assert store.children["CIG-0042/S11-replay"]["measurements"]["successes_with"] == 3
    assert store.children["CIG-0042/S11-replay"]["seed"] == 11


def test_accessor_mutation_does_not_change_stored_state():
    store = full_store()
    leaked = store.aggregates["CIG-0042"]
    leaked["verdict"] = "reject"
    leaked["child_records"].append("CIG-0042/S00")
    leaked["stages"]["replay"]["pass"] = False
    leaked_child = store.children["CIG-0042/S11-replay"]
    leaked_child["measurements"]["successes_with"] = 999
    assert store.aggregates["CIG-0042"]["verdict"] == "promote"
    assert store.aggregates["CIG-0042"]["child_records"] == child_ids()
    assert store.aggregates["CIG-0042"]["stages"]["replay"]["pass"] is True
    assert store.children["CIG-0042/S11-replay"]["measurements"]["successes_with"] == 3
    assert store.verify() == []  # stored evidence untouched


def test_snapshot_isolates_staged_commits_from_live_store():
    """snapshot() clones the CURRENT state; add_bundle on the clone proves
    the live add would succeed without touching the live store."""
    live = CigRecordStore()
    agg = aggregate_record(cig_id="CIG-0043", children=child_ids("CIG-0043"))
    children = m0_children("CIG-0043")
    staged = live.snapshot()
    staged.add_bundle(agg, children)  # succeeds on the clone
    assert live._aggregates == {} and live._children == {}
    # the clone accepted it, so the live add must succeed deterministically
    live.add_bundle(agg, children)
    live.require_consistent()
    with pytest.raises(RecordConsistencyError, match="already exists"):
        live.snapshot().add_bundle(agg, children)


def test_mutating_input_after_add_does_not_change_stored_envelope():
    store = SomaticStore()
    envelope = somatic_envelope(state="candidate")
    store.add(envelope)
    envelope["validation"]["state"] = "validated"
    envelope["validation"]["gate_reports"] = ["CIG-fake"]
    assert store.state("look_before_heat_v1") == "candidate"
    assert store.all()[("look_before_heat_v1", 1)]["validation"] == \
        {"state": "candidate"}


def test_somatic_accessor_mutation_does_not_change_stored_state():
    store = SomaticStore()
    store.add(somatic_envelope(state="candidate"))
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0042"])
    leaked = store.all()
    leaked[("look_before_heat_v1", 1)]["validation"]["state"] = "rejected"
    assert store.state("look_before_heat_v1") == "validated"


def test_somatic_lifecycle_candidate_to_validated():
    store = SomaticStore()
    store.add(somatic_envelope(state="candidate"))
    assert store.state("look_before_heat_v1") == "candidate"
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0042"])
    assert store.state("look_before_heat_v1") == "validated"


def test_somatic_rejected_kept_as_somatic_memory():
    store = SomaticStore()
    store.add(somatic_envelope(state="candidate"))
    store.decide("look_before_heat_v1", 1, "rejected", ["CIG-0008"])
    assert store.state("look_before_heat_v1") == "rejected"
    assert list(store.all()) == [("look_before_heat_v1", 1)]  # kept, not discarded
    assert list(store.by_state("rejected")) == [("look_before_heat_v1", 1)]


def test_somatic_terminal_decision_is_immutable():
    store = SomaticStore()
    store.add(somatic_envelope(state="candidate"))
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0042"])
    with pytest.raises(RecordConsistencyError,
                        match="decision requires previous state 'candidate'"):
        store.decide("look_before_heat_v1", 1, "rejected", ["CIG-0008"])


def test_somatic_decision_requires_gate_report():
    store = SomaticStore()
    store.add(somatic_envelope(state="candidate"))
    with pytest.raises(RecordConsistencyError, match="non-empty gate_reports"):
        store.decide("look_before_heat_v1", 1, "validated", [])
