"""Somatic lifecycle + two-level CIG record stores (criterion 4)."""

import pytest

from genome.validation.errors import GenomeValidationError, RecordConsistencyError
from genome.validation.records import CigRecordStore, SomaticStore
from tests.conftest import fake_uri, minimal_genome

STAGES = ("replay", "ablation", "generalization_in_scope",
          "generalization_out_of_scope", "interaction_regression")
STAGE_SUFFIX = {"replay": "", "ablation": "-ab", "generalization_in_scope": "-gi",
                "generalization_out_of_scope": "-go", "interaction_regression": "-ir"}
SEEDS = (11, 23, 41)


def child_ids(cig_id="CIG-0042"):
    """15 children: 5 stages x 3 seeds, ids /S<seed><stage-suffix>."""
    return [f"{cig_id}/S{seed}{STAGE_SUFFIX[stage]}" for seed in SEEDS for stage in STAGES]


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


def child_record(cig_id="CIG-0042/S11", stage="replay", seed=11, parent_cig_id="CIG-0042"):
    return {
        "cig_id": cig_id,
        "parent_cig_id": parent_cig_id,
        "seed": seed,
        "stage": stage,
        "measurements": {"successes_with": 3, "successes_without": 1, "n": 5},
    }


def full_store(cig_id="CIG-0042"):
    """Aggregate + all 15 per-(seed, stage) children, verified consistent."""
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(cig_id))
    for seed in SEEDS:
        for stage in STAGES:
            store.add_child(child_record(cig_id=f"{cig_id}/S{seed}{STAGE_SUFFIX[stage]}",
                                         stage=stage, seed=seed))
    store.require_consistent()
    return store


def somatic_envelope(state="candidate", gate_reports=None):
    genome = minimal_genome()
    candidate = genome["genes"]["cognition"]["planner"]
    candidate["gene_id"] = "look_before_heat_v1"
    candidate["type"] = "skill"
    candidate["artifact"] = fake_uri("skills", "look_before_heat_v1", 1)
    envelope = {"candidate": candidate, "validation": {"state": state}}
    if gate_reports is not None:
        envelope["validation"]["gate_reports"] = gate_reports
    return envelope


def test_two_level_cig_records_are_consistent():
    full_store()


def test_aggregate_referencing_missing_child_is_inconsistent():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record())  # references 15 children, none added
    issues = store.verify()
    assert any("missing child record" in i for i in issues)
    with pytest.raises(RecordConsistencyError):
        store.require_consistent()


def test_child_with_foreign_parent_prefix_rejected():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record())
    with pytest.raises(RecordConsistencyError, match="not a child"):
        store.add_child(child_record(cig_id="CIG-9999/S11"))


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
    bad = aggregate_record(children=("CIG-0042/S11", "CIG-0042/S11"))
    store = CigRecordStore()
    with pytest.raises(GenomeValidationError):
        store.add_aggregate(bad)


def test_undeclared_child_rejected_at_insert():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(children=("CIG-0042/S11",)))
    with pytest.raises(RecordConsistencyError, match="not declared"):
        store.add_child(child_record(cig_id="CIG-0042/S23", stage="ablation", seed=23))


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
    store.add_aggregate(aggregate_record(children=("CIG-0042/S11",)))
    with pytest.raises(RecordConsistencyError, match="seed"):
        store.add_child(child_record(cig_id="CIG-0042/S11", seed=23))


def test_seed_mismatch_detected_by_verify():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(children=("CIG-0042/S11",)))
    store.add_child(child_record(cig_id="CIG-0042/S11", seed=11))
    store._children["CIG-0042/S11"]["seed"] = 23  # simulate post-insert corruption
    assert any("seed" in i for i in store.verify())


def test_mutating_input_after_add_does_not_change_stored_aggregate():
    store = CigRecordStore()
    agg = aggregate_record(children=("CIG-0042/S11",))
    store.add_aggregate(agg)
    agg["verdict"] = "reject"
    agg["child_records"].append("CIG-0042/S99")
    agg["stages"]["replay"]["pass"] = False
    stored = store.aggregates["CIG-0042"]
    assert stored["verdict"] == "promote"
    assert stored["child_records"] == ["CIG-0042/S11"]
    assert stored["stages"]["replay"]["pass"] is True


def test_mutating_input_after_add_does_not_change_stored_child():
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(children=("CIG-0042/S11",)))
    child = child_record(cig_id="CIG-0042/S11", seed=11)
    store.add_child(child)
    child["measurements"]["successes_with"] = 999
    child["seed"] = 41
    assert store.children["CIG-0042/S11"]["measurements"]["successes_with"] == 3
    assert store.children["CIG-0042/S11"]["seed"] == 11


def test_accessor_mutation_does_not_change_stored_state():
    store = full_store()
    leaked = store.aggregates["CIG-0042"]
    leaked["verdict"] = "reject"
    leaked["child_records"].append("CIG-0042/S00")
    leaked["stages"]["replay"]["pass"] = False
    leaked_child = store.children["CIG-0042/S11"]
    leaked_child["measurements"]["successes_with"] = 999
    assert store.aggregates["CIG-0042"]["verdict"] == "promote"
    assert store.aggregates["CIG-0042"]["child_records"] == child_ids()
    assert store.aggregates["CIG-0042"]["stages"]["replay"]["pass"] is True
    assert store.children["CIG-0042/S11"]["measurements"]["successes_with"] == 3
    assert store.verify() == []  # stored evidence untouched


def test_mutating_input_after_add_does_not_change_stored_envelope():
    store = SomaticStore()
    envelope = somatic_envelope(state="candidate")
    store.add(envelope)
    envelope["validation"]["state"] = "validated"
    envelope["validation"]["gate_reports"] = ["CIG-fake"]
    assert store.state("look_before_heat_v1") == "candidate"
    assert store.all()["look_before_heat_v1"]["validation"] == {"state": "candidate"}


def test_somatic_accessor_mutation_does_not_change_stored_state():
    store = SomaticStore()
    store.add(somatic_envelope(state="validated", gate_reports=["CIG-0042"]))
    leaked = store.all()
    leaked["look_before_heat_v1"]["validation"]["state"] = "rejected"
    assert store.state("look_before_heat_v1") == "validated"


def test_somatic_lifecycle_candidate_to_validated():
    store = SomaticStore()
    store.add(somatic_envelope(state="candidate"))
    assert store.state("look_before_heat_v1") == "candidate"
    store.add(somatic_envelope(state="validated", gate_reports=["CIG-0042"]))
    assert store.state("look_before_heat_v1") == "validated"


def test_somatic_rejected_kept_as_somatic_memory():
    store = SomaticStore()
    store.add(somatic_envelope(state="candidate"))
    store.add(somatic_envelope(state="rejected", gate_reports=["CIG-0008"]))
    assert store.state("look_before_heat_v1") == "rejected"
    assert list(store.all()) == ["look_before_heat_v1"]  # kept, not silently discarded


def test_somatic_terminal_decision_is_immutable():
    store = SomaticStore()
    store.add(somatic_envelope(state="validated", gate_reports=["CIG-0042"]))
    with pytest.raises(RecordConsistencyError, match="immutable"):
        store.add(somatic_envelope(state="candidate"))


def test_somatic_decision_requires_gate_report():
    store = SomaticStore()
    with pytest.raises(GenomeValidationError, match="gate_reports"):
        store.add(somatic_envelope(state="validated", gate_reports=[]))
