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
