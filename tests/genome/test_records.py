"""Somatic lifecycle + two-level CIG record stores (criterion 4)."""

import pytest

from genome.validation.errors import GenomeValidationError, RecordConsistencyError
from genome.validation.records import CigRecordStore, SomaticStore
from tests.conftest import fake_uri, minimal_genome

STAGES = ("replay", "ablation", "generalization_in_scope",
          "generalization_out_of_scope", "interaction_regression")
SEED = 11


CANDIDATE_REF = {"gene_id": "look_before_heat_v1", "version": 1,
                 "artifact": fake_uri("skills", "look_before_heat_v1", 1)}
BINDING = {"candidate_version": 1, "candidate_artifact": CANDIDATE_REF["artifact"],
           "base_genome_id": "G-0f1e2d3c"}


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
        # the evaluated artifact and base genome (ADR-0010 B1)
        "artifacts": {"candidate": dict(CANDIDATE_REF), "base_genome": {"genome_id": "G-0f1e2d3c"}},
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
    mandatory candidate_gene_id, artifact and base genome binding (ADR-0010)
    + stage_pass in every child."""
    children = []
    for stage in STAGES:
        child = child_record(cig_id=f"{cig_id}/S{seed}-{stage}", stage=stage,
                             seed=seed, parent_cig_id=cig_id)
        child["measurements"]["candidate_gene_id"] = "look_before_heat_v1"
        child["measurements"].update(BINDING)
        child["measurements"]["stage_pass"] = True
        children.append(child)
    return children


def full_store(cig_id="CIG-0042"):
    """One M0 gate run: aggregate + the five per-stage children (one
    coherent seed, mandatory stage_pass + candidate binding), verified
    consistent."""
    store = CigRecordStore()
    store.add_aggregate(aggregate_record(cig_id))
    for child in m0_children(cig_id):
        store.add_child(child)
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


def test_bundle_rejects_missing_stage_pass():
    """stage_pass is MANDATORY in M0 bundle children: a child without it is
    rejected even though the aggregate stage pass would 'agree'."""
    store = CigRecordStore()
    agg = aggregate_record()
    children = m0_children()
    del children[2]["measurements"]["stage_pass"]
    with pytest.raises(RecordConsistencyError, match="stage_pass"):
        store.add_bundle(agg, children)


def test_bundle_rejects_non_boolean_stage_pass():
    """stage_pass must be a STRICT boolean in M0 children: truthy non-bool
    values (1, \"true\", {}, ...) are rejected — no bool() coercion."""
    store = CigRecordStore()
    agg = aggregate_record()
    for bad in (1, "true", {}, 1.0):
        children = m0_children()
        children[0]["measurements"]["stage_pass"] = bad
        with pytest.raises(RecordConsistencyError, match="strict boolean"):
            store.add_bundle(agg, children)


def test_bundle_rejects_wrong_or_missing_candidate_gene_id():
    store = CigRecordStore()
    agg = aggregate_record()
    children = m0_children()
    children[1]["measurements"]["candidate_gene_id"] = "other_gene"
    with pytest.raises(RecordConsistencyError, match="candidate_gene_id"):
        store.add_bundle(agg, children)
    children = m0_children()
    del children[3]["measurements"]["candidate_gene_id"]
    with pytest.raises(RecordConsistencyError, match="candidate_gene_id"):
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
    # MISSING stage_pass (white-box corruption of a stored M0 child)
    store = full_store()
    del store._children["CIG-0042/S11-replay"]["measurements"]["stage_pass"]
    assert any("missing required stage_pass" in i for i in store.verify())
    with pytest.raises(RecordConsistencyError):
        store.require_consistent()
    # NON-BOOLEAN stage_pass (white-box corruption): 1 / "true" / {} caught
    for bad in (1, "true", {}):
        store = full_store()
        store._children["CIG-0042/S11-replay"]["measurements"]["stage_pass"] = bad
        assert any("strict boolean" in i for i in store.verify())
        with pytest.raises(RecordConsistencyError):
            store.require_consistent()
    # wrong candidate_gene_id binding
    store = full_store()
    store._children["CIG-0042/S11-replay"]["measurements"]["candidate_gene_id"] = \
        "other_gene"
    assert any("candidate_gene_id" in i for i in store.verify())
    # MISSING candidate_gene_id binding
    store = full_store()
    del store._children["CIG-0042/S11-replay"]["measurements"]["candidate_gene_id"]
    assert any("candidate_gene_id" in i for i in store.verify())
    with pytest.raises(RecordConsistencyError):
        store.require_consistent()
    # inexact M0 child id
    store = full_store()
    child = store._children.pop("CIG-0042/S11-replay")
    child["cig_id"] = "CIG-0042/S11-replay-x"
    store._children["CIG-0042/S11-replay-x"] = child
    store._aggregates["CIG-0042"]["child_records"][0] = "CIG-0042/S11-replay-x"
    assert any("exact M0 form" in i for i in store.verify())


# -- ADR-0010 B4: artifact and base genome binding ---------------------------------
PARTNER_REF = {"gene_id": "check_door_v1", "version": 3,
               "artifact": fake_uri("skills", "check_door_v1", 3)}


def pair_entry(own=CANDIDATE_REF, partner=PARTNER_REF, **canonical_changes):
    """One stage-4 pair evidence entry as the gate writes it: the canonical
    record (genes in sorted order, one reference per gene) and its digest."""
    import copy
    from trajectory.storage.canonical import canonical_json, sha256_hex
    refs = sorted([own, partner], key=lambda ref: ref["gene_id"])
    canonical = {"kind": "heritagent-interaction-pair",
                 "gene_ids": [ref["gene_id"] for ref in refs],
                 "artifacts": copy.deepcopy(refs), "delta_pair": 0.0, "harmful": False}
    canonical.update(canonical_changes)
    return {"evidence_sha256": sha256_hex(canonical_json(canonical).encode("utf-8")),
            "canonical_evidence": canonical, "other_gene_id": partner["gene_id"],
            "delta_pair": 0.0, "harmful": False, "retained": True}


def pair_bundle(cig_id="CIG-0042", ref=CANDIDATE_REF, entries=None):
    """An M0 bundle of candidate ``ref`` whose interaction_regression child
    carries ``entries`` (default: one pair with the other reference)."""
    import copy
    agg = aggregate_record(cig_id)
    agg["candidate_gene_id"] = ref["gene_id"]
    agg["artifacts"]["candidate"] = copy.deepcopy(ref)
    children = m0_children(cig_id)
    for child in children:
        child["measurements"].update(candidate_gene_id=ref["gene_id"],
                                     candidate_version=ref["version"],
                                     candidate_artifact=ref["artifact"])
    if entries is None:
        partner = PARTNER_REF if ref is CANDIDATE_REF else CANDIDATE_REF
        entries = [pair_entry(ref, partner)]
    children[-1]["measurements"]["interaction"] = {"pair_evidence": entries}
    return agg, children


def paired_store():
    """Both copies of one pair evidence, in the two candidates' bundles."""
    store = CigRecordStore()
    store.add_bundle(*pair_bundle("CIG-0042", CANDIDATE_REF))
    store.add_bundle(*pair_bundle("CIG-0043", PARTNER_REF))
    store.require_consistent()
    return store


def test_bundle_with_both_copies_of_a_pair_evidence_is_consistent():
    store = CigRecordStore()
    store.add_bundle(*pair_bundle("CIG-0042", CANDIDATE_REF))
    # one copy alone: the partner's bundle is not there yet
    assert any("names partner 'check_door_v1'" in i for i in store.verify())
    store.add_bundle(*pair_bundle("CIG-0043", PARTNER_REF))
    store.require_consistent()


@pytest.mark.parametrize("change, match", [
    (lambda a: a["artifacts"].pop("candidate"), "artifacts.candidate None is not"),
    (lambda a: a["artifacts"]["candidate"].update(version=True), "not a strict int"),
    (lambda a: a["artifacts"]["candidate"].update(version="1"), "not a strict int"),
    (lambda a: a["artifacts"]["candidate"].update(artifact="registry://skills/x"),
     "is not a registry URI"),
    (lambda a: a["artifacts"]["candidate"].update(artifact=7), "is not a registry URI"),
    (lambda a: a["artifacts"]["candidate"].update(
        artifact=fake_uri("skills", "look_before_heat_v1", 2)), "does not name gene"),
    (lambda a: a["artifacts"]["candidate"].update(
        artifact=fake_uri("skills", "other_gene", 1)), "does not name gene"),
    (lambda a: a["artifacts"].update(candidate=dict(PARTNER_REF)),
     "gene_id 'check_door_v1' != its candidate_gene_id"),
    (lambda a: a["artifacts"].pop("base_genome"), "base_genome.genome_id None"),
    (lambda a: a["artifacts"].update(base_genome={"genome_id": "G-99999999"}),
     "base_genome.genome_id 'G-99999999'"),
    (lambda a: a.pop("artifacts"), "artifacts.candidate None is not"),
])
def test_bundle_needs_complete_coherent_references(change, match):
    store = CigRecordStore()
    agg = aggregate_record()
    change(agg)
    with pytest.raises(RecordConsistencyError, match=match):
        store.add_bundle(agg, m0_children())
    assert store.aggregates == {} and store.children == {}


@pytest.mark.parametrize("key, bad", [
    ("candidate_version", 2), ("candidate_version", True), ("candidate_version", "1"),
    ("candidate_artifact", fake_uri("skills", "look_before_heat_v1", 1, "b" * 64)),
    ("base_genome_id", "G-99999999")])
@pytest.mark.parametrize("stage_index", range(len(STAGES)))
def test_bundle_rejects_a_child_bound_to_another_artifact_or_genome(key, bad, stage_index):
    store = CigRecordStore()
    children = m0_children()
    children[stage_index]["measurements"][key] = bad
    with pytest.raises(RecordConsistencyError, match=f"{key} .* \\(ADR-0010\\)"):
        store.add_bundle(aggregate_record(), children)
    children[stage_index]["measurements"].pop(key)  # a missing field is a mismatch
    with pytest.raises(RecordConsistencyError, match=f"{key} None"):
        store.add_bundle(aggregate_record(), children)
    assert store.aggregates == {}


def _rehashed(entry):
    from trajectory.storage.canonical import canonical_json, sha256_hex
    entry["evidence_sha256"] = sha256_hex(
        canonical_json(entry["canonical_evidence"]).encode("utf-8"))
    return entry


def _self_pair(entry):
    """A self-consistent pair of the candidate with itself (digest recomputed)."""
    gene = CANDIDATE_REF["gene_id"]
    entry["canonical_evidence"].update(gene_ids=[gene, gene],
                                       artifacts=[dict(CANDIDATE_REF), dict(CANDIDATE_REF)])
    entry["other_gene_id"] = gene
    return _rehashed(entry)


def _three_genes(entry):
    """A self-consistent evidence naming three genes (digest recomputed)."""
    third = {"gene_id": "zz_third_v1", "version": 1,
             "artifact": fake_uri("skills", "zz_third_v1", 1)}
    entry["canonical_evidence"].update(
        gene_ids=[PARTNER_REF["gene_id"], CANDIDATE_REF["gene_id"], third["gene_id"]],
        artifacts=[dict(PARTNER_REF), dict(CANDIDATE_REF), third])
    return _rehashed(entry)


@pytest.mark.parametrize("corrupt, match", [
    (lambda e: e.update(evidence_sha256="0" * 64), "!= the digest of its canonical_evidence"),
    (_self_pair, "are not two distinct genes including"),
    (_three_genes, "are not two distinct genes including"),
    (lambda e: _rehashed(e) if e["canonical_evidence"].update(
        gene_ids=[CANDIDATE_REF["gene_id"]], artifacts=[dict(CANDIDATE_REF)]) is None else None,
     "are not two distinct genes including"),
    (lambda e: e["canonical_evidence"].update(delta_pair=0.5),
     "!= the digest of its canonical_evidence"),
    # the candidate's own entry names another version (digest recomputed)
    (lambda e: _rehashed(e) if e["canonical_evidence"]["artifacts"][1].update(
        version=2, artifact=fake_uri("skills", "look_before_heat_v1", 2)) is None else None,
     "entry for the candidate"),
    # the partner's entry is incoherent with its URI
    (lambda e: _rehashed(e) if e["canonical_evidence"]["artifacts"][0].update(
        version=4) is None else None, "artifact for 'check_door_v1'"),
    (lambda e: _rehashed(e) if e["canonical_evidence"]["artifacts"].reverse() is None
     else None, "where gene_ids order names"),
    (lambda e: _rehashed(e) if e["canonical_evidence"]["artifacts"].pop() else None,
     "are not one reference per gene"),
    (lambda e: _rehashed(e) if e["canonical_evidence"].pop("artifacts") else None,
     "are not one reference per gene"),
    (lambda e: _rehashed(e) if e["canonical_evidence"].update(
        gene_ids=["check_door_v1", "other_gene"]) is None else None,
     "are not two distinct genes including"),
    (lambda e: e.update(other_gene_id="other_gene"), "other_gene_id 'other_gene' != the partner"),
    (lambda e: e.pop("canonical_evidence"), "carries no canonical_evidence object"),
    (lambda e: e["canonical_evidence"].update(delta_pair=float("nan")),
     "is not canonical JSON"),
])
def test_bundle_rejects_incoherent_pair_evidence(corrupt, match):
    store = CigRecordStore()
    agg, children = pair_bundle()
    corrupt(children[-1]["measurements"]["interaction"]["pair_evidence"][0])
    with pytest.raises(RecordConsistencyError, match=match):
        store.add_bundle(agg, children)
    assert store.aggregates == {}


@pytest.mark.parametrize("interaction, match", [
    ([], "interaction \\[\\] is not an object"),
    ({"pair_evidence": {}}, "pair_evidence \\{\\} is not a list"),
    ({"pair_evidence": ["x"]}, "carries no canonical_evidence object"),
])
def test_bundle_rejects_malformed_interaction_measurements(interaction, match):
    store = CigRecordStore()
    agg, children = pair_bundle()
    children[-1]["measurements"]["interaction"] = interaction
    with pytest.raises(RecordConsistencyError, match=match):
        store.add_bundle(agg, children)


def test_verify_parity_for_artifact_binding_corruptions():
    """Whatever add_bundle rejects under ADR-0010 is also detected by verify()
    in stored state, and the two copies of a pair evidence are linked."""
    def corrupted(change):
        store = paired_store()
        change(store)
        issues = store.verify()
        with pytest.raises(RecordConsistencyError):
            store.require_consistent()
        return " | ".join(issues)

    assert "not a strict int" in corrupted(
        lambda s: s._aggregates["CIG-0042"]["artifacts"]["candidate"].update(version="1"))
    assert "base_genome.genome_id None" in corrupted(
        lambda s: s._aggregates["CIG-0042"]["artifacts"].pop("base_genome"))
    assert "candidate_version 2" in corrupted(
        lambda s: s._children["CIG-0042/S11-ablation"]["measurements"].update(
            candidate_version=2))
    assert "base_genome_id None" in corrupted(
        lambda s: s._children["CIG-0042/S11-replay"]["measurements"].pop("base_genome_id"))
    stage4 = "CIG-0042/S11-interaction_regression"

    def entry(store):
        return store._children[stage4]["measurements"]["interaction"]["pair_evidence"][0]

    assert "!= the digest" in corrupted(lambda s: entry(s).update(delta_pair=1.0) or
                                        entry(s)["canonical_evidence"].update(delta_pair=1.0))
    # one copy rewritten self-consistently: its digest no longer matches the
    # partner's copy, so BOTH copies lose their link
    issues = corrupted(lambda s: _rehashed(entry(s)) if entry(s)["canonical_evidence"].update(
        delta_pair=1.0) is None else None)
    assert "of 'look_before_heat_v1' names partner 'check_door_v1'" in issues
    assert "of 'check_door_v1' names partner 'look_before_heat_v1'" in issues
    # one copy dropped
    assert "of 'check_door_v1' names partner" in corrupted(
        lambda s: s._children[stage4]["measurements"]["interaction"].update(pair_evidence=[]))
    # the partner's copy sits in a non-M0 aggregate: no link
    def non_m0(store):
        store._aggregates["CIG-0043"]["stages"].pop("replay")
    assert "of 'look_before_heat_v1' names partner" in corrupted(non_m0)


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
