"""gate_reports replay/live parity + machine-addressable rejections (issue
#10 final re-review, group 2/3)."""

import json

import pytest

from genome.validation.errors import RecordConsistencyError
from somatic.store import SomaticStore
from somatic.journal import compute_record_digest
from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope


def re_sign_with_edit(path, edit):
    """Apply a semantic edit to the last record, then recompute ALL hashes
    (own + downstream chain) so the journal is hash-valid throughout — any
    rejection must therefore be on lifecycle/parity grounds."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    edit(records[-1])
    prev = None
    for r in records:
        r["prev_sha256"] = prev
        r["record_sha256"] = compute_record_digest(
            seq=r["seq"], op=r["op"], trait=r["trait"], envelope=r["envelope"],
            prev_sha256=r["prev_sha256"])
        prev = r["record_sha256"]
    path.write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"))
                               for r in records) + "\n")


def test_duplicate_gate_reports_rejected_on_replay(tmp_path, registry):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry))
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    store.close()
    re_sign_with_edit(path, lambda decision:
                       decision["envelope"]["validation"].update(
                           gate_reports=["CIG-0007", "CIG-0007"]))
    with pytest.raises(RecordConsistencyError) as exc_info:
        SomaticStore.open(path, registry)
    message = str(exc_info.value)
    assert "decision gate_reports contain duplicates" in message
    assert "record_sha256 mismatch" not in message  # hashes were valid


def test_unsorted_gate_reports_rejected_on_replay(tmp_path, registry):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry))
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    store.close()
    re_sign_with_edit(path, lambda decision:
                       decision["envelope"]["validation"].update(
                           gate_reports=["CIG-0009", "CIG-0002"]))  # unsorted
    with pytest.raises(RecordConsistencyError) as exc_info:
        SomaticStore.open(path, registry)
    message = str(exc_info.value)
    assert "decision gate_reports are not canonicalized" in message
    assert "record_sha256 mismatch" not in message


def test_canonical_gate_reports_accepted_on_replay(tmp_path, registry):
    """Sorted-unique gate_reports (exactly what live decide() writes) pass
    replay — parity in the accepted direction too."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry))
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    store.close()
    # rewrite the decision with canonical multi-report form, re-sign
    re_sign_with_edit(path, lambda decision:
                       decision["envelope"]["validation"].update(
                           gate_reports=["CIG-0002", "CIG-0009"]))
    reopened = SomaticStore.open(path, registry)
    assert reopened.state("look_before_heat_v1", 1) == "validated"
    assert reopened.get("look_before_heat_v1",
                         1)["validation"]["gate_reports"] == ["CIG-0002", "CIG-0009"]
    assert reopened.verify() == []
    reopened.close()


def test_rejected_entries_machine_addressable(tmp_path, registry, envelope):
    """invalid_applicability carries computed gene_id + proposal_index;
    exact_duplicate carries the SURVIVING gene_id; outside_cap carries
    gene_id + discovery_order."""
    from runtime.model_adapters import ScriptedAdapter
    from traits.miner.inputs import MiningBatch
    from traits.miner.miner import MiningSettings, TraitMiner
    from tests.miner.conftest import make_pair

    pair = make_pair(tmp_path, registry)
    proposals = [
        {   # invalid: family outside universe
            "name": "outside_skill",
            "principle": "p", "when_to_apply": "w",
            "applicability": {"task_families": ["quantum_chamber"]},
            "estimated_generality": 1.0,
        },
        {   # valid: accepted (discovery 0)
            "name": "kept_skill",
            "principle": "p", "when_to_apply": "w",
            "applicability": {"task_families": ["heat_and_place"]},
            "estimated_generality": 0.5,
        },
        {   # exact duplicate of kept_skill: SAME canonical payload (same
            # name), re-proposed with a higher generality
            "name": "kept_skill",
            "principle": "p", "when_to_apply": "w",
            "applicability": {"task_families": ["heat_and_place"]},
            "estimated_generality": 1.0,
        },
    ]
    miner = TraitMiner(model=ScriptedAdapter([{"proposals": proposals}]),
                        registry=registry, settings=MiningSettings(seed=11),
                        born_generation=1)
    batch = MiningBatch(pairs=(pair,), family_universe=("heat_and_place",))
    frozen = miner.mine(batch)

    rejected = frozen.rejected
    invalid = [r for r in rejected if r["reason"] == "invalid_applicability"][0]
    assert invalid["proposal_index"] == 0
    assert invalid["gene_id"].startswith("outside_skill_")
    assert invalid["discovery_order"] is None

    duplicate = [r for r in rejected if r["reason"] == "exact_duplicate"][0]
    assert duplicate["proposal_index"] == 2
    kept = frozen.candidates[0]["candidate"]["gene_id"]
    assert duplicate["gene_id"] == kept  # the SURVIVING gene id
    assert duplicate["discovery_order"] is None
    # same-name raw proposals are still unambiguously addressable via
    # proposal_index / gene_id / final disposition
    assert [r["proposal_name"] for r in rejected] == ["outside_skill", "kept_skill"]
    # first VALID discovery: invalid_applicability proposals never consume
    # a discovery slot, so kept_skill owns order 0
    assert frozen.mining_records[0]["accepted_proposals"][0]["discovery_order"] == 0
