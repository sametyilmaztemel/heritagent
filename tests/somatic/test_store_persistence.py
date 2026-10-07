"""Persistence/integrity tests: journal round trips, hash chain, tamper
detection, create/open semantics (issue #10 criteria 3-5, 12)."""

import json

import pytest

from genome.validation.errors import RecordConsistencyError
from somatic.journal import compute_record_digest
from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope


def build_two_decision_journal(tmp_path, registry):
    """One store: candidate → validated (trait 1) and candidate → rejected
    (trait 2); returns (path, registry)."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry, gene_id="trait_one_v1"))
    store.decide("trait_one_v1", 1, "validated", ["CIG-0007"])
    store.add_candidate(acquired_envelope(registry, gene_id="trait_two_v1"))
    store.decide("trait_two_v1", 1, "rejected", ["CIG-0008"])
    store.close()
    return path, registry


def test_create_insert_reopen_identical_state(tmp_path, registry):
    path, registry = build_two_decision_journal(tmp_path, registry)
    reopened = SomaticStore.open(path, registry)
    assert reopened.state("trait_one_v1", 1) == "validated"
    assert reopened.state("trait_two_v1", 1) == "rejected"
    assert reopened.verify() == []
    reopened.close()


def test_rejected_trait_preserved_across_reopen(tmp_path, registry):
    path, registry = build_two_decision_journal(tmp_path, registry)
    reopened = SomaticStore.open(path, registry)
    rejected = reopened.by_state("rejected")
    assert list(rejected) == [("trait_two_v1", 1)]  # retained, queryable
    # history shows candidate → rejected transition, original payload intact
    history = reopened.history("trait_two_v1", 1)
    assert [e["validation"]["state"] for e in history] == ["candidate", "rejected"]
    assert history[1]["candidate"] == history[0]["candidate"]
    reopened.close()


def test_validated_trait_preserved_across_reopen(tmp_path, registry):
    path, registry = build_two_decision_journal(tmp_path, registry)
    reopened = SomaticStore.open(path, registry)
    history = reopened.history("trait_one_v1", 1)
    assert [e["validation"]["state"] for e in history] == ["candidate", "validated"]
    # exact original artifact URI preserved through the decision
    assert history[1]["candidate"]["artifact"] == history[0]["candidate"]["artifact"]
    reopened.close()


def test_journal_lines_are_canonical_json(tmp_path, registry):
    from trajectory.storage.canonical import canonical_json
    path, _ = build_two_decision_journal(tmp_path, registry)
    for line in path.read_text().splitlines():
        assert line == canonical_json(json.loads(line))


def test_record_digest_deterministic(tmp_path, registry):
    envelope = acquired_envelope(registry)
    digest_a = compute_record_digest(seq=1, op="candidate_added",
                                      trait={"gene_id": "x_v1", "version": 1},
                                      envelope=envelope, prev_sha256=None)
    digest_b = compute_record_digest(seq=1, op="candidate_added",
                                      trait={"gene_id": "x_v1", "version": 1},
                                      envelope=envelope, prev_sha256=None)
    assert digest_a == digest_b


def test_hash_chain_detected_on_tampered_payload(tmp_path, registry):
    path, registry = build_two_decision_journal(tmp_path, registry)
    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    records[0]["envelope"]["candidate"]["provenance"]["born_generation"] = 42
    lines[0] = json.dumps(records[0], sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RecordConsistencyError, match="record_sha256 mismatch"):
        SomaticStore.open(path, registry)


def test_reordered_records_rejected(tmp_path, registry):
    path, registry = build_two_decision_journal(tmp_path, registry)
    lines = path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]  # swap two event records
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RecordConsistencyError, match="seq"):
        SomaticStore.open(path, registry)


def test_missing_intermediate_record_rejected(tmp_path, registry):
    path, registry = build_two_decision_journal(tmp_path, registry)
    lines = path.read_text().splitlines()
    del lines[1]  # drop the first event record
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RecordConsistencyError):
        SomaticStore.open(path, registry)


def test_wrong_schema_version_rejected(tmp_path, registry):
    path, registry = build_two_decision_journal(tmp_path, registry)
    lines = path.read_text().splitlines()
    header_record = json.loads(lines[0])
    header_record["schema_version"] = "9.9"
    lines[0] = json.dumps(header_record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RecordConsistencyError, match="schema violations"):
        SomaticStore.open(path, registry)


def test_non_finite_json_rejected(tmp_path, registry):
    path, registry = build_two_decision_journal(tmp_path, registry)
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["envelope"]["candidate"]["provenance"]["score"] = float("nan")
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=True)
    path.write_text("\n".join(lines) + "\n")
    from trajectory.recorder.errors import MalformedTrajectoryError
    with pytest.raises(MalformedTrajectoryError, match="non-standard JSON"):
        SomaticStore.open(path, registry)


def test_new_store_does_not_overwrite_existing_journal(tmp_path, registry):
    path, registry = build_two_decision_journal(tmp_path, registry)
    before = path.read_bytes()
    from trajectory.recorder.errors import RecorderError
    with pytest.raises(RecorderError, match="path already exists"):
        SomaticStore.create(path, registry)
    assert path.read_bytes() == before  # somatic memory never truncated


def test_decision_artifact_swap_rejected_by_replay(tmp_path, registry):
    """A decision whose envelope swapped in another trait's artifact is
    rejected on replay: the URI name/gene_id semantic check catches it, and
    the downstream chain breaks."""
    path, registry = build_two_decision_journal(tmp_path, registry)
    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    decision = records[1]  # trait_one validated decision
    decision["envelope"]["candidate"]["artifact"] = (
        acquired_envelope(registry, gene_id="trait_two_v1")["candidate"]["artifact"])
    decision["prev_sha256"] = records[0]["record_sha256"]
    decision["record_sha256"] = compute_record_digest(
        seq=decision["seq"], op=decision["op"], trait=decision["trait"],
        envelope=decision["envelope"], prev_sha256=decision["prev_sha256"])
    lines[1] = json.dumps(decision, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RecordConsistencyError) as exc_info:
        SomaticStore.open(path, registry)  # replay verification fails closed
    assert "artifact URI name" in str(exc_info.value)


def test_verify_detects_candidate_mutation_in_decision(tmp_path, registry):
    """A decision record that mutated candidate PROVENANCE (same gene name)
    is caught by the mutated-candidate replay check."""
    path, registry = build_two_decision_journal(tmp_path, registry)
    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    decision = records[1]
    decision["envelope"]["candidate"]["provenance"]["born_generation"] = 99
    decision["record_sha256"] = compute_record_digest(
        seq=decision["seq"], op=decision["op"], trait=decision["trait"],
        envelope=decision["envelope"], prev_sha256=decision["prev_sha256"])
    lines[1] = json.dumps(decision, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RecordConsistencyError) as exc_info:
        SomaticStore.open(path, registry)
    assert "mutated the candidate" in str(exc_info.value)
