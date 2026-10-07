"""Replay-lifecycle-parity tests (issue #10 re-review, group 1): semantic
records with VALID recomputed hashes must still be rejected by the
centralized lifecycle validation — tests target lifecycle enforcement, not
hash mismatches."""

import json

import pytest

from genome.validation.errors import RecordConsistencyError
from somatic.journal import compute_record_digest
from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope


def resign_record(record: dict) -> dict:
    """Recompute record_sha256 AND every downstream prev/record hash, so a
    semantically edited journal has a fully valid hash chain — replay must
    still reject it on lifecycle grounds."""
    lines = [json.loads(line) for line in
             (record["_path"]).read_text().splitlines()]
    index = records_index = None
    records = lines
    start = next(i for i, r in enumerate(records) if r["seq"] == record["seq"])
    prev = records[start - 1]["record_sha256"] if start > 0 else None
    for i in range(start, len(records)):
        r = records[i]
        r["prev_sha256"] = prev
        r["record_sha256"] = compute_record_digest(
            seq=r["seq"], op=r["op"], trait=r["trait"], envelope=r["envelope"],
            prev_sha256=r["prev_sha256"])
        prev = r["record_sha256"]
    record["_path"].write_text(
        "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"))
                   for r in records) + "\n")


def test_candidate_added_with_seed_origin_rejected_on_replay(tmp_path, registry):
    """A candidate_added record edited to origin=seed with a FULLY VALID
    recomputed hash chain is still rejected — replay enforces the same
    acquired-candidate contract as the live write path."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry))
    store.close()
    re_sign_with_edit(path, lambda r: r["envelope"]["candidate"].update(origin="seed"))
    with pytest.raises(RecordConsistencyError) as exc_info:
        SomaticStore.open(path, registry)
    message = str(exc_info.value)
    assert "candidate_added requires origin 'acquired'" in message
    assert "record_sha256 mismatch" not in message  # hashes were valid


def test_decision_back_to_candidate_rejected_on_replay(tmp_path, registry):
    """A decision_recorded whose resulting state rolls BACK to candidate
    (valid recomputed hashes) is rejected — decisions only go to terminal
    verdicts."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry))
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    store.close()
    re_sign_with_edit(path, lambda decision:
                       decision["envelope"]["validation"].update(state="candidate"))
    with pytest.raises(RecordConsistencyError) as exc_info:
        SomaticStore.open(path, registry)
    message = str(exc_info.value)
    assert "decision requires resulting state" in message
    assert "record_sha256 mismatch" not in message


def test_decision_without_gate_reports_rejected_on_replay(tmp_path, registry):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry))
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    store.close()
    re_sign_with_edit(path, lambda decision:
                       decision["envelope"]["validation"].update(gate_reports=[]))
    with pytest.raises(RecordConsistencyError) as exc_info:
        SomaticStore.open(path, registry)
    assert "decision requires non-empty gate_reports" in str(exc_info.value)


def re_sign_with_edit(path: Path, edit):
    """Apply a semantic edit to the last record, then recompute ALL hashes
    (own + downstream chain) so the journal is hash-valid throughout."""
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
