"""Same-count/same-tail payload tamper detection (issue #10 final re-review,
group 1): the append guard must compare the canonical parsed journal content
against the handle's cached records — not just count + tail string."""

import json

import pytest

from somatic.store import SomaticStore
from trajectory.recorder.errors import RecorderError
from tests.somatic.conftest import acquired_envelope


def test_payload_tamper_with_unchanged_count_and_tail_rejected(tmp_path, registry, envelope):
    """Externally edit an OLD record's payload WITHOUT changing the
    record_sha256 field: count and tail string stay identical, but the
    append must FAIL and the journal bytes must remain untouched."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    store.add_candidate(acquired_envelope(registry, gene_id="second_v1"))
    # external tamper: edit the FIRST record's payload, keep its
    # record_sha256 field untouched (count + tail unchanged)
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["envelope"]["candidate"]["provenance"]["source_trajectory"] = "T-tampered"
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    bytes_after_tamper = path.read_bytes()  # tamper persisted; append must not add

    with pytest.raises(RecorderError) as exc_info:
        store.add_candidate(acquired_envelope(registry, gene_id="third_v1"))
    message = str(exc_info.value)
    assert "stale somatic store handle" in message
    assert "journal content diverged" in message
    assert "first divergence at record 1" in message
    # the FAILED APPEND changed nothing: bytes identical to post-tamper state
    # (still 2 records, no third)
    assert path.read_bytes() == bytes_after_tamper
    assert len(path.read_text().splitlines()) == 2


def test_tampered_record_sha256_field_also_caught(tmp_path, registry, envelope):
    """Changing ONLY the stored record_sha256 field (payload intact) is also
    caught: the canonical comparison covers every record field."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    bytes_before = path.read_bytes()

    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["record_sha256"] = "f" * 64  # fake digest, payload intact
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    bytes_after_tamper = path.read_bytes()

    with pytest.raises(RecorderError, match="journal content diverged"):
        store.add_candidate(acquired_envelope(registry, gene_id="other_v1"))
    # no APPEND happened: the handle refused before writing
    assert path.read_bytes() == bytes_after_tamper
    assert len(path.read_text().splitlines()) == 1
