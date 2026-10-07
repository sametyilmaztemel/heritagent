"""Journal source-of-truth + stale-writer protection tests (issue #10
re-review, group 2)."""

import json

import pytest

from genome.validation.errors import RecordConsistencyError
from trajectory.recorder.errors import RecorderError
from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope


def test_verify_rereads_journal_from_disk(tmp_path, registry, envelope):
    """verify() must read the journal FILE, not cached records: external
    truncation (0 records on disk vs 1 expected) is detected."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    store.close()
    path.write_text("")  # external truncation to empty
    # the CLOSED original handle still expects 1 record: verify() re-reads
    # the disk journal and reports the truncation
    issues = store.verify()
    assert any("truncation" in i or "expected state is 1" in i for i in issues)


def test_verify_detects_external_tamper(tmp_path, registry, envelope):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    store.close()
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["envelope"]["candidate"]["provenance"]["born_generation"] = 77
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    # a fresh open fails closed on the tampered journal...
    with pytest.raises(RecordConsistencyError, match="record_sha256 mismatch"):
        SomaticStore.open(path, registry)
    # ...and a handle opened BEFORE the tamper detects it via verify()
    live = SomaticStore.open.__wrapped__ if False else None  # noqa: F841
    # (open a fresh store handle first, then tamper)
    path2 = tmp_path / "j2.jsonl"
    store2 = SomaticStore.create(path2, registry)
    store2.add_candidate(envelope)
    lines2 = path2.read_text().splitlines()
    record2 = json.loads(lines2[0])
    record2["envelope"]["candidate"]["provenance"]["born_generation"] = 77
    lines2[0] = json.dumps(record2, sort_keys=True, separators=(",", ":"))
    path2.write_text("\n".join(lines2) + "\n")
    issues = store2.verify()  # re-reads the journal from disk
    assert any("record_sha256 mismatch" in i for i in issues)


def test_stale_second_handle_append_rejected_journal_intact(tmp_path, registry):
    """Two handles on the same journal: after the first appends, the second
    handle's append is rejected (stale count/tail) and the journal stays
    valid — exactly the original evidence, no corruption."""
    path = tmp_path / "journal.jsonl"
    first = SomaticStore.create(path, registry)
    second = SomaticStore.open(path, registry)  # both handles on one journal

    first.add_candidate(acquired_envelope(registry, gene_id="trait_a_v1"))
    bytes_after_first = path.read_bytes()

    with pytest.raises(RecorderError) as exc_info:
        second.add_candidate(acquired_envelope(registry, gene_id="trait_b_v1"))
    message = str(exc_info.value)
    assert "stale somatic store handle" in message
    assert "another writer appended" in message

    # journal intact: byte-identical to the first handle's state, and it
    # still verifies cleanly
    assert path.read_bytes() == bytes_after_first
    verifier = SomaticStore.open(path, registry)
    assert verifier.verify() == []
    assert verifier.state("trait_a_v1", 1) == "candidate"
    verifier.close()
    first.close()
    second.close()


def test_external_append_between_writes_rejected(tmp_path, registry, envelope):
    """A valid record appended EXTERNALLY (correct chain) still makes the
    open handle stale — append fails closed; re-opening sees both records."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)

    # externally append a well-formed candidate for another trait
    import json as _json
    from somatic.journal import build_record
    external = acquired_envelope(registry, gene_id="external_v1")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    record = build_record(seq=len(lines) + 1, op="candidate_added",
                           gene_id="external_v1", version=1, envelope=external,
                           prev_sha256=lines[-1]["record_sha256"])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(_json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(RecorderError) as exc_info:
        store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    assert "stale somatic store handle" in str(exc_info.value)

    # a fresh handle sees both traits and verifies cleanly
    fresh = SomaticStore.open(path, registry)
    assert fresh.state("look_before_heat_v1", 1) == "candidate"
    assert fresh.state("external_v1", 1) == "candidate"
    assert fresh.verify() == []
    fresh.close()
    store.close()


def test_external_truncation_rejected_on_append(tmp_path, registry, envelope):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    path.write_text("")  # external truncation
    with pytest.raises(RecorderError) as exc_info:
        store.add_candidate(acquired_envelope(registry, gene_id="other_v1"))
    assert "stale somatic store handle" in str(exc_info.value)
    store.close()
