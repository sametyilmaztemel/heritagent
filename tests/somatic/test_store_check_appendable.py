"""SomaticStore.check_appendable (PR #46 review): the same refusal an append
would give, with no side effect on the handle or the journal."""

import json

import pytest

from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope
from trajectory.recorder.errors import RecorderError


def test_healthy_store_passes_and_leaves_the_journal_untouched(tmp_path, registry, envelope):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    before = path.read_bytes()
    store.check_appendable()
    store.check_appendable()
    assert path.read_bytes() == before
    store.add_candidate(acquired_envelope(registry, gene_id="second_v1"))  # still appendable


def test_in_memory_store_passes_until_closed(registry, envelope):
    store = SomaticStore.create(None, registry)
    store.add_candidate(envelope)
    store.check_appendable()
    store.close()
    with pytest.raises(RecorderError, match="closed"):
        store.check_appendable()


def test_same_count_payload_tamper_is_refused(tmp_path, registry, envelope):
    """The guard's documented case: one old record's payload edited, record
    count and tail unchanged."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    store.add_candidate(acquired_envelope(registry, gene_id="second_v1"))
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["envelope"]["candidate"]["provenance"]["source_trajectory"] = "T-tampered"
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RecorderError, match="first divergence at record 1"):
        store.check_appendable()


def test_a_refused_check_does_not_heal_or_close_the_handle(tmp_path, registry, envelope):
    """After a refusal the handle is still stale (not reloaded) and still
    open (the refusal names staleness, not closure); the journal is
    unchanged."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    SomaticStore.open(path, registry).add_candidate(acquired_envelope(registry, gene_id="other_v1"))
    after_other = path.read_bytes()
    for _ in range(2):
        with pytest.raises(RecorderError, match="stale somatic store handle"):
            store.check_appendable()
    assert path.read_bytes() == after_other
    assert store.get("other_v1", 1) is None  # the handle did not reload
