"""Close lifecycle + compat create/open/identity tests (issue #10 re-review,
groups 3-4) — exercised through the genome.validation.records import path."""

import pytest

from genome.validation.errors import RecordConsistencyError
from trajectory.recorder.errors import RecorderError
from genome.validation.records import SomaticStore
from tests.somatic.conftest import acquired_envelope


def test_compat_create_and_open_classmethods(tmp_path, registry, envelope):
    """create()/open() work through the genome.validation.records import
    path (critic group 3)."""
    path = tmp_path / "compat.jsonl"
    creator = SomaticStore.create(path, registry)
    creator.add_candidate(envelope)
    creator.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    creator.close()

    opener = SomaticStore.open(path, registry)
    assert opener.state("look_before_heat_v1", 1) == "validated"
    assert opener.verify() == []
    opener.close()


def test_logical_identity_two_versions_independent(tmp_path, registry):
    """Two versions of the same gene_id stay independently addressable:
    (gene_id, version) identity never collapses."""
    store = SomaticStore()
    store.add_candidate(acquired_envelope(registry, version=1))
    store.add_candidate(acquired_envelope(registry, version=2))

    assert store.get("look_before_heat_v1", 1)["candidate"]["version"] == 1
    assert store.get("look_before_heat_v1", 2)["candidate"]["version"] == 2
    assert store.state("look_before_heat_v1", 1) == "candidate"
    assert store.state("look_before_heat_v1", 2) == "candidate"
    # all() keeps (gene_id, version) keys — no collapse under bare gene_id
    assert set(store.all()) == {("look_before_heat_v1", 1), ("look_before_heat_v1", 2)}


def test_versionless_shorthand_ambiguity_contract(tmp_path, registry):
    """Version-less get()/state() shorthand: 0 versions -> None; exactly 1
    version -> convenience works; >1 versions -> typed ambiguity error."""
    store = SomaticStore()
    assert store.get("missing_v1") is None
    assert store.state("missing_v1") is None

    store.add_candidate(acquired_envelope(registry, version=1))
    assert store.get("look_before_heat_v1")["candidate"]["version"] == 1
    assert store.state("look_before_heat_v1") == "candidate"

    store.add_candidate(acquired_envelope(registry, version=2))
    with pytest.raises(RecordConsistencyError, match="ambiguous gene_id"):
        store.get("look_before_heat_v1")
    with pytest.raises(RecordConsistencyError, match="ambiguous gene_id"):
        store.state("look_before_heat_v1")
    # explicit versions still work
    assert store.get("look_before_heat_v1", 1) is not None
    assert store.get("look_before_heat_v1", 2) is not None


def test_close_is_idempotent_and_blocks_mutations_persistent(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    store.close()
    store.close()  # idempotent
    with pytest.raises(RecorderError, match="store is closed"):
        store.add_candidate(envelope)


def test_close_blocks_mutations_in_memory_too(registry, envelope):
    store = SomaticStore()  # in-memory variant
    store.add(envelope)
    store.close()
    store.close()  # idempotent
    with pytest.raises(RecorderError, match="store is closed"):
        store.add(envelope)
    with pytest.raises(RecorderError, match="store is closed"):
        store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])


def test_read_apis_usable_after_close(tmp_path, registry, envelope):
    """Read APIs remain usable after close (documented choice)."""
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    store.close()
    assert store.state("look_before_heat_v1", 1) == "validated"
    assert store.get("look_before_heat_v1", 1)["validation"]["gate_reports"] == ["CIG-0007"]
    assert list(store.by_state("validated"))
    assert len(store.history("look_before_heat_v1", 1)) == 2
    assert store.candidate_ref("look_before_heat_v1", 1)["gene_id"] == "look_before_heat_v1"
    assert store.verify() == []  # re-reads the journal from disk even when closed


def test_failed_decision_keeps_store_recoverable(tmp_path, registry, envelope):
    """A rejected DECISION argument (empty gate reports) is recoverable: the
    store stays open and the caller may retry with valid arguments. Only
    close() releases the handle."""
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    with pytest.raises(RecordConsistencyError):
        store.decide("look_before_heat_v1", 1, "validated", [])  # empty gate reports
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])  # retry works
    assert store.state("look_before_heat_v1", 1) == "validated"
    store.close()
