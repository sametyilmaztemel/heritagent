"""Persistent-store registry requirement tests (issue #10 final re-review).

Persistent evidence requires registry-backed validation: artifact existence,
digest match, and identity binding. In-memory stores keep the legacy
registry-less compatibility."""

import pytest

from genome.validation.registry import TraitRegistry
from somatic.store import SomaticStore
from trajectory.recorder.errors import RecorderError
from tests.somatic.conftest import acquired_envelope


def test_create_without_registry_rejected_before_file_creation(tmp_path):
    path = tmp_path / "journal.jsonl"
    with pytest.raises(RecorderError, match="requires a TraitRegistry"):
        SomaticStore.create(path)  # no registry
    assert not path.exists()  # journal file was NEVER created


def test_open_without_registry_rejected(tmp_path, registry):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry))
    store.close()
    with pytest.raises(RecorderError, match="requires a TraitRegistry"):
        SomaticStore.open(path)  # no registry: replay/write must not start
    store.close()


def test_missing_artifact_rejected_by_persistent_store(tmp_path, registry, envelope):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    broken = dict(envelope)
    # correct name/version/grammar, but the digest does not exist in the registry
    broken["candidate"] = dict(envelope["candidate"],
                                artifact="registry://skills/look_before_heat_v1@1/sha256:" + "ab" * 32)
    with pytest.raises(Exception, match="not found in registry"):
        store.add_candidate(broken)
    # nothing written: journal contains no candidate_added record
    assert path.read_text() == ""
    store.close()


def test_unbound_artifact_rejected_by_persistent_store(tmp_path, registry, envelope):
    """An artifact URI whose identity was never bound via registry.put()
    (digest belonging to another identity) must not enter the store."""
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    bound = acquired_envelope(registry, gene_id="real_trait_v1")
    store.add_candidate(bound)
    bound_digest = bound["candidate"]["artifact"].split("sha256:")[1]
    forged = dict(envelope)
    forged["candidate"] = dict(envelope["candidate"],
                                gene_id="forged_v1",
                                artifact=f"registry://skills/forged_v1@1/sha256:{bound_digest}")
    with pytest.raises(Exception, match="not bound"):
        store.add_candidate(forged)
    store.close()
    reopened = SomaticStore.open(path, registry)
    assert reopened.state("forged_v1", 1) is None  # never entered the store
    reopened.close()


def test_in_memory_registry_less_still_works(envelope, registry):
    """Legacy in-memory compatibility through the compat import path:
    registry-less construction works (with a placeholder artifact URI,
    since no registry checks are possible), and a registry may still be
    supplied for binding checks."""
    from genome.validation.records import SomaticStore as CompatSomaticStore
    store = CompatSomaticStore()  # no path, no registry
    envelope_mem = dict(envelope)
    envelope_mem["candidate"] = dict(envelope["candidate"],
                                      artifact="registry://skills/look_before_heat_v1@1/sha256:" + "cd" * 32)
    store.add(envelope_mem)
    assert store.state("look_before_heat_v1") == "candidate"

    registered = CompatSomaticStore(registry=registry)  # in-memory with registry
    registered.add(acquired_envelope(registry))
    assert registered.state("look_before_heat_v1") == "candidate"
