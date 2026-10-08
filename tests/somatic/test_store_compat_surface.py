"""Compat create/open surface tests (issue #10 final re-review, group 3):
persistent create/open must return the COMPAT class instance so
version-less convenience APIs and the add() alias survive."""

import pytest

from genome.validation.errors import RecordConsistencyError

import genome.validation.records as records_module
from genome.validation.records import SomaticStore
from tests.somatic.conftest import acquired_envelope


def test_persistent_create_returns_compat_instance(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    assert isinstance(store, records_module.SomaticStore)
    assert isinstance(store, SomaticStore)  # and the base implementation
    # compat add() alias works on the persistent path
    store.add(envelope)
    assert store.state("look_before_heat_v1") == "candidate"
    store.close()


def test_persistent_open_returns_compat_instance(tmp_path, registry, envelope):
    path = tmp_path / "j.jsonl"
    creator = SomaticStore.create(path, registry)
    creator.add_candidate(envelope)
    creator.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    creator.close()

    store = SomaticStore.open(path, registry)
    assert isinstance(store, records_module.SomaticStore)
    # single version -> version-less state/get convenience works
    assert store.state("look_before_heat_v1") == "validated"
    assert store.get("look_before_heat_v1")["candidate"]["version"] == 1
    store.close()


def test_two_versions_independently_addressable_on_persistent_path(tmp_path, registry):
    path = tmp_path / "versions.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry, version=1))
    store.add_candidate(acquired_envelope(registry, version=2))
    store.close()

    reopened = SomaticStore.open(path, registry)
    # logical identity (gene_id, version) never collapses
    assert set(reopened.all()) == {("look_before_heat_v1", 1), ("look_before_heat_v1", 2)}
    # version-less shorthand with >1 versions -> typed ambiguity error
    with pytest.raises(RecordConsistencyError, match="ambiguous gene_id"):
        reopened.get("look_before_heat_v1")
    with pytest.raises(RecordConsistencyError, match="ambiguous gene_id"):
        reopened.state("look_before_heat_v1")
    # explicit versions work independently
    assert reopened.get("look_before_heat_v1", 1)["candidate"]["version"] == 1
    assert reopened.get("look_before_heat_v1", 2)["candidate"]["version"] == 2
    reopened.close()
