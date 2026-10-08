"""Immutability hardening tests (issue #10 criterion 6) — stored state,
journal records, and history must be safe from caller mutation."""

import json

from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope


def build_store(tmp_path, registry):
    store = SomaticStore.create(tmp_path / "journal.jsonl", registry)
    store.add_candidate(acquired_envelope(registry))
    store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    return store


def test_mutating_original_candidate_after_insert_changes_nothing(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    envelope["candidate"]["provenance"]["source_trajectory"] = "T-tampered"
    envelope["candidate"]["artifact"] = "registry://skills/hacked@1/sha256:" + "0" * 64
    envelope["validation"]["state"] = "validated"
    assert store.state("look_before_heat_v1", 1) == "candidate"
    stored = store.get("look_before_heat_v1", 1)
    assert stored["candidate"]["provenance"]["source_trajectory"] == "T-aaaa1111aaaa"
    assert stored["validation"]["state"] == "candidate"
    store.close()


def test_get_returns_deep_copies(tmp_path, registry):
    store = build_store(tmp_path, registry)
    leaked = store.get("look_before_heat_v1", 1)
    leaked["candidate"]["gene_id"] = "tampered"
    leaked["validation"]["gate_reports"].append("CIG-FAKE")
    fresh = store.get("look_before_heat_v1", 1)
    assert fresh["candidate"]["gene_id"] == "look_before_heat_v1"
    assert fresh["validation"]["gate_reports"] == ["CIG-0007"]
    store.close()


def test_all_returns_deep_copies(tmp_path, registry):
    store = build_store(tmp_path, registry)
    leaked = store.all()
    leaked[("look_before_heat_v1", 1)]["candidate"]["origin"] = "assimilation"
    fresh = store.get("look_before_heat_v1", 1)
    assert fresh["candidate"]["origin"] == "acquired"  # validated ≠ assimilated
    store.close()


def test_by_state_returns_deep_copies(tmp_path, registry):
    store = build_store(tmp_path, registry)
    leaked = store.by_state("validated")
    leaked[("look_before_heat_v1", 1)]["validation"]["gate_reports"] = []
    fresh = store.get("look_before_heat_v1", 1)
    assert fresh["validation"]["gate_reports"] == ["CIG-0007"]
    store.close()


def test_history_returns_deep_copies(tmp_path, registry):
    store = build_store(tmp_path, registry)
    history = store.history("look_before_heat_v1", 1)
    history[0]["candidate"]["provenance"]["source_trajectory"] = "T-hacked"
    history[1]["validation"]["gate_reports"] = ["CIG-HACKED"]
    fresh_history = store.history("look_before_heat_v1", 1)
    assert fresh_history[0]["candidate"]["provenance"]["source_trajectory"] == \
        "T-aaaa1111aaaa"
    assert fresh_history[1]["validation"]["gate_reports"] == ["CIG-0007"]
    store.close()


def test_candidate_ref_returns_deep_copy(tmp_path, registry):
    store = build_store(tmp_path, registry)
    ref = store.candidate_ref("look_before_heat_v1", 1)
    ref["artifact"] = "registry://skills/hacked@1/sha256:" + "1" * 64
    fresh = store.candidate_ref("look_before_heat_v1", 1)
    assert fresh["artifact"].startswith("registry://skills/look_before_heat_v1@1/")
    store.close()
