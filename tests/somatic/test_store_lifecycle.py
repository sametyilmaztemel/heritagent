"""Somatic lifecycle contract tests (issue #10 criterion 1, 9)."""

import pytest

from genome.validation.errors import GenomeValidationError, RecordConsistencyError
from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope


def test_valid_acquired_candidate_insert(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "journal.jsonl", registry)
    stored = store.add_candidate(envelope)
    assert stored["validation"]["state"] == "candidate"
    assert store.state("look_before_heat_v1", 1) == "candidate"
    store.close()


def test_add_accepts_only_candidate_state(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    terminal = dict(envelope)
    terminal["validation"] = {"state": "validated", "gate_reports": ["CIG-0001"]}
    with pytest.raises(RecordConsistencyError,
                        match="candidate_added requires state 'candidate'"):
        store.add_candidate(terminal)
    store.close()


def test_duplicate_candidate_insert_rejected(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    with pytest.raises(RecordConsistencyError, match="no silent overwrite"):
        store.add_candidate(envelope)
    store.close()


def test_candidate_to_validated(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    terminal = store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    assert terminal["validation"] == {"state": "validated",
                                       "gate_reports": ["CIG-0007"]}
    assert store.state("look_before_heat_v1", 1) == "validated"
    store.close()


def test_candidate_to_rejected(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    store.decide("look_before_heat_v1", 1, "rejected", ["CIG-0008"])
    assert store.state("look_before_heat_v1", 1) == "rejected"
    store.close()


@pytest.mark.parametrize("verdict", ["validated", "rejected"])
def test_terminal_decisions_are_immutable(tmp_path, registry, envelope, verdict):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    store.decide("look_before_heat_v1", 1, verdict, ["CIG-0007"])
    with pytest.raises(RecordConsistencyError,
                        match="decision requires previous state 'candidate'"):
        store.decide("look_before_heat_v1", 1,
                      "rejected" if verdict == "validated" else "validated",
                      ["CIG-0008"])
    store.close()


def test_decision_cannot_swap_artifact_or_provenance(tmp_path, registry, envelope):
    """A lifecycle transition may change ONLY the validation envelope."""
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    # simulate an envelope-swap attempt inside the decision path
    original = store.get("look_before_heat_v1", 1)
    other = acquired_envelope(registry, gene_id="other_skill_v1")
    tampered = dict(envelope)
    tampered["candidate"] = dict(envelope["candidate"],
                                  artifact=other["candidate"]["artifact"])
    # decide() works on the STORED copy, so an external swap attempt cannot
    # reach it; assert the invariant directly on the stored envelope
    terminal = store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    assert terminal["candidate"]["artifact"] == original["candidate"]["artifact"]
    assert terminal["candidate"]["provenance"] == original["candidate"]["provenance"]
    assert terminal["validation"] != original["validation"]  # only envelope changed
    store.close()


def test_validated_still_acquired_not_assimilation(tmp_path, registry, envelope):
    """validated = 'eligible for assimilation', NOT inherited: origin stays
    acquired and no germline ref is created by the store."""
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    terminal = store.decide("look_before_heat_v1", 1, "validated", ["CIG-0007"])
    assert terminal["candidate"]["origin"] == "acquired"
    assert "cig_record" not in terminal["candidate"]["provenance"]
    store.close()


def test_decision_requires_non_empty_gate_reports(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    with pytest.raises(RecordConsistencyError, match="non-empty gate_reports"):
        store.decide("look_before_heat_v1", 1, "validated", [])
    store.close()


def test_gate_reports_canonicalized_and_duplicates_rejected(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    with pytest.raises(RecordConsistencyError, match="duplicates"):
        store.decide("look_before_heat_v1", 1, "validated",
                      ["CIG-0007", "CIG-0007"])
    terminal = store.decide("look_before_heat_v1", 1, "validated",
                             ["CIG-0009", "CIG-0002"])
    assert terminal["validation"]["gate_reports"] == ["CIG-0002", "CIG-0009"]  # sorted unique
    store.close()


def test_unknown_verdict_rejected(tmp_path, registry, envelope):
    store = SomaticStore.create(tmp_path / "j.jsonl", registry)
    store.add_candidate(envelope)
    with pytest.raises(RecordConsistencyError, match="verdict"):
        store.decide("look_before_heat_v1", 1, "assimilated", ["CIG-0007"])
    store.close()
