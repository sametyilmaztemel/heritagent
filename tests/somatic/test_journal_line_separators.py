"""Somatic journal record splitting (module audit, 2026-10-13): a
schema-valid envelope whose text holds U+2028, U+2029 or U+0085 is written
raw by canonical JSON; the journal must still read back, and the store must
reopen and verify."""

import copy

import pytest

from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope
from trajectory.recorder.errors import MalformedTrajectoryError

NOTES = "first\u2028second\u2029third\u0085fourth"


def test_journal_with_unicode_line_separators_reopens(tmp_path, registry):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    envelope = copy.deepcopy(acquired_envelope(registry))
    envelope["candidate"]["provenance"]["notes"] = NOTES
    store.add_candidate(envelope)
    store.add_candidate(acquired_envelope(registry, gene_id="second_v1"))
    assert "\u2028".encode() in path.read_bytes()
    store.close()
    reopened = SomaticStore.open(path, registry)
    reopened.verify_strict()
    assert reopened.get(envelope["candidate"]["gene_id"], 1)["candidate"]["provenance"][
        "notes"] == NOTES
    reopened.check_appendable()


def test_invalid_utf8_in_a_journal_fails_closed(tmp_path, registry, envelope):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(envelope)
    store.close()
    path.write_bytes(path.read_bytes()[:-3] + b"\xff\n")
    with pytest.raises(MalformedTrajectoryError, match="invalid UTF-8 in journal at line 1"):
        SomaticStore.open(path, registry)
