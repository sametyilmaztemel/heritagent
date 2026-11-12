"""Somatic journal strictness (module audit and review of #54, 2026-10-13).

Every commit writes its record and its newline together, so a journal
whose last record has no newline holds a torn write or an external edit.
The store used to reopen such a journal, and the next append glued a
second record onto that line, leaving a journal that no longer parsed.
It now fails closed when it reads the journal: at open, and in the guard
every append runs. Lines that are not JSON objects, and overflowing
numbers, are refused as well."""

import json

import pytest

from genome.validation.errors import RecordConsistencyError
from somatic.journal import build_record
from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope
from trajectory.recorder.errors import MalformedTrajectoryError
from trajectory.storage.canonical import canonical_json


def journal_with_one_candidate(tmp_path, registry):
    path = tmp_path / "journal.jsonl"
    store = SomaticStore.create(path, registry)
    store.add_candidate(acquired_envelope(registry))
    store.close()
    assert path.read_bytes().endswith(b"\n")
    return path


def test_an_unterminated_last_record_fails_closed_at_open(tmp_path, registry):
    path = journal_with_one_candidate(tmp_path, registry)
    path.write_bytes(path.read_bytes()[:-1])  # a write torn just before the newline
    with pytest.raises(MalformedTrajectoryError, match="does not end with a newline"):
        SomaticStore.open(path, registry)


def test_an_unterminated_last_record_refuses_the_next_append(tmp_path, registry):
    path = journal_with_one_candidate(tmp_path, registry)
    store = SomaticStore.open(path, registry)
    before = path.read_bytes()[:-1]
    path.write_bytes(before)  # edited while the handle is open
    with pytest.raises(MalformedTrajectoryError, match="does not end with a newline"):
        store.add_candidate(acquired_envelope(registry, gene_id="second_v1"))
    assert path.read_bytes() == before  # nothing glued on


@pytest.mark.parametrize("line, message", [
    (b"[1, 2]\n", "journal line 2 is not a JSON object"),
    (b"42\n", "journal line 2 is not a JSON object"),
    (b'{"record": "somatic-journal", "seq": 1e999}\n', "non-finite JSON number '1e999'"),
    (b'{"record": "somatic-journal", "seq": -1e999}\n', "non-finite JSON number '-1e999'"),
])
def test_a_hostile_line_fails_closed_with_a_typed_error(tmp_path, registry, line, message):
    path = journal_with_one_candidate(tmp_path, registry)
    path.write_bytes(path.read_bytes() + line)
    with pytest.raises(MalformedTrajectoryError, match=message):
        SomaticStore.open(path, registry)


def test_a_line_with_a_duplicate_key_fails_closed(tmp_path, registry):
    """A parser that keeps the first value would read the forged op; the
    digest binds only the last one, so such a line is refused."""
    path = journal_with_one_candidate(tmp_path, registry)
    path.write_bytes(b'{"op":"decision_recorded",' + path.read_bytes()[1:])
    with pytest.raises(MalformedTrajectoryError, match="line 1: duplicate key 'op'"):
        SomaticStore.open(path, registry)


def chained_record(path, op, gene_id, envelope):
    """A record for ``envelope`` whose seq and hash chain follow the journal."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return build_record(seq=len(records) + 1, op=op, gene_id=gene_id, version=1,
                        envelope=envelope, prev_sha256=records[-1]["record_sha256"])


def hostile_records(path, registry):
    """Well-formed JSON objects whose inner shape the journal record schema
    leaves open; replay must report each one as an issue of record 2."""
    no_state = acquired_envelope(registry, gene_id="second_v1")
    no_state["validation"] = {}
    decided = acquired_envelope(registry)
    decided["validation"] = {"state": "rejected", "gate_reports": [{}]}
    return {
        "empty object": {},
        "no seq": {"record": "somatic-journal"},
        "empty candidate": chained_record(path, "candidate_added", "x_v1",
                                          {"candidate": {}, "validation": {}}),
        "no validation state": chained_record(path, "candidate_added", "second_v1",
                                              no_state),
        "unhashable gate report": chained_record(path, "decision_recorded",
                                                 "look_before_heat_v1", decided),
    }


@pytest.mark.parametrize("name", ["empty object", "no seq", "empty candidate",
                                  "no validation state", "unhashable gate report"])
def test_a_well_formed_hostile_record_is_reported_with_its_record_number(
        tmp_path, registry, name):
    path = journal_with_one_candidate(tmp_path, registry)
    store = SomaticStore.open(path, registry)
    record = hostile_records(path, registry)[name]
    path.write_bytes(path.read_bytes() + canonical_json(record).encode("utf-8") + b"\n")
    issues = store.verify()
    store.close()
    assert any(issue.startswith("record 2: ") for issue in issues), issues
    with pytest.raises(RecordConsistencyError, match="record 2: "):
        SomaticStore.open(path, registry)
