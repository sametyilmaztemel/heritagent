"""Somatic journal strictness (module audit and review of #54, 2026-10-13).

Every commit writes its record and its newline together, so a journal
whose last record has no newline holds a torn write or an external edit.
The store used to reopen such a journal, and the next append glued a
second record onto that line, leaving a journal that no longer parsed.
It now fails closed when it reads the journal: at open, and in the guard
every append runs. Lines that are not JSON objects, and overflowing
numbers, are refused as well."""

import pytest

from somatic.store import SomaticStore
from tests.somatic.conftest import acquired_envelope
from trajectory.recorder.errors import MalformedTrajectoryError


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
