"""A restore that takes an older snapshot than the newest manifest keeps a
lasting record of the manifests it passed over (ADR-0008, the accounting
gap): their number and why go into the restore record and the ledger's
restore note, and each is printed. The ledger, the figures' accounting
rows and repair-ledger's pending note accept the key; an older note
without it still loads. Real ``RunMirror``, ledger and restore; the ``run``
command's case is in test_exp0001_mirror_cli."""

import json
from pathlib import Path

from evaluation.exp0001 import mirror as mirror_module
from evaluation.exp0001.figures import accounting_rows
from evaluation.exp0001.ledger import RunLedger, append_note, repair_torn_tail
from evaluation.exp0001.mirror import (
    RESTORE_RECORD,
    RunMirror,
    append_pending_restore_note,
    restore,
)
from tests.evaluation.test_exp0001_lifetime import Clock


def three_snapshots(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    """A run directory snapshotted three times, a new episode and a new
    file in each; returns it, the mirror and the three manifests."""
    local, drive = tmp_path / "vm1", tmp_path / "drive"
    ledger = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    mirror = RunMirror(local, drive)
    for index in range(3):
        ledger.admit(f"e{index}")
        ledger.record(f"e{index}", model_calls=1, outcome={})
        (local / f"f{index}.json").write_text(f"file {index}")
        assert mirror.snapshot() == index + 1
    return local, drive, [json.loads((drive / f"snapshot-{seq:08d}.json").read_text())
                          for seq in (1, 2, 3)]


def blob(drive: Path, doc: dict, rel: str) -> Path:
    return mirror_module._blob(drive, doc["files"][rel]["sha256"])


def last_note(local: Path) -> dict:
    return json.loads((local / "ledger.jsonl").read_text().splitlines()[-1])


def test_a_restore_past_missing_blobs_records_and_prints_the_manifest(tmp_path, capsys):
    _, drive, docs = three_snapshots(tmp_path)
    blob(drive, docs[2], "f2.json").unlink()  # never uploaded
    vm2 = tmp_path / "vm2"
    assert restore(drive, vm2)["seq"] == 2
    record = json.loads((vm2 / RESTORE_RECORD).read_text())
    assert [entry["seq"] for entry in record["passed_over"]] == [3]
    assert "the blobs of 1 of its" in record["passed_over"][0]["why"]
    assert "f2.json" in record["passed_over"][0]["why"]
    note = last_note(vm2)
    assert note["type"] == "restore" and note["snapshot_seq"] == 2
    assert note["passed_over"] == record["passed_over"]
    err = capsys.readouterr().err
    assert "[restore] snapshot 3 is passed over: the blobs of 1 of its" in err


def test_a_restore_past_a_hash_failure_records_and_prints_the_manifest(tmp_path, capsys):
    _, drive, docs = three_snapshots(tmp_path)
    blob(drive, docs[2], "f2.json").write_bytes(b"bit rot")
    vm2 = tmp_path / "vm2"
    assert restore(drive, vm2)["seq"] == 2
    record = json.loads((vm2 / RESTORE_RECORD).read_text())
    assert [entry["seq"] for entry in record["passed_over"]] == [3]
    assert "fail their hash" in record["passed_over"][0]["why"]
    assert last_note(vm2)["passed_over"] == record["passed_over"]
    assert "[restore] snapshot 3 is not whole (f2.json:" in capsys.readouterr().err


def test_every_manifest_passed_over_is_on_record_newest_first(tmp_path):
    _, drive, docs = three_snapshots(tmp_path)
    blob(drive, docs[2], "f2.json").unlink()
    blob(drive, docs[1], "f1.json").write_bytes(b"bit rot")
    vm2 = tmp_path / "vm2"
    assert restore(drive, vm2)["seq"] == 1
    passed = last_note(vm2)["passed_over"]
    assert [entry["seq"] for entry in passed] == [3, 2]
    assert "missing" in passed[0]["why"] and "fail their hash" in passed[1]["why"]
    ledger = RunLedger(vm2 / "ledger.jsonl", run_id="r", clock=Clock(), readonly=True)
    assert ledger.notes[-1]["passed_over"] == passed  # the loader keeps the key


def test_a_restore_of_the_newest_snapshot_adds_no_key_to_its_note(tmp_path):
    _, drive, _ = three_snapshots(tmp_path)
    vm2 = tmp_path / "vm2"
    assert restore(drive, vm2)["seq"] == 3
    assert "passed_over" not in last_note(vm2)
    assert json.loads((vm2 / RESTORE_RECORD).read_text())["passed_over"] == []


def test_the_pending_note_of_a_torn_ledger_keeps_what_was_passed_over(tmp_path):
    """The restored ledger ends with a torn line: the note waits in the
    restore record with its passed-over manifests, and repair-ledger's path
    appends it whole."""
    local, drive = tmp_path / "vm1", tmp_path / "drive"
    ledger = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    mirror = RunMirror(local, drive)
    ledger.admit("e0")
    path = local / "ledger.jsonl"
    path.write_bytes(path.read_bytes() + b'{"type":"epi')  # a write a kill interrupted
    assert mirror.snapshot() == 1
    (local / "later.json").write_text("later")
    assert mirror.snapshot() == 2
    doc = json.loads((drive / "snapshot-00000002.json").read_text())
    blob(drive, doc, "later.json").unlink()
    vm2 = tmp_path / "vm2"
    assert restore(drive, vm2)["seq"] == 1
    pending = json.loads((vm2 / RESTORE_RECORD).read_text())["pending_note"]
    assert [entry["seq"] for entry in pending["passed_over"]] == [2]
    assert repair_torn_tail(vm2 / "ledger.jsonl") is not None
    assert append_pending_restore_note(vm2) == pending
    assert last_note(vm2) == pending
    reloaded = RunLedger(vm2 / "ledger.jsonl", run_id="r", clock=Clock(), readonly=True)
    assert [n["type"] for n in reloaded.notes] == ["repair", "restore"]


def test_the_figures_and_an_older_note_accept_the_key(tmp_path):
    local = tmp_path / "run"
    ledger = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    ledger.admit("e0")
    ledger.record("e0", model_calls=1, outcome={})
    old = {"type": "restore", "snapshot_seq": 4, "snapshot_sha256": "a" * 64}
    new = {**old, "snapshot_seq": 7, "passed_over": [{"seq": 9, "why": "blobs missing"}]}
    append_note(local / "ledger.jsonl", old)
    append_note(local / "ledger.jsonl", new)
    notes = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock(), readonly=True).notes
    assert list(notes) == [old, new]
    caps = {"episodes_max": 10, "model_calls_max": 10, "wall_clock_hours_max": 1}
    rows = accounting_rows({
        "run_start": {"run_id": "r", "protocol": {"caps": caps}, "model": {"model_id": "m"},
                      "code": {"commit": "c"}},
        "state": "running", "report": None, "ledger": {"note_records": list(notes)}})
    values = {row["item"]: row["value"] for row in rows}
    assert values["restores"] == 2
    assert values["restore 1"] == "snapshot_seq=4" and values["restore 2"] == "snapshot_seq=7"
