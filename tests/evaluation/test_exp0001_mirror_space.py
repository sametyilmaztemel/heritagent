"""The run mirror's room on Drive (ADR-0008 D5): blobs are stored
gzip-compressed under the hash of their uncompressed content, a mirror of
the older uncompressed layout still reads and restores, a manifest of a
layout this code does not know fails closed, the collection keeps exactly
what the kept snapshots reference, and ``mirror-check`` reports the free
space of the mirror's filesystem. No GPU, no vLLM, no ALFWorld."""

import gzip
import hashlib
import json
import shutil
from collections import namedtuple
from pathlib import Path

import pytest

import evaluation.exp0001.mirror as mirror_module
from evaluation.exp0001 import cli
from evaluation.exp0001.mirror import (
    MirrorError,
    RunMirror,
    latest_snapshot,
    mirror_state,
    read_snapshot_file,
    restore,
    snapshot_file,
)

PROMPT = "You are in the middle of a room. Looking quickly around you, you see " * 40


def local_files(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in mirror_module.SKIPPED_NAMES}


def blob_files(drive: Path) -> dict:
    return {p.name: p.read_bytes() for p in (drive / "blobs").glob("*/*")}


def grow(local: Path, index: int) -> None:
    """One more episode: a trajectory of repeated prompt text and a ledger line."""
    (local / "trajectories").mkdir(parents=True, exist_ok=True)
    (local / "trajectories" / f"T-{index}.jsonl").write_text(
        json.dumps({"messages": [PROMPT] * (index + 1)}) + "\n")
    with (local / "ledger.jsonl").open("a") as handle:
        handle.write(json.dumps({"type": "episode", "id": f"e{index}", "prompt": PROMPT}) + "\n")


def a_run(tmp_path: Path, episodes: int = 3) -> tuple[Path, Path]:
    local, drive = tmp_path / "local", tmp_path / "drive"
    local.mkdir()
    (local / "run-start.json").write_text(json.dumps({"code": {"commit": "c" * 40}}))
    for index in range(episodes):
        grow(local, index)
    return local, drive


def test_a_blob_is_stored_compressed_under_the_hash_of_its_uncompressed_content(tmp_path):
    local, drive = a_run(tmp_path)
    RunMirror(local, drive).snapshot()
    doc = latest_snapshot(drive)
    assert doc["schema"] == mirror_module.MIRROR_SCHEMA
    stored = blob_files(drive)
    for rel, data in local_files(local).items():
        entry = doc["files"][rel]
        assert entry == {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        # deterministic bytes: a fixed level and no time in the header
        assert stored[entry["sha256"]] == gzip.compress(data, compresslevel=6, mtime=0)
        assert read_snapshot_file(drive, doc, rel) == data


def test_snapshots_of_a_growing_run_leave_only_compressed_blobs_and_restore_byte_for_byte(
        tmp_path, monkeypatch):
    monkeypatch.setattr(mirror_module, "GC_EVERY", 1)
    local, drive = a_run(tmp_path, episodes=1)
    mirror = RunMirror(local, drive)
    raw = 0
    for index in range(1, 30):
        grow(local, index)
        assert mirror.snapshot() is not None
        raw += sum(len(data) for data in local_files(local).values())
    kept = [json.loads(p.read_bytes()) for p in drive.glob("snapshot-*.json")]
    contents = {}
    for doc in kept:
        for rel, entry in doc["files"].items():
            contents[entry["sha256"]] = read_snapshot_file(drive, doc, rel)
    stored = blob_files(drive)
    assert set(stored) == set(contents)
    assert sum(map(len, stored.values())) <= sum(
        len(gzip.compress(data, compresslevel=6, mtime=0)) for data in contents.values())
    assert sum(map(len, stored.values())) * 10 < sum(map(len, contents.values()))
    restored = tmp_path / "vm2"
    restore(drive, restored)
    expected = local_files(local)
    got = local_files(restored)
    assert got.pop("ledger.jsonl").startswith(expected.pop("ledger.jsonl"))
    assert got == expected


def test_the_collection_keeps_exactly_the_blobs_the_kept_snapshots_reference(
        tmp_path, monkeypatch):
    monkeypatch.setattr(mirror_module, "GC_EVERY", 1)
    local, drive = a_run(tmp_path, episodes=1)
    mirror = RunMirror(local, drive)
    for index in range(1, 12):
        grow(local, index)
        mirror.snapshot()
    manifests = sorted(drive.glob("snapshot-*.json"))
    assert len(manifests) == mirror_module.KEEP_SNAPSHOTS >= 5
    referenced = {entry["sha256"] for path in manifests
                  for entry in json.loads(path.read_bytes())["files"].values()}
    assert set(blob_files(drive)) == referenced
    # every kept snapshot is whole and restores (a margin against Drive's upload lag)
    newest = int(manifests[-1].name[9:17])
    for lost, path in enumerate(reversed(manifests[1:]), 1):  # the newest never arrived
        path.unlink()
        assert latest_snapshot(drive)["seq"] == newest - lost
    assert restore(drive, tmp_path / "vm2")["seq"] == newest - mirror_module.KEEP_SNAPSHOTS + 1


def test_the_wholeness_check_lists_blobs_and_reads_none(tmp_path, monkeypatch):
    local, drive = a_run(tmp_path)
    RunMirror(local, drive).snapshot()
    real, read = Path.read_bytes, []

    def counting(self):
        if drive / "blobs" in self.parents:
            read.append(self.name)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    assert latest_snapshot(drive) is not None and mirror_state(drive) == "whole"
    assert read == []


# -- the uncompressed layout 0.1 (the finished pilot's mirror) still reads ------------------
def a_mirror_of_layout_0_1(tmp_path: Path) -> tuple[Path, dict]:
    """A mirror as the code before compressed blobs wrote it: raw blobs and
    a manifest of schema 0.1."""
    local, drive = a_run(tmp_path)
    (drive / "blobs").mkdir(parents=True)
    entries = {}
    for rel, data in local_files(local).items():
        digest = hashlib.sha256(data).hexdigest()
        (drive / "blobs" / digest[:2]).mkdir(exist_ok=True)
        (drive / "blobs" / digest[:2] / digest).write_bytes(data)
        entries[rel] = {"sha256": digest, "size": len(data)}
    doc = {"schema": "heritagent/exp0001-mirror/0.1", "seq": 13, "files": entries}
    (drive / "snapshot-00000013.json").write_bytes(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return drive, local_files(local)


def test_a_mirror_of_the_uncompressed_layout_still_reads_and_restores(tmp_path, capsys):
    drive, expected = a_mirror_of_layout_0_1(tmp_path)
    doc = latest_snapshot(drive)
    assert doc["seq"] == 13 and mirror_state(drive) == "whole"
    assert read_snapshot_file(drive, doc, "trajectories/T-2.jsonl") == \
        expected["trajectories/T-2.jsonl"]
    assert snapshot_file(drive, "run-start.json") == expected["run-start.json"]
    assert cli.main(["run-commit", "--mirror-dir", str(drive)]) == 0
    assert capsys.readouterr().out.strip() == "c" * 40
    restored = tmp_path / "vm2"
    restore(drive, restored)
    got = local_files(restored)
    assert got.pop("ledger.jsonl").startswith(expected.pop("ledger.jsonl"))
    assert got == expected


def test_a_mirror_of_the_older_layout_is_never_continued_by_this_code(tmp_path):
    drive, _ = a_mirror_of_layout_0_1(tmp_path)
    before = sorted(p.relative_to(drive).as_posix() for p in drive.rglob("*"))
    restore(drive, tmp_path / "vm2")
    with pytest.raises(MirrorError, match="0.1.*the commit it started with"):
        RunMirror(tmp_path / "vm2", drive)
    with pytest.raises(MirrorError, match="0.1.*the commit it started with"):
        RunMirror(tmp_path / "local", drive)  # the run directory the mirror was made from
    assert sorted(p.relative_to(drive).as_posix() for p in drive.rglob("*")) == before


# -- fail closed -----------------------------------------------------------------------------
@pytest.mark.parametrize("damage", ["truncated", "not gzip"])
def test_a_damaged_compressed_blob_is_an_error_that_names_its_file(damage, tmp_path):
    local, drive = a_run(tmp_path)
    RunMirror(local, drive).snapshot()
    doc = latest_snapshot(drive)
    blob = mirror_module._blob(drive, doc["files"]["trajectories/T-1.jsonl"]["sha256"])
    data = blob.read_bytes()
    blob.write_bytes(data[:len(data) // 2] if damage == "truncated" else b"plain text")
    with pytest.raises(MirrorError, match="trajectories/T-1.jsonl"):
        read_snapshot_file(drive, doc, "trajectories/T-1.jsonl")
    with pytest.raises(MirrorError, match="no whole snapshot.*trajectories/T-1.jsonl"):
        restore(drive, tmp_path / "vm2")
    assert not (tmp_path / "vm2").exists()


def test_a_manifest_of_a_layout_this_code_does_not_know_fails_closed(tmp_path, capsys):
    local, drive = a_run(tmp_path)
    RunMirror(local, drive).snapshot()
    doc = json.loads((drive / "snapshot-00000001.json").read_bytes())
    doc.update(seq=2, schema="heritagent/exp0001-mirror/0.9")
    (drive / "snapshot-00000002.json").write_text(json.dumps(doc))
    for read in (latest_snapshot, mirror_state, lambda d: restore(d, tmp_path / "vm2"),
                 lambda d: RunMirror(local, d)):
        with pytest.raises(MirrorError, match="0.9.*newer version of the code"):
            read(drive)
    assert not (tmp_path / "vm2").exists()
    assert cli.main(["run-commit", "--mirror-dir", str(drive)]) == 2
    out = capsys.readouterr()
    assert out.out == "" and "newer version of the code" in out.err


# -- mirror-check: the free space of the mirror's filesystem ---------------------------------
Usage = namedtuple("Usage", "total used free")


def mirror_check(monkeypatch, drive: Path, usage, *extra: str) -> int:
    def disk_usage(path):
        if isinstance(usage, BaseException):
            raise usage
        return usage

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    return cli.main(["mirror-check", "--mirror-dir", str(drive), *extra])


def test_mirror_check_fails_below_the_free_space_a_run_needs(tmp_path, monkeypatch, capsys):
    drive = tmp_path / "drive"
    assert mirror_check(monkeypatch, drive, Usage(15 * 10**9, 14 * 10**9, 10**9)) == 2
    err = capsys.readouterr().err
    assert "1.0 GB free of 15.0 GB" in err and "10.0 GB" in err
    assert "Trash" in err and "30 days" in err and "mirror-check" in err
    assert not (drive / ".mirror-check").exists()
    # a smaller bound passes, and the report shows what the filesystem reported
    assert mirror_check(monkeypatch, drive, Usage(15 * 10**9, 14 * 10**9, 10**9),
                        "--need-gb", "0.5") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["total_bytes"] == 15 * 10**9 and report["free_bytes"] == 10**9
    assert report["need_bytes"] == 5 * 10**8


def test_mirror_check_passes_with_room_and_reports_a_zero_total_as_unknown(
        tmp_path, monkeypatch, capsys):
    drive = tmp_path / "drive"
    assert mirror_check(monkeypatch, drive, Usage(10**12, 0, 10**12)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["free_bytes"] == 10**12 and report["need_bytes"] == 10 * 10**9
    for unknown in (Usage(0, 0, 0), OSError(95, "Operation not supported")):
        assert mirror_check(monkeypatch, drive, unknown) == 0
        out = capsys.readouterr()
        report = json.loads(out.out)
        assert report["total_bytes"] is None and report["free_bytes"] is None
        assert "unknown" in out.err
    for wrong in ("-1", "inf", "nan"):
        assert cli.main(["mirror-check", "--mirror-dir", str(drive), "--need-gb", wrong]) == 2
