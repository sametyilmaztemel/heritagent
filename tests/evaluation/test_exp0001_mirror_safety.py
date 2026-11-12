"""The run mirror keeps a paid run's records safe (ADR-0008 D5): a restore
is all or nothing; a mirror with snapshots but none whole, or one that
cannot be read now, is never a new run; a snapshot goes only into the run
directory's own mirror, and only while it is there; a run directory behind
its mirror is not resumed; and the run's own failures end with an
explained error. No GPU, no vLLM, no ALFWorld."""

import gzip
import json
import os
import shutil
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

import evaluation.exp0001.mirror as mirror_module
from evaluation.exp0001 import cli
from evaluation.exp0001.ledger import RunLedger
from evaluation.exp0001.mirror import (
    MirrorError,
    RunMirror,
    latest_snapshot,
    mirror_state,
    read_snapshot_file,
    restore,
)
from tests.evaluation.test_exp0001_colab import command_argv, files, stub_command
from tests.evaluation.test_exp0001_lifetime import Clock


def a_run(local: Path, episodes: int = 3, start: str = "s") -> RunLedger:
    """A run directory: a start record, a ledger and one trajectory per
    recorded episode."""
    local.mkdir(parents=True, exist_ok=True)
    (local / "run-start.json").write_text(json.dumps({"run": start, "code": {"commit": "c" * 40}}))
    ledger = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    (local / "trajectories").mkdir(exist_ok=True)
    for index in range(episodes):
        more(local, ledger, index)
    return ledger


def more(local: Path, ledger: RunLedger, index: int) -> None:
    ledger.admit(f"e{index}")
    ledger.record(f"e{index}", model_calls=1, outcome={})
    (local / "trajectories" / f"T-{index}.jsonl").write_text(f"trajectory {index}\n")


def mirrored(tmp_path: Path, episodes: int = 3):
    local, drive = tmp_path / "local", tmp_path / "drive"
    a_run(local, episodes)
    assert RunMirror(local, drive).snapshot() == 1
    return local, drive


def snapshot_files(drive: Path) -> dict:
    doc = latest_snapshot(drive)
    return {rel: read_snapshot_file(drive, doc, rel) for rel in doc["files"]}


def restore_notes(local: Path) -> list:
    return [json.loads(line) for line in (local / "ledger.jsonl").read_text().splitlines()
            if json.loads(line).get("type") == "restore"]


def failing_read(monkeypatch, error, *, after: int, drive: Path):
    """Blob reads from ``drive`` raise ``error`` from the ``after``-th on."""
    real, count = Path.read_bytes, [0]

    def read_bytes(self):
        if drive / "blobs" in self.parents:
            count[0] += 1
            if count[0] >= after:
                raise error
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)


# -- a restore is all or nothing -----------------------------------------------------------
@pytest.mark.parametrize("error", [OSError(5, "Input/output error"), KeyboardInterrupt()])
def test_a_restore_cut_part_way_leaves_nothing_and_the_next_command_restores_in_full(
        error, tmp_path, monkeypatch, capsys):
    """A Drive read error or a Stop part-way through `run`'s restore: the
    local directory stays absent, and the same command again restores every
    file and appends one restore note."""
    local, drive = mirrored(tmp_path)
    expected = snapshot_files(drive)
    stub_command(monkeypatch, tmp_path)
    from evaluation.exp0001 import pipeline
    seen = []

    class Run:
        def __init__(self, **kwargs):
            seen.append(files(kwargs["run_dir"]))

        def run(self):
            return {"status": "complete"}

    monkeypatch.setattr(pipeline, "Exp0001Run", Run)
    fresh = tmp_path / "vm2"
    with monkeypatch.context() as broken:
        failing_read(broken, error, after=3, drive=drive)
        code = cli.main(command_argv("run", fresh, drive))
    assert code == (130 if isinstance(error, KeyboardInterrupt) else 2)
    assert not fresh.exists() and seen == []
    assert not fresh.with_name(fresh.name + mirror_module.RESTORING).exists()
    capsys.readouterr()
    assert cli.main(command_argv("run", fresh, drive)) == 0
    restored = seen[0]
    ledger = restored.pop("ledger.jsonl")
    assert ledger.startswith(expected.pop("ledger.jsonl"))
    assert {k: v for k, v in restored.items() if k not in mirror_module.SKIPPED_NAMES} == expected
    assert len(restore_notes(fresh)) == 1
    assert "[restore] 5/5 files" in capsys.readouterr().out


def test_a_killed_restores_staging_directory_is_removed_by_the_next_restore(tmp_path):
    local, drive = mirrored(tmp_path)
    target = tmp_path / "vm2"
    staging = target.with_name(target.name + mirror_module.RESTORING)
    (staging / "trajectories").mkdir(parents=True)
    (staging / "ledger.jsonl").write_text("half")
    restore(drive, target)
    assert not staging.exists() and len(restore_notes(target)) == 1


def test_a_snapshot_whose_file_fails_its_hash_gives_way_to_the_older_whole_one(
        tmp_path, capsys):
    local, drive = mirrored(tmp_path)
    ledger = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    more(local, ledger, 3)
    mirror = RunMirror(local, drive)
    assert mirror.snapshot() == 2
    digest = latest_snapshot(drive)["files"]["trajectories/T-3.jsonl"]["sha256"]
    mirror_module._blob(drive, digest).write_text("tampered")
    doc = restore(drive, tmp_path / "vm2")
    assert doc["seq"] == 1 and restore_notes(tmp_path / "vm2")[0]["snapshot_seq"] == 1
    err = capsys.readouterr().err
    assert "snapshot 2 is not whole" in err and "T-3.jsonl" in err
    first = json.loads((drive / "snapshot-00000001.json").read_text())
    mirror_module._blob(drive, first["files"]["ledger.jsonl"]["sha256"]).write_text("tampered")
    with pytest.raises(MirrorError, match="snapshot 2: .*snapshot 1: ledger.jsonl"):
        restore(drive, tmp_path / "vm3")
    assert not (tmp_path / "vm3").exists()


def test_a_blob_that_vanishes_during_a_restore_is_an_explained_error(tmp_path, monkeypatch):
    local, drive = mirrored(tmp_path)
    real = mirror_module._blob_names

    def listed_then_collected(mirror):
        names = real(mirror)
        for blob in (mirror / "blobs").glob("*/*"):
            blob.unlink()  # a live run collects them after the listing
        return names

    monkeypatch.setattr(mirror_module, "_blob_names", listed_then_collected)
    with pytest.raises(MirrorError, match="gone from the mirror"):
        restore(drive, tmp_path / "vm2")
    assert not (tmp_path / "vm2").exists()


# -- a mirror without a whole snapshot is never a new run ----------------------------------
def broken(tmp_path):
    """Two manifests whose ledger blob never reached Drive."""
    local, drive = mirrored(tmp_path)
    ledger = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    more(local, ledger, 3)
    assert RunMirror(local, drive).snapshot() == 2
    for doc_file in drive.glob("snapshot-*.json"):
        digest = json.loads(doc_file.read_text())["files"]["ledger.jsonl"]["sha256"]
        mirror_module._blob(drive, digest).unlink(missing_ok=True)
    assert mirror_state(drive) == "broken"
    return local, drive


@pytest.mark.parametrize("command", ["run", "pilot"])
def test_a_mirror_with_snapshots_but_none_whole_stops_every_command_that_would_start_a_run(
        command, tmp_path, monkeypatch, capsys):
    local, drive = broken(tmp_path)
    before = files(drive)
    assert cli.main(["run-commit", "--mirror-dir", str(drive)]) == 2
    err = capsys.readouterr()
    assert err.out == "" and "no whole snapshot" in err.err and "new RUN_ID" in err.err
    stub_command(monkeypatch, tmp_path)
    fresh = tmp_path / "vm2"
    assert cli.main(command_argv(command, fresh, drive)) == 2
    assert "Trash" in capsys.readouterr().err
    assert not fresh.exists() and files(drive) == before


def test_the_run_directory_itself_makes_its_broken_mirror_whole_again(tmp_path):
    local, drive = broken(tmp_path)
    assert cli.main(["snapshot", "--run-dir", str(local), "--mirror-dir", str(drive)]) == 0
    assert mirror_state(drive) == "whole"


def test_a_manifest_that_cannot_be_read_now_is_never_taken_for_none(tmp_path, monkeypatch,
                                                                     capsys):
    local, drive = mirrored(tmp_path)
    real_bytes, real_text = Path.read_bytes, Path.read_text

    def eio(real):
        def read(self, *args, **kwargs):
            if self.name.startswith("snapshot-"):
                raise OSError(5, "Input/output error")
            return real(self, *args, **kwargs)
        return read

    monkeypatch.setattr(Path, "read_bytes", eio(real_bytes))
    monkeypatch.setattr(Path, "read_text", eio(real_text))
    assert cli.main(["run-commit", "--mirror-dir", str(drive)]) == 2
    out = capsys.readouterr()
    assert out.out == "" and "cannot be read now" in out.err
    with pytest.raises(MirrorError, match="run the command again"):
        cli._restore_if_missing(tmp_path / "vm2", drive)
    assert not (tmp_path / "vm2").exists()


# -- a snapshot goes only into the run directory's own mirror, while it is there -----------
def test_a_mirror_gone_between_two_snapshots_fails_the_last_one_and_keeps_the_marker(tmp_path):
    """An unmounted Drive leaves an empty mount point: the snapshot fails,
    the marker stays, and nothing is made where the mirror was."""
    mount = tmp_path / "drive"
    local, drive = tmp_path / "local", mount / "MyDrive" / "mirror"
    drive.mkdir(parents=True)
    a_run(local)
    mirror = RunMirror(local, drive)
    assert mirror.snapshot() == 1
    shutil.rmtree(mount / "MyDrive")  # unmounted
    assert cli._final_snapshot(mirror, local) is False
    assert (local / cli.UNMIRRORED).exists() and list(mount.iterdir()) == []
    # a new process (`snapshot`, the recovery step) fails the same way, before any write
    assert cli.main(["snapshot", "--run-dir", str(local), "--mirror-dir", str(drive)]) == 2
    assert (local / cli.UNMIRRORED).exists() and list(mount.iterdir()) == []
    drive.mkdir(parents=True)  # a path made again, but not the mirror
    assert cli.main(["snapshot", "--run-dir", str(local), "--mirror-dir", str(drive)]) == 2
    assert (local / cli.UNMIRRORED).exists() and list(drive.iterdir()) == []


def test_a_new_mirror_is_made_without_parents_and_a_lone_file_without_a_ledger_is_no_run(
        tmp_path):
    local = tmp_path / "local"
    a_run(local)
    with pytest.raises(MirrorError, match="cannot be used"):
        RunMirror(local, tmp_path / "MyDrive" / "heritagent" / "mirror")
    assert not (tmp_path / "MyDrive").exists()
    drive = tmp_path / "drive"
    assert RunMirror(local, drive).snapshot() == 1
    leftover = tmp_path / "vm2"
    (leftover / "registry").mkdir(parents=True)
    with pytest.raises(cli.CliError, match="holds files but no ledger.*move"):
        cli._restore_if_missing(leftover, drive)


def test_after_a_failed_snapshot_the_next_one_lists_the_mirror_again(tmp_path, monkeypatch):
    """A blob written before a failure need not be in the mirror afterwards
    (a remount): the next manifest never names a blob the mirror lacks."""
    local, drive = mirrored(tmp_path)
    mirror = RunMirror(local, drive)
    (local / "trajectories" / "T-9.jsonl").write_text("new\n")
    real = mirror_module._write_atomic

    def no_manifest(path, data):
        if path.name.startswith("snapshot-"):
            raise OSError(5, "Input/output error")
        real(path, data)

    monkeypatch.setattr(mirror_module, "_write_atomic", no_manifest)
    assert mirror.snapshot() is None
    monkeypatch.setattr(mirror_module, "_write_atomic", real)
    for blob in (drive / "blobs").glob("*/*"):
        if gzip.decompress(blob.read_bytes()) == b"new\n":
            blob.unlink()  # the upload was lost
    seq = mirror.snapshot()
    assert seq is not None and latest_snapshot(drive)["seq"] == seq


def test_a_snapshot_never_takes_over_another_run_directorys_mirror(tmp_path, capsys):
    local, drive = mirrored(tmp_path)
    other = tmp_path / "pilot"
    a_run(other, start="another run")
    before = files(drive)
    assert cli.main(["snapshot", "--run-dir", str(other), "--mirror-dir", str(drive)]) == 2
    assert "another run directory" in capsys.readouterr().err
    assert files(drive) == before
    # a mistyped mirror path is not made into a new mirror
    (local / cli.UNMIRRORED).write_text("{}")
    typo = tmp_path / "drvie"
    assert cli.main(["snapshot", "--run-dir", str(local), "--mirror-dir", str(typo)]) == 2
    assert not typo.exists() and (local / cli.UNMIRRORED).exists()


def test_a_mirror_written_without_an_id_is_restored_and_resumed_by_its_own_run_only(tmp_path):
    local, drive = mirrored(tmp_path)
    (drive / mirror_module.MIRROR_ID_FILE).unlink()  # a mirror written at 52d4ff2
    restore(drive, tmp_path / "vm2")
    assert not (tmp_path / "vm2" / mirror_module.BINDING_FILE).exists()
    assert RunMirror(tmp_path / "vm2", drive).snapshot() == 2
    other = tmp_path / "other"
    a_run(other, start="another run")
    with pytest.raises(MirrorError, match="another run directory"):
        RunMirror(other, drive)


# -- a run directory behind its mirror is not resumed --------------------------------------
def test_a_run_directory_behind_its_mirror_is_refused_and_a_repaired_or_restored_one_is_not(
        tmp_path):
    local, drive = mirrored(tmp_path)
    stale = tmp_path / "stale"
    shutil.copytree(local, stale)
    ledger = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    more(local, ledger, 3)
    assert RunMirror(local, drive).snapshot() == 2
    with pytest.raises(MirrorError, match="behind its mirror.*Move"):
        RunMirror(stale, drive)
    # a torn last line, snapshotted, then repaired; a restore's note: both resume
    path = local / "ledger.jsonl"
    path.write_bytes(path.read_bytes() + b'{"type":"epi')
    assert RunMirror(local, drive).snapshot() == 3
    assert cli.main(["repair-ledger", "--run-dir", str(local)]) == 0
    assert RunMirror(local, drive).snapshot() == 4
    restore(drive, tmp_path / "vm2")
    assert RunMirror(tmp_path / "vm2", drive).snapshot() == 5


# -- the run's own failures, explained -----------------------------------------------------
def test_run_checks_its_free_disk_before_anything_is_written(tmp_path, monkeypatch, capsys):
    stub_command(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_require_run_disk", RUN_DISK)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: type("U", (), {"free": 2 ** 30})())
    local = tmp_path / "work"
    assert cli.main(command_argv("run", local, tmp_path / "mirror")) == 2
    err = capsys.readouterr().err
    # 16 workers x 50 games x 33 MiB, and 14,500 episodes x 1.25 MB, on one disk
    assert "needs about 43 GB of free disk" in err and "1 GB are free" in err
    assert not local.exists() and not (tmp_path / "mirror").exists()


RUN_DISK = cli._require_run_disk


def test_the_disk_a_resumed_run_needs_counts_only_the_episodes_left(tmp_path, monkeypatch):
    local = tmp_path / "work"
    a_run(local, episodes=4)
    path = local / "ledger.jsonl"
    path.write_bytes(path.read_bytes() + b'{"type":"epi')  # a torn last line is not counted
    monkeypatch.setattr(shutil, "disk_usage", lambda path: type("U", (), {"free": 0})())
    with pytest.raises(cli.CliError, match="14496 episodes left under the cap"):
        cli._require_run_disk(16, local)


@pytest.mark.parametrize("command", ["run", "pilot"])
def test_a_worker_killed_by_memory_or_disk_ends_with_an_explained_error_after_the_last_snapshot(
        command, tmp_path, monkeypatch, capsys):
    from evaluation.exp0001 import pipeline
    stub_command(monkeypatch, tmp_path)

    class Dies:
        def __init__(self, **kwargs):
            pass

        def run(self):
            raise BrokenProcessPool("a child process terminated abruptly")

    def dies(*args, **kwargs):
        raise BrokenProcessPool("a child process terminated abruptly")

    monkeypatch.setattr(pipeline, "Exp0001Run", Dies)
    monkeypatch.setattr(cli, "run_pilot", dies)
    snapshots = []
    monkeypatch.setattr(cli, "_final_snapshot",
                        lambda mirror, local, *, marker=True: snapshots.append(marker) or True)
    assert cli.main(command_argv(command, tmp_path / "work", tmp_path / "mirror")) == 2
    assert "out of memory or disk?" in capsys.readouterr().err and snapshots == [True]


@pytest.mark.parametrize("hours", ["nan", "inf", "0", "-1", "soon"])
def test_stop_after_hours_is_a_finite_positive_number(hours, capsys):
    argv = command_argv("run", "work", "mirror") + ["--stop-after-hours", hours]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(argv)
    assert "a finite number of hours above 0" in capsys.readouterr().err
    assert cli.parser().parse_args(argv[:-1] + ["6"]).stop_after_hours == 6.0


def test_repair_ledger_waits_for_the_command_that_holds_the_run_directory(tmp_path, capsys):
    from evaluation.exp0001.pipeline import lock_run_dir
    local = tmp_path / "work"
    a_run(local)
    path = local / "ledger.jsonl"
    path.write_bytes(path.read_bytes() + b'{"type":"epi')  # a line still being written
    before = path.read_bytes()
    held = lock_run_dir(local)
    try:
        assert cli.main(["repair-ledger", "--run-dir", str(local)]) == 2
    finally:
        held.close()
    assert path.read_bytes() == before and "in use by another run" in capsys.readouterr().err


def test_a_json_file_cut_short_by_a_failed_write_keeps_its_previous_content(tmp_path,
                                                                             monkeypatch):
    """`serve` reads its server file to reuse its own server: a write that
    fails part-way never leaves it half-written."""
    info = tmp_path / "vllm-server.json"
    cli._write_json(info, {"pid": 1})
    real = Path.write_text

    def torn(self, text, *args, **kwargs):
        real(self, text[:5], *args, **kwargs)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", torn)
    with pytest.raises(OSError):
        cli._write_json(info, {"pid": 2})
    monkeypatch.setattr(Path, "write_text", real)
    assert json.loads(info.read_text()) == {"pid": 1} and os.listdir(tmp_path) == [info.name]


# -- a failing mirror ----------------------------------------------------------------------
def test_a_failed_write_leaves_no_temporary_file_in_the_mirror(tmp_path, monkeypatch):
    local, drive = mirrored(tmp_path)
    mirror = RunMirror(local, drive)
    (local / "trajectories" / "T-9.jsonl").write_text("new\n")
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(5, "EIO")))
    assert mirror.snapshot() is None
    monkeypatch.undo()
    assert [p.name for p in drive.rglob("*") if ".tmp-" in p.name] == []
    # and the collection removes any left by a process that died mid-write
    left = drive / "blobs" / "ab" / ".ab12.tmp-1"
    left.parent.mkdir(parents=True, exist_ok=True)
    left.write_text("x")
    (drive / ".snapshot-00000009.json.tmp-1").write_text("x")
    monkeypatch.setattr(mirror_module, "GC_EVERY", 1)
    assert mirror.snapshot() is not None
    assert [p.name for p in drive.rglob("*") if ".tmp-" in p.name] == []


def test_a_failing_mirror_is_tried_again_after_the_interval_or_at_a_terminal_record(tmp_path,
                                                                                   monkeypatch):
    local, drive = mirrored(tmp_path)
    now = [0.0]
    mirror = RunMirror(local, drive, min_interval_seconds=60, clock=lambda: now[0])
    attempts = []

    def failing():
        attempts.append(now[0])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(mirror, "_snapshot", failing)
    for _ in range(5):
        mirror.on_append({"type": "episode"})
    assert attempts == [0.0]
    now[0] = 61.0
    mirror.on_append({"type": "episode"})
    mirror.on_append({"type": "episode"})
    mirror.on_append({"type": "complete"})  # a terminal record is always snapshotted
    assert attempts == [0.0, 61.0, 61.0]
