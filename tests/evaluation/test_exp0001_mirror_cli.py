"""The run mirror through the commands, with the real ``RunMirror``,
ledger and restore (ADR-0008 D5): a restored run directory takes its
mirror over and resumes, also after a fallback to an older snapshot; a
blob that failed its hash is written again; a restored ledger with a torn
last line waits for ``repair-ledger``; a refused attach and a full disk
still leave the unmirrored marker; blobs without a manifest are never a new
run; a torn mirror id is explained. Only the run's preparation and its
episodes are stand-ins. No GPU, no vLLM, no ALFWorld."""

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import evaluation.exp0001.mirror as mirror_module
from evaluation.exp0001 import cli, pipeline
from evaluation.exp0001.ledger import RunLedger
from evaluation.exp0001.mirror import (
    MIRROR_ID_FILE,
    RESTORE_RECORD,
    RunMirror,
    latest_snapshot,
    mirror_state,
    restore,
)
from tests.evaluation.test_exp0001_colab import command_argv, stub_command
from tests.evaluation.test_exp0001_lifetime import Clock


class Run:
    """Stands for Exp0001Run: each session records ``EPISODES`` episodes,
    each with its trajectory, through the real ledger and its mirror hook."""
    EPISODES = 2
    start = {"run": "R1", "code": {"commit": "c" * 40}}
    extra: dict[str, str] = {}  # files every session writes again, the same content each time

    def __init__(self, **kwargs):
        self.dir = Path(kwargs["run_dir"])
        if not (self.dir / "run-start.json").exists():
            (self.dir / "run-start.json").write_text(json.dumps(type(self).start))
        self.ledger = RunLedger(self.dir / "ledger.jsonl", run_id="r", clock=Clock(),
                                on_append=kwargs["on_ledger_append"])

    def run(self):
        (self.dir / "trajectories").mkdir(exist_ok=True)
        done = self.ledger.episodes
        for index in range(done, done + self.EPISODES):
            self.ledger.admit(f"e{index}")
            (self.dir / "trajectories" / f"T-{index}.jsonl").write_text(f"trajectory {index}\n")
            self.ledger.record(f"e{index}", model_calls=1, outcome={})
        for name, content in type(self).extra.items():
            (self.dir / name).write_text(content)
        return {"status": "running"}


@pytest.fixture
def command(monkeypatch, tmp_path):
    """``run`` through cli.main with the real mirror and ledger; returns a
    function of the run directory that gives the exit code."""
    stub_command(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "RunMirror", RunMirror)
    monkeypatch.setattr(cli, "RunLedger", RunLedger)

    class ThisRun(Run):
        extra = {}

    monkeypatch.setattr(pipeline, "Exp0001Run", ThisRun)
    drive = tmp_path / "drive" / "mirror"
    drive.parent.mkdir()

    def run(local: Path, what: str = "run") -> int:
        return cli.main(command_argv(what, local, drive))

    run.drive, run.Run = drive, ThisRun
    return run


def records(local: Path, kind: str | None = None) -> list[dict]:
    lines = [json.loads(line) for line in (local / "ledger.jsonl").read_text().splitlines()]
    return [r for r in lines if kind is None or r.get("type") == kind]


def manifest(drive: Path, seq: int) -> dict:
    return json.loads((drive / f"snapshot-{seq:08d}.json").read_text())


def newest_two(drive: Path) -> tuple[dict, dict]:
    seqs = [seq for seq, _ in mirror_module._snapshot_paths(drive)]
    return manifest(drive, seqs[0]), manifest(drive, seqs[1])


def test_a_restored_run_resumes_on_a_second_and_a_third_machine(command, tmp_path):
    vm1, vm2, vm3 = (tmp_path / f"vm{n}" / "work" for n in (1, 2, 3))
    assert command(vm1) == 0 and command(vm1) == 0
    assert command(vm2) == 0  # restored, taken over, resumed, snapshotted
    assert not (vm2 / RESTORE_RECORD).exists()  # removed by its first snapshot
    assert command(vm2) == 0  # in place
    assert command(vm3) == 0
    assert len(records(vm3, "episode")) == 10 and len(records(vm3, "restore")) == 2


def test_a_directory_restored_and_snapshotted_is_checked_again_when_it_resumes(command, tmp_path,
                                                                                capsys):
    vm1, vm2 = tmp_path / "vm1" / "work", tmp_path / "vm2" / "work"
    assert command(vm1) == 0
    stale = tmp_path / "stale"
    shutil.copytree(vm1, stale)
    assert command(vm2) == 0 and command(vm2) == 0
    shutil.rmtree(vm1)
    shutil.copytree(stale, vm1)  # the first machine's copy, behind the mirror now
    capsys.readouterr()
    assert command(vm1) == 2
    assert "behind its mirror" in capsys.readouterr().err
    assert not (vm1 / cli.UNMIRRORED).exists()  # the mirror holds every record it has


def test_a_fallback_restore_resumes_and_writes_again_the_blob_that_failed(command, tmp_path,
                                                                         capsys):
    """The newest snapshot holds a blob that fails its hash (a file every
    session writes again; its ledger is whole): the restore takes the older
    snapshot and the run resumes, and the next snapshots hold the file in a
    whole blob, so a later restore takes the newest one."""
    vm1 = tmp_path / "vm1" / "work"
    assert command(vm1) == 0
    command.Run.extra = {"derived.json": '{"the same": "every time"}'}
    assert command(vm1) == 0
    newest, older = newest_two(command.drive)
    assert "derived.json" in newest["files"] and "derived.json" not in older["files"]
    mirror_module._blob(command.drive, newest["files"]["derived.json"]["sha256"]).write_bytes(
        b"bit rot")
    vm2 = tmp_path / "vm2" / "work"
    capsys.readouterr()
    assert command(vm2) == 0, capsys.readouterr().err
    assert records(vm2, "restore")[0]["snapshot_seq"] == older["seq"]
    assert command(vm2) == 0
    vm3 = tmp_path / "vm3" / "work"  # the second machine is lost
    latest = latest_snapshot(command.drive)
    assert "derived.json" in latest["files"]
    assert command(vm3) == 0, capsys.readouterr().err
    assert records(vm3, "restore")[-1]["snapshot_seq"] == latest["seq"]
    assert (vm3 / "derived.json").read_text() == command.Run.extra["derived.json"]


def test_a_restore_records_every_file_of_a_snapshot_it_skips(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    ledger = RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    mirror = RunMirror(local, drive)
    (local / "a.json").write_text("a")
    assert mirror.snapshot() == 1
    ledger.admit("e0")
    (local / "b.json").write_text("b")
    (local / "c.json").write_text("c")
    assert mirror.snapshot() == 2
    files = manifest(drive, 2)["files"]
    for rel in ("b.json", "c.json"):
        mirror_module._blob(drive, files[rel]["sha256"]).write_bytes(b"bit rot")
    assert restore(drive, tmp_path / "vm2")["seq"] == 1
    record = json.loads((tmp_path / "vm2" / RESTORE_RECORD).read_text())
    assert [(s["seq"], s["file"], s["sha256"]) for s in record["skipped"]] == [
        (2, rel, files[rel]["sha256"]) for rel in ("b.json", "c.json")]
    assert record["snapshot_seq"] == 1 and record["pending_note"] is None


def test_a_second_session_takes_the_mirror_over_and_stops_the_first(command, tmp_path,
                                                                    monkeypatch, capsys):
    """The first session still runs: it takes one more snapshot between the
    second session's restore and its attach. The second takes the mirror
    over; the first stops at its next snapshot."""
    vm1 = tmp_path / "vm1" / "work"
    assert command(vm1) == 0
    ledger = RunLedger(vm1 / "ledger.jsonl", run_id="r", clock=Clock())
    first = RunMirror(vm1, command.drive)

    def first_goes_on(*args):
        ledger.admit("late")
        ledger.record("late", model_calls=1, outcome={})
        assert first.snapshot() is not None
        return {}

    monkeypatch.setattr(cli, "_run_start", first_goes_on)
    vm2 = tmp_path / "vm2" / "work"
    capsys.readouterr()
    assert command(vm2) == 0, capsys.readouterr().err
    with pytest.raises(mirror_module.MirrorError, match="took over"):
        first.snapshot()


def mirror_a_torn_ledger(command, tmp_path) -> None:
    vm1 = tmp_path / "vm1" / "work"
    assert command(vm1) == 0
    path = vm1 / "ledger.jsonl"
    path.write_bytes(path.read_bytes() + b'{"type":"epi')  # a write a kill interrupted
    assert command(vm1, "snapshot") == 0  # snapshots the torn line as it is


def test_a_restored_ledger_with_a_torn_line_waits_for_repair_ledger(command, tmp_path, capsys):
    """The refusal leaves the marker, so the notebook keeps the runtime for
    repair-ledger; the next snapshot of the same command removes it."""
    mirror_a_torn_ledger(command, tmp_path)
    vm2 = tmp_path / "vm2" / "work"
    capsys.readouterr()
    assert command(vm2) == 2
    err = capsys.readouterr().err
    assert f"python -m evaluation.exp0001.cli repair-ledger --run-dir {vm2}" in err
    assert f'cli("repair-ledger", "--run-dir", "{vm2}")' in err
    assert "then the same command again" in err
    assert f"UNMIRRORED: {vm2} waits for repair-ledger" in err
    assert (vm2 / cli.UNMIRRORED).exists()
    assert (vm2 / "ledger.jsonl").read_bytes().endswith(b'{"type":"epi')  # restored, not cut
    for what in ("run", "snapshot"):  # nothing goes on while its restore note waits
        assert command(vm2, what) == 2
        assert "repair-ledger" in capsys.readouterr().err
        assert (vm2 / cli.UNMIRRORED).exists()
    assert cli.main(["repair-ledger", "--run-dir", str(vm2)]) == 0
    assert (vm2 / cli.UNMIRRORED).exists()  # until a snapshot holds the repaired ledger
    assert command(vm2) == 0
    assert not (vm2 / cli.UNMIRRORED).exists()
    notes = [r["type"] for r in records(vm2) if r.get("type") in ("repair", "restore")]
    assert notes == ["repair", "restore"]
    assert cli.main(["repair-ledger", "--run-dir", str(vm2)]) == 0  # nothing pending: no note
    assert len(records(vm2, "restore")) == 1
    vm3 = tmp_path / "vm3" / "work"
    assert command(vm3) == 0 and len(records(vm3, "restore")) == 2


def test_a_rehearsal_waiting_for_repair_ledger_leaves_no_marker(command, tmp_path, monkeypatch,
                                                                capsys):
    """A rehearsal's records are disposable: it is started again, never resumed."""
    import functools
    mirror_a_torn_ledger(command, tmp_path)
    monkeypatch.setattr(cli, "_run", functools.partial(cli._run, disposable=True))
    vm2 = tmp_path / "vm2" / "work"
    capsys.readouterr()
    assert command(vm2) == 2
    assert f"repair-ledger --run-dir {vm2}" in capsys.readouterr().err
    assert not (vm2 / cli.UNMIRRORED).exists()


def test_a_pending_restore_note_is_appended_once(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    (local / "run-start.json").write_text("{}")
    path = local / "ledger.jsonl"
    path.write_bytes(path.read_bytes() + b'{"torn')
    assert RunMirror(local, drive).snapshot() == 1
    restore(drive, tmp_path / "vm2")
    ledger = tmp_path / "vm2" / "ledger.jsonl"
    record = json.loads((tmp_path / "vm2" / RESTORE_RECORD).read_text())
    with pytest.raises(cli.LedgerError):  # the line is cut on purpose first, never here
        mirror_module.append_pending_restore_note(tmp_path / "vm2")
    cli.repair_torn_tail(ledger)
    mirror_module.append_note(ledger, record["pending_note"])  # a repair stopped before it cleared
    assert mirror_module.append_pending_restore_note(tmp_path / "vm2") == record["pending_note"]
    lines = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert [x["type"] for x in lines[-2:]] == ["repair", "restore"]
    assert mirror_module.append_pending_restore_note(tmp_path / "vm2") is None


def test_the_marker_is_left_when_its_content_cannot_be_written(tmp_path, monkeypatch):
    """A full local disk fails the marker's write through a temporary file;
    the marker itself, an empty file, is still there for the notebooks."""
    local, drive = tmp_path / "local", tmp_path / "drive"
    RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    mirror = RunMirror(local, drive)
    shutil.rmtree(drive)  # Drive gone: the last snapshot fails

    def disk_full(path, document):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli, "_write_json", disk_full)
    assert cli._final_snapshot(mirror, local) is False
    assert (local / cli.UNMIRRORED).exists()


def test_a_sigterm_leaves_the_marker_on_a_full_disk(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    child = ("import time\nfrom pathlib import Path\nfrom evaluation.exp0001 import cli\n"
             "def disk_full(path, document):\n"
             "    raise OSError(28, 'No space left on device')\n"
             "cli._write_json = disk_full\n"
             f"with cli._run_dir_lock(Path({str(local)!r})):\n"
             "    print('held', flush=True)\n"
             "    time.sleep(60)\n")
    repo = Path(cli.__file__).parents[2]
    process = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True,
                               cwd=repo,
                               env={**os.environ, "PYTHONPATH": str(repo)})
    try:
        assert process.stdout.readline().strip() == "held"
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
    finally:
        process.kill()
    assert process.returncode == -signal.SIGTERM
    assert (local / cli.UNMIRRORED).exists()


@pytest.mark.parametrize("what", ["run", "snapshot"])
def test_a_refused_attach_of_a_directory_with_a_ledger_leaves_the_marker(what, command, tmp_path,
                                                                         capsys):
    vm1 = tmp_path / "vm1" / "work"
    assert command(vm1) == 0
    shutil.move(str(command.drive), str(tmp_path / "elsewhere"))  # Drive not mounted
    capsys.readouterr()
    assert command(vm1, what) == 2
    assert "UNMIRRORED" in capsys.readouterr().err
    assert (vm1 / cli.UNMIRRORED).exists()


def test_a_directory_behind_its_mirror_is_refused_without_a_marker(command, tmp_path):
    vm1 = tmp_path / "vm1" / "work"
    assert command(vm1) == 0
    path = vm1 / "ledger.jsonl"
    path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])  # an older copy
    assert command(vm1, "snapshot") == 2
    assert not (vm1 / cli.UNMIRRORED).exists()


def test_a_new_mirror_without_its_folder_names_the_folder(command, tmp_path, capsys):
    assert cli.main(command_argv("run", tmp_path / "vm1", tmp_path / "not-made" / "mirror")) == 2
    err = capsys.readouterr().err
    assert "mount Google Drive first" in err and f"{tmp_path / 'not-made'} does not exist" in err


def test_blobs_without_a_manifest_are_never_a_new_run(command, tmp_path, capsys):
    """The manifests are gone (in the Drive Trash) and the blobs remain:
    no command starts a new run there, and no blob is collected; the run's
    own directory still snapshots."""
    vm1 = tmp_path / "vm1" / "work"
    assert command(vm1) == 0
    for path in command.drive.glob("snapshot-*.json"):
        path.unlink()
    blobs = mirror_module._blob_names(command.drive)
    assert mirror_state(command.drive) == "broken"
    capsys.readouterr()
    assert cli.main(["run-commit", "--mirror-dir", str(command.drive)]) == 2
    out = capsys.readouterr()
    assert out.out == "" and "Drive Trash" in out.err
    command.Run.start = {"run": "a new run", "code": {"commit": "d" * 40}}
    assert command(tmp_path / "vm2" / "work") == 2
    other = tmp_path / "other"  # a directory with a ledger, not bound to this mirror
    shutil.copytree(vm1, other)
    (other / mirror_module.BINDING_FILE).unlink()
    assert command(other) == 2 and "Drive Trash" in capsys.readouterr().err
    assert blobs <= mirror_module._blob_names(command.drive)
    assert command(vm1, "snapshot") == 0


def test_a_torn_mirror_id_is_written_again_from_the_runs_own_binding(command, tmp_path, capsys):
    vm1, vm2 = tmp_path / "vm1" / "work", tmp_path / "vm2" / "work"
    assert command(vm1) == 0
    bound = json.loads((vm1 / mirror_module.BINDING_FILE).read_text())
    (command.drive / MIRROR_ID_FILE).write_bytes(b"")
    capsys.readouterr()
    assert command(vm2) == 2  # a new machine cannot tell: explained
    err = capsys.readouterr().err
    assert f"{command.drive / MIRROR_ID_FILE} is not a mirror id" in err and "delete" in err
    assert not vm2.exists()
    assert command(vm1, "snapshot") == 0  # the run's own directory writes it again
    assert json.loads((command.drive / MIRROR_ID_FILE).read_text()) == bound
    assert command(vm2) == 0


def test_a_torn_mirror_id_deleted_by_hand_restores_the_run(command, tmp_path):
    vm1, vm2 = tmp_path / "vm1" / "work", tmp_path / "vm2" / "work"
    assert command(vm1) == 0
    (command.drive / MIRROR_ID_FILE).write_bytes(b'{"schema": "heritagent/exp0001-mirr')
    assert command(vm2) == 2
    (command.drive / MIRROR_ID_FILE).unlink()  # the advice when no machine holds the run
    assert command(vm2) == 0 and len(records(vm2, "restore")) == 1
    assert (command.drive / MIRROR_ID_FILE).exists()


def test_a_command_lists_the_mirror_once_and_reads_each_manifest_once(tmp_path, monkeypatch):
    local, drive = tmp_path / "local", tmp_path / "drive"
    RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    (local / "run-start.json").write_text("{}")
    mirror = RunMirror(local, drive)
    for _ in range(3):
        assert mirror.snapshot() is not None
    calls = {"blobs": 0, "manifests": 0}
    names, read = mirror_module._blob_names, mirror_module._read_snapshot

    def count_blobs(path):
        calls["blobs"] += 1
        return names(path)

    def count_reads(seq, path):
        calls["manifests"] += 1
        return read(seq, path)

    monkeypatch.setattr(mirror_module, "_blob_names", count_blobs)
    monkeypatch.setattr(mirror_module, "_read_snapshot", count_reads)
    RunMirror(local, drive)
    assert calls == {"blobs": 1, "manifests": 3}


def test_the_space_refusal_names_the_option_and_where_to_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mirror_module, "_space", lambda path: (100 * 10 ** 9, 2 * 10 ** 9))
    assert cli.main(["mirror-check", "--mirror-dir", str(tmp_path / "m")]) == 2
    err = capsys.readouterr().err
    assert "--need-gb" in err and "drive.google.com/settings/storage" in err


def test_the_messages_say_what_to_do_next(tmp_path, capsys):
    (tmp_path / "empty-mirror").mkdir()
    (tmp_path / "empty-mirror" / "snapshot-00000001.json").write_text(json.dumps(
        {"schema": mirror_module.MIRROR_SCHEMA, "seq": 1,
         "files": {"a": {"sha256": "0" * 64, "size": 1}}}))
    mirror_module._blob(tmp_path / "empty-mirror", "0" * 64).parent.mkdir(parents=True)
    mirror_module._blob(tmp_path / "empty-mirror", "0" * 64).write_bytes(b"bit rot")
    with pytest.raises(mirror_module.MirrorError, match="no whole snapshot .*Wait a few minutes"):
        restore(tmp_path / "empty-mirror", tmp_path / "vm")
    import fcntl
    with open(tmp_path / ".vm.restore-lock", "a+") as lock:  # another restore holds it
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(mirror_module.MirrorError, match="wait until it ends"):
            restore(tmp_path / "empty-mirror", tmp_path / "vm")
    with pytest.raises(cli.CliError, match="GiB.*run the command again"):
        cli._require_free({str(tmp_path): 2 ** 60}, "the run")


def test_the_restore_record_is_never_snapshotted(command, tmp_path):
    vm1, vm2 = tmp_path / "vm1" / "work", tmp_path / "vm2" / "work"
    assert command(vm1) == 0
    restore(command.drive, vm2)
    assert (vm2 / RESTORE_RECORD).exists()
    assert RESTORE_RECORD not in latest_snapshot(command.drive)["files"]
    assert command(vm2, "snapshot") == 0 and not (vm2 / RESTORE_RECORD).exists()
    assert RESTORE_RECORD not in latest_snapshot(command.drive)["files"]
