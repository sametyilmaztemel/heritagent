"""A failing mirror pauses the run (ADR-0008 D5): after
``cli.MIRROR_FAILURES_BEFORE_PAUSE`` failed snapshots in a row no new
episode is admitted, the admitted ones are recorded, and the last snapshot
decides the exit. Through ``cli.main(["run", ...])`` with the real
``RunMirror`` and ledger; only the run's preparation and its episodes are
stand-ins, and the stand-in asks the pause predicate before each admission
as the executor does. No GPU, no vLLM, no ALFWorld."""

import json
from pathlib import Path

import pytest

from evaluation.exp0001 import cli, pipeline
from evaluation.exp0001.episodes import RunPaused
from evaluation.exp0001.ledger import RunLedger
from evaluation.exp0001.mirror import RunMirror
from tests.evaluation.test_exp0001_colab import command_argv, stub_command
from tests.evaluation.test_exp0001_lifetime import Clock


class PausingRun:
    """Stands for Exp0001Run: ``EPISODES`` episodes through the real ledger
    and its mirror hook, the pause predicate asked before each admission;
    ``after(index)`` runs once episode ``index`` is recorded."""
    EPISODES = 8
    after = staticmethod(lambda index: None)
    made: list = []

    def __init__(self, **kwargs):
        self.dir = Path(kwargs["run_dir"])
        if not (self.dir / "run-start.json").exists():
            (self.dir / "run-start.json").write_text(json.dumps({"run": "R1"}))
        self.ledger = RunLedger(self.dir / "ledger.jsonl", run_id="r", clock=Clock(),
                                on_append=kwargs["on_ledger_append"])
        self.pause_when, self.answers, self.in_a_row = kwargs["pause_when"], [], []
        type(self).made.append(self)

    def run(self):
        mirror = self.made_mirrors[-1]
        for index in range(self.ledger.episodes, self.EPISODES):
            if self.pause_when is not None:
                self.answers.append(bool(self.pause_when()))
                if self.answers[-1]:
                    raise RunPaused(f"paused before 'e{index}'")
            self.in_a_row.append(mirror.failures_in_a_row)  # at this admission
            self.ledger.admit(f"e{index}")
            self.ledger.record(f"e{index}", model_calls=1, outcome={})
            type(self).after(index)
        return {"status": "running"}


@pytest.fixture
def command(monkeypatch, tmp_path):
    """``run`` through cli.main, its mirror snapshotting at every ledger
    record; ``unmount()`` and ``mount()`` take the mirror away and back as an
    unmounted and remounted Drive does."""
    stub_command(monkeypatch, tmp_path)
    mirrors = []

    def make_mirror(*args, **kwargs):
        mirrors.append(RunMirror(*args, min_interval_seconds=0, **kwargs))
        return mirrors[-1]

    monkeypatch.setattr(cli, "RunMirror", make_mirror)
    monkeypatch.setattr(cli, "RunLedger", RunLedger)

    class ThisRun(PausingRun):
        made = []
        made_mirrors = mirrors

    monkeypatch.setattr(pipeline, "Exp0001Run", ThisRun)
    drive = tmp_path / "drive" / "mirror"
    drive.parent.mkdir()
    away = tmp_path / "drive" / "unmounted"
    local = tmp_path / "vm1" / "work"

    def run() -> int:
        return cli.main(command_argv("run", local, drive))

    run.Run, run.local, run.mirrors, run.drive = ThisRun, local, mirrors, drive
    run.unmount = lambda: drive.rename(away)
    run.mount = lambda: away.exists() and away.rename(drive)
    return run


def records(local: Path, kind: str) -> list[str]:
    lines = [json.loads(line) for line in (local / "ledger.jsonl").read_text().splitlines()]
    return [r["key"] for r in lines if r.get("type") == kind]


def test_a_mirror_gone_mid_run_pauses_the_run_and_leaves_the_marker(command, capsys):
    """Drive goes away after the second episode and does not come back:
    after five failed snapshots in a row no episode is admitted, every
    admitted one is recorded, the last snapshot fails, the marker stays."""
    command.Run.after = staticmethod(lambda index: index == 1 and command.unmount())
    assert command() == cli.EXIT_UNMIRRORED
    run, mirror = command.Run.made[-1], command.mirrors[-1]
    assert run.answers[-1] is True and not any(run.answers[:-1])
    assert all(n < cli.MIRROR_FAILURES_BEFORE_PAUSE for n in run.in_a_row)
    assert mirror.failures_in_a_row >= cli.MIRROR_FAILURES_BEFORE_PAUSE
    admitted, recorded = records(command.local, "attempt"), records(command.local, "episode")
    assert admitted == recorded and 2 < len(recorded) < PausingRun.EPISODES
    assert (command.local / cli.UNMIRRORED).exists()
    err = capsys.readouterr().err
    assert "snapshots to the mirror failed in a row" in err
    # the pause names the mirror's own last error
    assert mirror.last_error.startswith(f"the mirror {command.drive} cannot be read (")
    assert f"failed in a row (last: {mirror.last_error}); no new episode is admitted" in err
    assert "exist only on this runtime: keep the runtime, mount Google Drive again " \
        "(force_remount=True) and run the same command again, or `snapshot` to save the " \
        "records without resuming. The last snapshot is tried now." in err
    assert "UNMIRRORED: the last snapshot failed" in err


def test_failed_snapshots_that_are_not_in_a_row_do_not_pause_the_run(command, capsys):
    """Two bursts of four failed snapshots, a good one between them: eight
    failures, never five in a row; the run goes to its end."""
    burst = cli.MIRROR_FAILURES_BEFORE_PAUSE - 1  # failed snapshots; two per episode

    def drive(index: int) -> None:
        if index in (1, 4):
            command.unmount()
        elif index in (1 + burst // 2, 4 + burst // 2):
            command.mount()  # the next record's snapshot succeeds

    command.Run.after = staticmethod(drive)
    assert command() == 0, capsys.readouterr().err
    run, mirror = command.Run.made[-1], command.mirrors[-1]
    assert run.answers == [False] * PausingRun.EPISODES  # asked at every admission
    assert mirror.failures == 2 * burst and max(run.in_a_row) == burst
    assert len(records(command.local, "episode")) == PausingRun.EPISODES
    assert not (command.local / cli.UNMIRRORED).exists()
    assert "failed in a row" not in capsys.readouterr().err


def test_a_mirror_back_before_the_last_snapshot_ends_as_a_pause(command, monkeypatch, capsys):
    """Drive comes back after the pause, before the last snapshot: that
    snapshot succeeds, the command exits 0 without a marker and says that
    the run paused and why."""
    command.Run.after = staticmethod(lambda index: index == 1 and command.unmount())
    final = cli._final_snapshot

    def mounted_again(*args, **kwargs):
        command.mount()
        return final(*args, **kwargs)

    monkeypatch.setattr(cli, "_final_snapshot", mounted_again)
    assert command() == 0
    assert command.Run.made[-1].answers[-1] is True
    assert not (command.local / cli.UNMIRRORED).exists()
    err = capsys.readouterr().err
    assert "paused (paused before" in err and "snapshots to the mirror failed in a row" in err
    assert "the last snapshot succeeded: the mirror holds every record again" in err
    admitted, recorded = records(command.local, "attempt"), records(command.local, "episode")
    assert admitted == recorded and len(recorded) < PausingRun.EPISODES
    assert command() == 0  # the same command resumes and finishes
    assert len(records(command.local, "episode")) == PausingRun.EPISODES


def test_the_predicate_is_installed_without_a_time_limit(command):
    assert "--stop-after-hours" not in command_argv("run", command.local, "m")
    assert command() == 0
    run, mirror = command.Run.made[-1], command.mirrors[-1]
    assert run.pause_when is not None
    mirror.failures_in_a_row = cli.MIRROR_FAILURES_BEFORE_PAUSE - 1
    assert run.pause_when() is False
    mirror.failures_in_a_row = cli.MIRROR_FAILURES_BEFORE_PAUSE
    assert run.pause_when() is True


def test_a_snapshot_that_succeeds_resets_the_count_in_a_row(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    RunLedger(local / "ledger.jsonl", run_id="r", clock=Clock())
    mirror = RunMirror(local, drive, min_interval_seconds=0)
    drive.rename(tmp_path / "away")
    assert mirror.snapshot() is None and mirror.snapshot() is None
    assert (mirror.failures, mirror.failures_in_a_row) == (2, 2)
    (tmp_path / "away").rename(drive)
    assert mirror.snapshot() is not None
    assert (mirror.failures, mirror.failures_in_a_row) == (2, 0)


@pytest.mark.parametrize("disposable, outer", [(True, None), (True, lambda: False),
                                               (False, None)])
def test_a_failing_mirror_pauses_the_real_run_and_never_a_rehearsal(
        command, monkeypatch, capsys, disposable, outer):
    """Through the real ``_run`` with ``disposable`` set as ``rehearse-run``
    sets it: Drive goes away after the second episode and six or more
    snapshots fail in a row. A rehearsal goes to its end (its records are
    disposable, and its directories cannot be run again), with or without
    an outer ``pause_when``; its failed last snapshot ends it with 3 and no
    marker. The real run pauses."""
    real = cli._run

    def as_set(*args, **kwargs):
        return real(*args, **{**kwargs, "disposable": disposable, "pause_when": outer})

    monkeypatch.setattr(cli, "_run", as_set)
    command.Run.after = staticmethod(lambda index: index == 1 and command.unmount())
    assert command() == cli.EXIT_UNMIRRORED
    run, mirror = command.Run.made[-1], command.mirrors[-1]
    recorded = records(command.local, "episode")
    err = capsys.readouterr().err
    if disposable:
        assert max(run.in_a_row) >= cli.MIRROR_FAILURES_BEFORE_PAUSE + 1  # six in a row
        assert len(recorded) == PausingRun.EPISODES
        assert (run.pause_when is None) == (outer is None)
        assert not any(run.answers)
        assert not (command.local / cli.UNMIRRORED).exists()
        assert "failed in a row" not in err
        assert "the rehearsal's last snapshot failed" in err
    else:
        assert run.answers[-1] is True and len(recorded) < PausingRun.EPISODES
        assert (command.local / cli.UNMIRRORED).exists()
        assert "snapshots to the mirror failed in a row" in err
