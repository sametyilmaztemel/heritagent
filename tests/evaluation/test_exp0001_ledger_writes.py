"""A ledger write that fails part-way (ADR-0008 D5): the line is cut back
off, or left as a torn last line that repair_torn_tail cuts off, and the
process writes nothing more, so no later line is glued onto it. Scripted
games and models only."""

import errno
import os
from pathlib import Path

import pytest

from evaluation.exp0001.episodes import EpisodeExecutor
from evaluation.exp0001.ledger import LedgerError, RunLedger, repair_torn_tail
from evaluation.exp0001.lifetime import lifetime_specs
from evaluation.exp0001.protocol import CAPS, MINING_SEEDS
from runtime.seed_g0 import g0_runtime_config
from tests.conftest import registry  # noqa: F401 (fixture)
from tests.evaluation.exp0001_workers import GameFactory
from tests.evaluation.test_exp0001_lifetime import (
    COMMIT,
    RUN_ID,
    Clock,
    SeededModel,
    tasks,
)


class TornWrite:
    """A file handle whose first write of a line holding ``marker`` puts
    half the line on disk, then fails as a full disk does."""

    def __init__(self, handle, marker, armed):
        self.handle, self.marker, self.armed = handle, marker, armed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.close()

    def write(self, text):
        if self.marker not in text:
            return self.handle.write(text)
        self.armed[0] = False
        self.handle.write(text[: len(text) // 2])
        self.handle.flush()
        raise OSError(errno.ENOSPC, "No space left on device")

    def flush(self):
        self.handle.flush()


def fail_once(monkeypatch, path, marker):
    """The first append to ``path`` whose line holds ``marker`` is torn."""
    real_open, armed = Path.open, [True]

    def open_(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if armed[0] and mode == "a" and self == path:
            return TornWrite(handle, marker, armed)
        return handle

    monkeypatch.setattr(Path, "open", open_)


def test_a_failed_write_is_cut_off_and_nothing_more_is_written(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock())
    ledger.admit("a")
    ledger.admit("b")
    before = path.read_bytes()
    fail_once(monkeypatch, path, '"key":"a"')
    with pytest.raises(LedgerError, match="No space left"):
        ledger.record("a", model_calls=1, outcome={})
    assert path.read_bytes() == before
    with pytest.raises(LedgerError, match="earlier write in this process failed"):
        ledger.record("b", model_calls=1, outcome={})
    assert path.read_bytes() == before
    reloaded = RunLedger(path, run_id=RUN_ID, clock=Clock())
    assert (reloaded.episodes, reloaded.aborted_attempts) == (0, 2)


def test_a_failed_write_that_cannot_be_cut_off_leaves_a_repairable_tail(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock())
    ledger.admit("a")
    ledger.admit("b")
    fail_once(monkeypatch, path, '"key":"a"')

    def no_truncate(*args):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "truncate", no_truncate)
    with pytest.raises(LedgerError, match="No space left"):
        ledger.record("a", model_calls=1, outcome={})
    with pytest.raises(LedgerError, match="earlier write in this process failed"):
        ledger.record("b", model_calls=1, outcome={})
    monkeypatch.undo()
    assert not path.read_bytes().endswith(b"\n")
    assert repair_torn_tail(path)["type"] == "repair"
    reloaded = RunLedger(path, run_id=RUN_ID, clock=Clock())
    assert (reloaded.episodes, reloaded.aborted_attempts) == (0, 2)


class FaultyWrite(TornWrite):
    """The first write of a line holding ``marker`` raises ``error`` before
    any byte reaches the file, after half the line, or after the whole line."""

    def __init__(self, handle, marker, armed, point, error):
        super().__init__(handle, marker, armed)
        self.point, self.error = point, error

    def write(self, text):
        if self.marker not in text:
            return self.handle.write(text)
        self.armed[0] = False
        cut = {"before": 0, "half": len(text) // 2, "after": len(text)}[self.point]
        self.handle.write(text[:cut])
        self.handle.flush()
        raise self.error


def fault_once(monkeypatch, path, marker, point, error):
    real_open, armed = Path.open, [True]

    def open_(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if armed[0] and mode == "a" and self == path:
            return FaultyWrite(handle, marker, armed, point, error)
        return handle

    monkeypatch.setattr(Path, "open", open_)


def test_the_first_failed_write_names_the_ledger_the_error_and_how_to_resume(tmp_path,
                                                                             monkeypatch):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock())
    ledger.admit("a")
    fail_once(monkeypatch, path, '"key":"a"')
    with pytest.raises(LedgerError) as raised:
        ledger.record("a", model_calls=1, outcome={})
    message = str(raised.value)
    assert str(path) in message and "No space left on device" in message
    assert "resume the run in a new process" in message and "repair-ledger" not in message
    assert isinstance(raised.value.__cause__, OSError)


def test_a_ledger_that_cannot_be_measured_before_a_write_gives_an_explained_error(tmp_path,
                                                                                 monkeypatch):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock())
    ledger.admit("a")
    before = path.read_bytes()
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self == path:
            raise OSError(errno.EIO, "Input/output error")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    with pytest.raises(LedgerError) as raised:
        ledger.record("a", model_calls=1, outcome={})
    monkeypatch.undo()
    message = str(raised.value)
    assert str(path) in message and "Input/output error" in message
    assert "nothing was written" in message
    assert isinstance(raised.value.__cause__, OSError)
    assert path.read_bytes() == before
    ledger.record("a", model_calls=1, outcome={})
    assert RunLedger(path, run_id=RUN_ID, clock=Clock()).episodes == 1


@pytest.mark.parametrize("point", ["before", "half", "after"])
def test_any_error_inside_a_write_cuts_the_line_and_stops_the_writes(tmp_path, monkeypatch,
                                                                     point):
    """Not only a disk error: whatever reached the file is cut back off, so
    file and memory agree, and nothing more is written."""
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock())
    ledger.admit("a")
    ledger.admit("b")
    before = path.read_bytes()
    fault_once(monkeypatch, path, '"key":"a"', point, RuntimeError("encoder broke"))
    with pytest.raises(LedgerError, match="encoder broke"):
        ledger.record("a", model_calls=1, outcome={})
    assert path.read_bytes() == before
    with pytest.raises(LedgerError, match="earlier write in this process failed"):
        ledger.record("b", model_calls=1, outcome={})
    assert path.read_bytes() == before


@pytest.mark.parametrize("point", ["before", "after"])
def test_a_stop_that_leaves_no_part_written_line_keeps_the_ledger_writable(tmp_path,
                                                                           monkeypatch, point):
    """A Stop is not a disk error: a line it left whole is kept and counted
    on reload, and the episodes still in flight are still recorded."""
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock())
    ledger.admit("a")
    ledger.admit("b")
    before = path.read_bytes()
    fault_once(monkeypatch, path, '"key":"a"', point, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        ledger.record("a", model_calls=1, outcome={})
    kept = path.read_bytes()
    assert (kept == before) == (point == "before") and kept.endswith(b"\n")
    ledger.record("b", model_calls=1, outcome={})
    assert path.read_bytes().startswith(kept)
    reloaded = RunLedger(path, run_id=RUN_ID, clock=Clock())
    assert reloaded.episodes == (2 if point == "after" else 1)
    assert reloaded.aborted_attempts == (0 if point == "after" else 1)


def test_a_stop_in_the_middle_of_a_line_cuts_it_and_stops_the_writes(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock())
    ledger.admit("a")
    ledger.admit("b")
    before = path.read_bytes()
    fault_once(monkeypatch, path, '"key":"a"', "half", KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        ledger.record("a", model_calls=1, outcome={})
    assert path.read_bytes() == before
    with pytest.raises(LedgerError, match="a Stop interrupted it part-way"):
        ledger.record("b", model_calls=1, outcome={})
    assert path.read_bytes() == before


@pytest.mark.parametrize("point, error", [("half", OSError(errno.ENOSPC, "No space left")),
                                          ("half", KeyboardInterrupt()),
                                          ("before", KeyboardInterrupt()),
                                          ("after", OSError(errno.ENOSPC, "No space left"))])
def test_a_failed_header_write_leaves_no_file_and_a_fresh_start_works(tmp_path, monkeypatch,
                                                                      point, error):
    """An empty or half-written ledger would be refused at the next start as
    having no header; the file is removed instead."""
    path = tmp_path / "ledger.jsonl"
    fault_once(monkeypatch, path, '"type":"header"', point, error)
    with pytest.raises((LedgerError, KeyboardInterrupt)):
        RunLedger(path, run_id=RUN_ID, clock=Clock())
    assert not path.exists()
    monkeypatch.undo()
    ledger = RunLedger(path, run_id=RUN_ID, clock=Clock())
    ledger.admit("a")
    assert RunLedger(path, run_id=RUN_ID, clock=Clock()).aborted_attempts == 1


def test_a_stop_after_the_whole_header_keeps_it(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    fault_once(monkeypatch, path, '"type":"header"', "after", KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        RunLedger(path, run_id=RUN_ID, clock=Clock())
    monkeypatch.undo()
    assert path.read_bytes().count(b"\n") == 1
    RunLedger(path, run_id=RUN_ID, clock=Clock()).admit("a")


def executor(tmp_path, workers):
    ledger = RunLedger(tmp_path / "ledger.jsonl", run_id=RUN_ID, clock=Clock(), caps=CAPS)
    return EpisodeExecutor(model=SeededModel(), env_factory=GameFactory(), ledger=ledger,
                           trajectories_dir=tmp_path / "trajectories", run_id=RUN_ID,
                           code_commit=COMMIT, workers=workers,
                           sleep_between_retries=False), ledger


@pytest.mark.parametrize("workers", [1, 2])
def test_a_failed_episode_record_leaves_a_ledger_that_reloads(tmp_path, registry,  # noqa: F811
                                                              monkeypatch, workers):
    """The parallel path goes on recording the episodes still in flight
    after the failure: those records are refused, not glued on."""
    specs = lifetime_specs(tasks(), g0_runtime_config(registry), MINING_SEEDS[0])
    path = tmp_path / "ledger.jsonl"
    run, ledger = executor(tmp_path, workers)
    fail_once(monkeypatch, path, '"type":"episode"')
    try:
        with pytest.raises(LedgerError, match="No space left"):
            run.run_many(specs)
        with pytest.raises(LedgerError, match="earlier write in this process failed"):
            run.run(specs[-1])
    finally:
        run.close()
    monkeypatch.undo()
    assert path.read_bytes().endswith(b"\n")
    reloaded = RunLedger(path, run_id=RUN_ID, clock=Clock(), caps=CAPS)
    assert reloaded.episodes == 0 and reloaded.aborted_attempts == workers
    resumed, again = executor(tmp_path, workers)
    try:
        assert len(resumed.run_many(specs)) == len(specs)
    finally:
        resumed.close()
    assert again.episodes == len(specs)
