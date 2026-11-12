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
    with pytest.raises(OSError, match="No space left"):
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
    with pytest.raises(OSError, match="No space left"):
        ledger.record("a", model_calls=1, outcome={})
    with pytest.raises(LedgerError, match="earlier write in this process failed"):
        ledger.record("b", model_calls=1, outcome={})
    monkeypatch.undo()
    assert not path.read_bytes().endswith(b"\n")
    assert repair_torn_tail(path)["type"] == "repair"
    reloaded = RunLedger(path, run_id=RUN_ID, clock=Clock())
    assert (reloaded.episodes, reloaded.aborted_attempts) == (0, 2)


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
        with pytest.raises(OSError, match="No space left"):
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
