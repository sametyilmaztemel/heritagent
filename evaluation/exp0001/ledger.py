"""The EXP-0001 run ledger: hard caps, lifecycle and resume (issue #13
Phase A; design section 8 "hard rule"; ADR-0007).

An append-only JSONL file. Every finished episode is one record with its
key and its model-call count; the lifecycle moves only forward
(``running`` -> ``complete`` or ``incomplete``). Before an episode starts
the ledger admits it only if no hard cap would be exceeded; otherwise the
run is stopped and marked incomplete with its reason, and nothing can be
admitted afterwards. A resumed run reads the ledger back and skips every
key it already holds, so no recorded episode runs twice.

Every admission first appends an ``attempt`` record with the calls it
reserves (an episode's worst case, or the miner's one call per pair). An
attempt that a crash left without its finishing record is charged its
whole reservation when the ledger is read back, so calls made before a
crash still count against the caps (conservatively), and the number of
aborted attempts stays on record."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from evaluation.exp0001.protocol import CAPS, max_model_calls_per_episode
from trajectory.storage.jsonl import finite_float, jsonl_lines

LEDGER_SCHEMA = "heritagent/exp0001-ledger/0.1"
TERMINAL = ("complete", "incomplete")


class LedgerError(ValueError):
    """The ledger is malformed, or an operation does not fit its state."""


class CapExceeded(Exception):
    """A hard cap would be exceeded: the run is now incomplete."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class RunLedger:
    def __init__(self, path: Path, *, run_id: str, clock: Callable[[], float],
                 caps: Mapping = CAPS):
        self._path = Path(path)
        self._clock = clock
        self._caps = dict(caps)
        self._done: dict[str, dict] = {}
        self._charged: dict[str, dict] = {}  # model calls outside episodes (the miner's)
        self._aborted: list[dict] = []  # attempts a crash left unfinished
        self._open: dict[str, int] = {}  # attempts admitted by this process: key -> reserved
        self._model_calls = 0
        self._state = "running"
        self._reason: str | None = None
        if self._path.exists():
            self._load(run_id)
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._started = float(clock())
            self._append({"type": "header", "schema": LEDGER_SCHEMA, "run_id": run_id,
                          "started_clock": self._started, "caps": self._caps})

    def _load(self, run_id: str) -> None:
        data = self._path.read_bytes()
        if data and not data.endswith(b"\n"):
            # a torn last write: appending would glue the next record onto it
            raise LedgerError(f"{self._path} does not end with a newline (torn write); "
                              f"it is never silently salvaged")
        records = []
        for index, line in enumerate(jsonl_lines(self._path)):
            try:
                record = json.loads(line.decode("utf-8"), parse_float=finite_float)
            except ValueError as exc:  # a torn or corrupt line is never salvaged
                raise LedgerError(f"{self._path}: line {index + 1} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise LedgerError(f"{self._path}: line {index + 1} is not an object")
            records.append(record)
        if not records or records[0].get("type") != "header" \
                or records[0].get("schema") != LEDGER_SCHEMA:
            raise LedgerError(f"{self._path}: missing ledger header")
        header = records[0]
        if header.get("run_id") != run_id:
            raise LedgerError(f"{self._path} belongs to run {header.get('run_id')!r}, not {run_id!r}")
        if header.get("caps") != self._caps:
            raise LedgerError(f"{self._path} was opened with other caps {header.get('caps')}")
        self._started = float(header["started_clock"])
        open_attempts: dict[str, list[dict]] = {}
        for record in records[1:]:
            kind = record.get("type")
            if self._state in TERMINAL:
                raise LedgerError(f"{self._path}: a record follows the terminal state")
            if kind == "attempt":
                key, reserved = record.get("key"), record.get("reserved")
                if not isinstance(key, str) or isinstance(reserved, bool) \
                        or not isinstance(reserved, int) or reserved < 0:
                    raise LedgerError(f"{self._path}: bad attempt record {record!r}")
                if key in self._done or key in self._charged:
                    raise LedgerError(f"{self._path}: attempt after {key!r} was recorded")
                open_attempts.setdefault(key, []).append(record)
                continue
            if kind in ("episode", "calls"):
                key = record.get("key")
                if not isinstance(key, str) or key in self._done or key in self._charged:
                    raise LedgerError(f"{self._path}: bad or duplicate {kind} key {key!r}")
                calls = record.get("model_calls")
                if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
                    raise LedgerError(f"{self._path}: bad model_calls for {key!r}")
                (self._done if kind == "episode" else self._charged)[key] = record
                self._model_calls += calls
                if open_attempts.get(key):
                    open_attempts[key].pop()  # the attempt this record finished
            elif kind in TERMINAL:
                self._state, self._reason = kind, record.get("reason")
            else:
                raise LedgerError(f"{self._path}: unknown record type {kind!r}")
        for attempts in open_attempts.values():
            for attempt in attempts:
                self._aborted.append(attempt)
                self._model_calls += attempt["reserved"]

    def _append(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    @property
    def state(self) -> str:
        return self._state

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def episodes(self) -> int:
        return len(self._done)

    @property
    def model_calls(self) -> int:
        """Recorded calls, plus the reservation of every aborted attempt."""
        return self._model_calls

    @property
    def aborted_attempts(self) -> int:
        return len(self._aborted)

    def is_done(self, key: str) -> bool:
        return key in self._done

    def done_record(self, key: str) -> dict:
        return dict(self._done[key])

    def _stop(self, reason: str) -> None:
        self._state, self._reason = "incomplete", reason
        self._append({"type": "incomplete", "reason": reason, "episodes": self.episodes,
                      "model_calls": self._model_calls, "aborted_attempts": len(self._aborted)})
        raise CapExceeded(reason)

    def abandon(self, reason: str) -> None:
        """Stop the run as incomplete for a protocol rule that fails closed
        (not a cap): nothing can be admitted afterwards."""
        if self._state != "running":
            raise LedgerError(f"the run is already {self._state}")
        if not isinstance(reason, str) or not reason.strip():
            raise LedgerError("an incomplete run records its reason")
        self._state, self._reason = "incomplete", reason
        self._append({"type": "incomplete", "reason": reason, "episodes": self.episodes,
                      "model_calls": self._model_calls, "aborted_attempts": len(self._aborted)})

    def admit(self, key: str) -> bool:
        """True if the episode should run now, False if the ledger already
        holds it (resume). Raises CapExceeded and marks the run incomplete if
        it would exceed a hard cap."""
        if self._state != "running":
            raise LedgerError(f"the run is {self._state}; no episode can start")
        if key in self._done:
            return False
        if self.episodes + 1 > self._caps["episodes_max"]:
            self._stop(f"hard cap: episodes_max {self._caps['episodes_max']} would be exceeded")
        worst = max_model_calls_per_episode()
        self._abort_open(key)  # charged before the cap check, as a reload would
        self._check_calls(worst)
        self._append({"type": "attempt", "key": key, "reserved": worst})
        self._open[key] = worst
        return True

    def admit_calls(self, key: str, worst_case: int) -> bool:
        """Admit model calls made outside an episode (the miner's teacher
        calls): False if already charged, CapExceeded if they could exceed
        a cap."""
        if self._state != "running":
            raise LedgerError(f"the run is {self._state}; nothing can start")
        if key in self._charged:
            return False
        if isinstance(worst_case, bool) or not isinstance(worst_case, int) or worst_case < 0:
            raise LedgerError(f"worst_case must be a non-negative int, got {worst_case!r}")
        self._abort_open(key)  # charged before the cap check, as a reload would
        self._check_calls(worst_case)
        self._append({"type": "attempt", "key": key, "reserved": worst_case})
        self._open[key] = worst_case
        return True

    def _abort_open(self, key: str) -> None:
        """Admitting a key again in the same process: its earlier attempt
        failed without a record, so it is aborted and charged now, as a
        reload would charge it."""
        reserved = self._open.pop(key, None)
        if reserved is not None:
            self._aborted.append({"type": "attempt", "key": key, "reserved": reserved})
            self._model_calls += reserved

    def record_calls(self, key: str, *, model_calls: int) -> None:
        if self._state != "running":
            raise LedgerError(f"the run is {self._state}; nothing can be recorded")
        if key in self._charged or key in self._done:
            raise LedgerError(f"{key!r} is already recorded")
        if isinstance(model_calls, bool) or not isinstance(model_calls, int) or model_calls < 0:
            raise LedgerError(f"model_calls must be a non-negative int, got {model_calls!r}")
        record = {"type": "calls", "key": key, "model_calls": model_calls}
        self._append(record)
        self._charged[key] = record
        self._model_calls += model_calls
        self._finish(key)

    def is_charged(self, key: str) -> bool:
        return key in self._charged

    def _check_calls(self, worst: int) -> None:
        if self._model_calls + worst > self._caps["model_calls_max"]:
            self._stop(f"hard cap: model_calls_max {self._caps['model_calls_max']} would be "
                       f"exceeded (used {self._model_calls}, the next step may use {worst})")
        elapsed_hours = (float(self._clock()) - self._started) / 3600
        if elapsed_hours > self._caps["wall_clock_hours_max"]:
            self._stop(f"hard cap: wall_clock_hours_max {self._caps['wall_clock_hours_max']} "
                       f"exceeded ({elapsed_hours:.2f} h)")

    def record(self, key: str, *, model_calls: int, outcome: Mapping) -> None:
        if self._state != "running":
            raise LedgerError(f"the run is {self._state}; nothing can be recorded")
        if key in self._done or key in self._charged:
            raise LedgerError(f"episode {key!r} is already recorded")
        if isinstance(model_calls, bool) or not isinstance(model_calls, int) or model_calls < 0:
            raise LedgerError(f"model_calls must be a non-negative int, got {model_calls!r}")
        record = {"type": "episode", "key": key, "model_calls": model_calls,
                  "outcome": dict(outcome)}
        self._append(record)
        self._done[key] = record
        self._model_calls += model_calls
        self._finish(key)

    def _finish(self, key: str) -> None:
        """A finishing record closes the latest open attempt of its key, as a
        reload reads it: this process's own attempt, else the latest aborted
        one, whose reservation is then no longer charged."""
        if self._open.pop(key, None) is not None:
            return
        for index in range(len(self._aborted) - 1, -1, -1):
            if self._aborted[index]["key"] == key:
                self._model_calls -= self._aborted.pop(index)["reserved"]
                return

    def complete(self) -> None:
        if self._state != "running":
            raise LedgerError(f"the run is already {self._state}")
        self._state = "complete"
        self._append({"type": "complete", "episodes": self.episodes,
                      "model_calls": self._model_calls, "aborted_attempts": len(self._aborted)})
