"""The EXP-0001 run ledger: hard caps, lifecycle and resume (issue #13;
design section 8 "hard rule"; ADR-0007; ADR-0008).

An append-only JSONL file. Every finished episode is one record with its
key and its model-call count; the lifecycle moves only forward
(``running`` -> ``complete`` or ``incomplete``). Before an episode starts
the ledger admits it only if no hard cap would be exceeded; otherwise the
run is stopped and marked incomplete with its reason, and nothing can be
admitted afterwards. A resumed run reads the ledger back and skips every
key it already holds, so no recorded episode runs twice.

Every admission first appends an ``attempt`` record with the calls it
reserves (an episode's worst case, or the miner's two calls per pair). An
attempt that a crash left without its finishing record is charged its
whole reservation when the ledger is read back, so calls made before a
crash still count against the caps (conservatively), and the number of
aborted attempts stays on record.

Episodes may run several at once (ADR-0008). An admitted attempt without
its finishing record is in flight, and every cap check counts the worst
case of every attempt in flight, so parallel episodes never pass a cap
that the same episodes run one at a time would respect.

The wall-clock cap counts active run time only (ADR-0008, owner decision).
Every process that opens a running ledger starts a ``session`` record, and
every later record carries its clock reading. A session lasts from its
start to its last record; the run's active time is the sum over sessions.
Time between sessions (a disconnected notebook, say) is not counted, and
neither is the time between a crashed session's last record and the
crash: that gap is at most the time between two records.

Before each gate round the pipeline asks for a ``reserve`` check (ADR-0012
D9): the round's unrecorded episodes plus the final evaluation's largest
count must fit under the episode cap, and, at the run's mean calls per
recorded episode, under the call cap; otherwise the run stops before the
round, marked incomplete. Each check is a record with its numbers.

Notes (``repair``, ``restore``) record a torn tail cut off on purpose or a
restore from a mirror snapshot. They change no count, carry no clock, and
may follow a terminal record (a finished run can be restored). A
``heartbeat`` record carries only a clock: it keeps a session's span up to
date through a long stretch without episodes (the miner's teacher calls)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path

from evaluation.exp0001.protocol import CAPS, max_model_calls_per_episode
from trajectory.storage.jsonl import finite_float, jsonl_lines

LEDGER_SCHEMA = "heritagent/exp0001-ledger/0.3"  # 0.3 adds the reserve record (ADR-0012 D9)
READABLE_SCHEMAS = ("heritagent/exp0001-ledger/0.2", LEDGER_SCHEMA)
WALL_CLOCK_BASIS = "active-run-time"  # ADR-0008
TERMINAL = ("complete", "incomplete")
NOTES = ("repair", "restore")
EPISODE, CALLS = "episode", "calls"
HEARTBEAT = "heartbeat"
RESERVE = "reserve"


class LedgerError(ValueError):
    """The ledger is malformed, or an operation does not fit its state."""


class CapExceeded(Exception):
    """A hard cap would be exceeded: the run is now incomplete."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class AdmissionDeferred(Exception):
    """A cap would block this admission while other attempts are in flight:
    nothing was written; the caller finishes them and asks again."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _clock_of(record: Mapping, path: Path) -> float:
    clock = record.get("clock")
    if isinstance(clock, bool) or not isinstance(clock, (int, float)) or not math.isfinite(clock):
        raise LedgerError(f"{path}: a {record.get('type')} record has no valid clock")
    return float(clock)


class RunLedger:
    def __init__(self, path: Path, *, run_id: str, clock: Callable[[], float],
                 caps: Mapping = CAPS, session_info: Mapping | None = None,
                 on_append: Callable[[dict], None] | None = None, readonly: bool = False):
        """``session_info`` is recorded in this process's session record
        (for example the accelerator); ``on_append`` is called after every
        appended record, when no other run file is half-written (the run
        mirror takes its snapshots there). A ``readonly`` ledger reads the
        file and never writes, so it opens no session."""
        self._path = Path(path)
        self._clock = clock
        self._caps = dict(caps)
        self._on_append = on_append
        self._readonly = readonly
        self._done: dict[str, dict] = {}
        self._charged: dict[str, dict] = {}  # model calls outside episodes (the miner's)
        self._aborted: list[dict] = []  # attempts a crash left unfinished
        # attempts admitted by this process and not yet finished: key -> (kind, reserved)
        self._open: dict[str, tuple[str, int]] = {}
        self._model_calls = 0
        self._state = "running"
        self._reason: str | None = None
        self._sessions: list[dict] = []  # index, start, last, info
        self._notes: list[dict] = []
        self._reserve_checks: list[dict] = []
        self._session_start: float | None = None  # this process's session
        if self._path.exists():
            self._load(run_id)
        elif readonly:
            raise LedgerError(f"{self._path} does not exist")
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._write({"type": "header", "schema": LEDGER_SCHEMA, "run_id": run_id,
                         "started_clock": float(clock()), "caps": self._caps,
                         "wall_clock_basis": WALL_CLOCK_BASIS})
        if self._state == "running" and not readonly:
            now = float(clock())
            self._session_start = now
            self._sessions.append({"index": len(self._sessions) + 1, "start": now, "last": now,
                                   "info": dict(session_info or {})})
            self._write({"type": "session", "index": len(self._sessions), "clock": now,
                         "info": dict(session_info or {})})

    def _load(self, run_id: str) -> None:
        data = self._path.read_bytes()
        if data and not data.endswith(b"\n"):
            # a torn last write: appending would glue the next record onto it
            raise LedgerError(f"{self._path} does not end with a newline (torn write); "
                              f"it is never silently salvaged (repair_torn_tail cuts it off "
                              f"on purpose and keeps it on record)")
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
                or records[0].get("schema") not in READABLE_SCHEMAS:
            raise LedgerError(f"{self._path}: missing ledger header")
        header = records[0]
        if header.get("run_id") != run_id:
            raise LedgerError(f"{self._path} belongs to run {header.get('run_id')!r}, not {run_id!r}")
        if header.get("caps") != self._caps:
            raise LedgerError(f"{self._path} was opened with other caps {header.get('caps')}")
        if header.get("wall_clock_basis") != WALL_CLOCK_BASIS:
            raise LedgerError(f"{self._path}: wall clock basis {header.get('wall_clock_basis')!r}")
        open_attempts: dict[str, list[dict]] = {}
        for record in records[1:]:
            kind = record.get("type")
            if kind in NOTES:  # a restore or a repair may follow the terminal record
                self._notes.append(record)
                continue
            if self._state in TERMINAL:
                raise LedgerError(f"{self._path}: a record follows the terminal state")
            if kind == "session":
                if record.get("index") != len(self._sessions) + 1:
                    raise LedgerError(f"{self._path}: session {record.get('index')!r} is out of "
                                      f"order")
                start = _clock_of(record, self._path)
                self._sessions.append({"index": record["index"], "start": start, "last": start,
                                       "info": record.get("info") or {}})
                continue
            if kind == "attempt":
                key, reserved = record.get("key"), record.get("reserved")
                if not isinstance(key, str) or isinstance(reserved, bool) \
                        or not isinstance(reserved, int) or reserved < 0:
                    raise LedgerError(f"{self._path}: bad attempt record {record!r}")
                if key in self._done or key in self._charged:
                    raise LedgerError(f"{self._path}: attempt after {key!r} was recorded")
                open_attempts.setdefault(key, []).append(record)
            elif kind in (EPISODE, CALLS):
                key = record.get("key")
                if not isinstance(key, str) or key in self._done or key in self._charged:
                    raise LedgerError(f"{self._path}: bad or duplicate {kind} key {key!r}")
                calls = record.get("model_calls")
                if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
                    raise LedgerError(f"{self._path}: bad model_calls for {key!r}")
                (self._done if kind == EPISODE else self._charged)[key] = record
                self._model_calls += calls
                if open_attempts.get(key):
                    open_attempts[key].pop()  # the attempt this record finished
            elif kind in TERMINAL:
                self._state, self._reason = kind, record.get("reason")
            elif kind == RESERVE:
                # as this session's checks hold them: the clock is the session's, not the check's
                self._reserve_checks.append({k: v for k, v in record.items() if k != "clock"})
            elif kind != HEARTBEAT:
                raise LedgerError(f"{self._path}: unknown record type {kind!r}")
            if not self._sessions:
                raise LedgerError(f"{self._path}: a {kind} record outside a session")
            session = self._sessions[-1]
            session["last"] = max(session["last"], _clock_of(record, self._path))
        for attempts in open_attempts.values():
            for attempt in attempts:
                self._aborted.append(attempt)
                self._model_calls += attempt["reserved"]

    def _write(self, record: dict) -> None:
        if self._readonly:
            raise LedgerError(f"{self._path} is open read-only")
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        if self._on_append is not None:
            self._on_append(record)

    def _append(self, record: dict) -> None:
        """A run record of this process's session, stamped with its clock."""
        if self._session_start is None:
            raise LedgerError(f"{self._path}: no session is open")
        now = float(self._clock())
        self._sessions[-1]["last"] = max(self._sessions[-1]["last"], now)
        self._write({**record, "clock": now})

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

    @property
    def in_flight(self) -> int:
        """Attempts this process admitted that have no finishing record."""
        return len(self._open)

    @property
    def sessions(self) -> tuple[dict, ...]:
        return tuple(dict(s) for s in self._sessions)

    @property
    def notes(self) -> tuple[dict, ...]:
        return tuple(dict(n) for n in self._notes)

    @property
    def reserve_checks(self) -> tuple[dict, ...]:
        return tuple(dict(r) for r in self._reserve_checks)

    def mean_calls_per_episode(self) -> Fraction | None:
        """The run's mean model calls per recorded episode so far (episode
        records only: teacher calls and aborted reservations are not
        episodes), or None before the first episode."""
        if not self._done:
            return None
        return Fraction(sum(r["model_calls"] for r in self._done.values()), len(self._done))

    def check_reserve(self, label: str, *, round_episodes: int, final_reserve: int) -> dict:
        """ADR-0012 D9's pre-round check, with nothing in flight: room for the
        round's unrecorded episodes plus the final evaluation's largest count,
        under the episode cap and, at the run's mean calls per recorded
        episode, under the call cap. Records the check; stops the run as
        incomplete (CapExceeded) when it fails."""
        if self._state != "running":
            raise LedgerError(f"the run is {self._state}; nothing can start")
        if self._open:
            raise LedgerError("a reserve check needs no episode in flight")
        for name, value in (("round_episodes", round_episodes), ("final_reserve", final_reserve)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LedgerError(f"{name} must be a non-negative int, got {value!r}")
        ahead = round_episodes + final_reserve
        mean = self.mean_calls_per_episode()
        # before any episode the worst case stands in for the mean
        per_episode = Fraction(max_model_calls_per_episode()) if mean is None else mean
        calls_ahead = math.ceil(per_episode * ahead)
        episodes_ok = self.episodes + ahead <= self._caps["episodes_max"]
        calls_ok = self._model_calls + calls_ahead <= self._caps["model_calls_max"]
        record = {"type": RESERVE, "label": label, "round_episodes": round_episodes,
                  "final_reserve": final_reserve, "episodes": self.episodes,
                  "model_calls": self._model_calls,
                  "mean_calls_per_episode": None if mean is None else {
                      "exact": f"{mean.numerator}/{mean.denominator}", "float": float(mean)},
                  "calls_ahead": calls_ahead, "passed": episodes_ok and calls_ok}
        self._append(record)
        self._reserve_checks.append(record)
        if not episodes_ok:
            self._stop(f"reserve: before {label}, {self.episodes} episodes + {round_episodes} "
                       f"in the round + {final_reserve} for the final evaluation would exceed "
                       f"episodes_max {self._caps['episodes_max']} (ADR-0012 D9)")
        if not calls_ok:
            self._stop(f"reserve: before {label}, {self._model_calls} model calls + "
                       f"{calls_ahead} ahead ({ahead} episodes at {float(per_episode):.2f} "
                       f"calls each) would exceed model_calls_max "
                       f"{self._caps['model_calls_max']} (ADR-0012 D9)")
        return dict(record)

    def active_hours(self) -> float:
        """Active run time so far: every session's span, this one up to now."""
        seconds = 0.0
        for session in self._sessions:
            end = session["last"]
            if self._session_start is not None and session is self._sessions[-1]:
                end = max(end, float(self._clock()))
            seconds += max(0.0, end - session["start"])
        return seconds / 3600

    def summary(self) -> dict:
        return {"state": self._state, "reason": self._reason, "episodes": self.episodes,
                "model_calls": self._model_calls, "aborted_attempts": len(self._aborted),
                "in_flight": len(self._open), "active_hours": round(self.active_hours(), 4),
                "sessions": len(self._sessions), "notes": [n.get("type") for n in self._notes],
                "caps": dict(self._caps)}

    def is_done(self, key: str) -> bool:
        return key in self._done

    def done_keys(self, prefix: str = "") -> tuple[str, ...]:
        """The recorded episode keys that start with ``prefix``."""
        return tuple(key for key in self._done if key.startswith(prefix))

    def heartbeat(self) -> None:
        """Stamp this session's clock (no count changes)."""
        if self._state == "running":
            self._append({"type": HEARTBEAT})

    def done_record(self, key: str) -> dict:
        return dict(self._done[key])

    def _terminal_counts(self) -> dict:
        return {"episodes": self.episodes, "model_calls": self._model_calls,
                "aborted_attempts": len(self._aborted)}

    def _stop(self, reason: str) -> None:
        self._state, self._reason = "incomplete", reason
        self._append({"type": "incomplete", "reason": reason, **self._terminal_counts()})
        raise CapExceeded(reason)

    def abandon(self, reason: str) -> None:
        """Stop the run as incomplete for a protocol rule that fails closed
        (not a cap): nothing can be admitted afterwards."""
        if self._state != "running":
            raise LedgerError(f"the run is already {self._state}")
        if not isinstance(reason, str) or not reason.strip():
            raise LedgerError("an incomplete run records its reason")
        self._state, self._reason = "incomplete", reason
        self._append({"type": "incomplete", "reason": reason, **self._terminal_counts()})

    def _cap_reason(self, key: str, worst: int, *, episode: bool) -> str | None:
        """Why admitting ``key`` now would exceed a cap, or None. Re-admitting
        a key aborts its own open attempt, which moves that reservation from
        in flight into the charged count: the sum checked is the same."""
        in_flight_episodes = sum(1 for k, (kind, _) in self._open.items()
                                 if kind == EPISODE and k != key)
        if episode and self.episodes + in_flight_episodes + 1 > self._caps["episodes_max"]:
            return f"hard cap: episodes_max {self._caps['episodes_max']} would be exceeded"
        reserved = sum(r for _, r in self._open.values())
        if self._model_calls + reserved + worst > self._caps["model_calls_max"]:
            return (f"hard cap: model_calls_max {self._caps['model_calls_max']} would be exceeded "
                    f"(used {self._model_calls}, reserved in flight {reserved}, the next step may "
                    f"use {worst})")
        hours = self.active_hours()
        if hours > self._caps["wall_clock_hours_max"]:
            return (f"hard cap: wall_clock_hours_max {self._caps['wall_clock_hours_max']} "
                    f"exceeded ({hours:.2f} active run hours)")
        return None

    def admission_blocked(self, key: str) -> bool:
        """True if ``admit(key)`` would stop the run on a cap right now. It
        changes nothing: a caller with episodes in flight finishes them
        first, so the stop never cuts an admitted episode short."""
        if self._state != "running" or key in self._done:
            return False
        return self._cap_reason(key, max_model_calls_per_episode(), episode=True) is not None

    def admit(self, key: str, *, defer: bool = False) -> bool:
        """True if the episode should run now, False if the ledger already
        holds it (resume). Raises CapExceeded and marks the run incomplete if
        it would exceed a hard cap; with ``defer`` (the caller has episodes in
        flight) it raises AdmissionDeferred instead and writes nothing, so a
        stop never cuts an admitted episode short."""
        if self._state != "running":
            raise LedgerError(f"the run is {self._state}; no episode can start")
        if key in self._done:
            return False
        self._abort_open(key)  # charged before every cap check, as a reload would
        worst = max_model_calls_per_episode()
        reason = self._cap_reason(key, worst, episode=True)  # one clock reading decides
        if reason is not None:
            if defer:
                raise AdmissionDeferred(reason)
            self._stop(reason)
        self._append({"type": "attempt", "key": key, "reserved": worst})
        self._open[key] = (EPISODE, worst)
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
        reason = self._cap_reason(key, worst_case, episode=False)
        if reason is not None:
            self._stop(reason)
        self._append({"type": "attempt", "key": key, "reserved": worst_case})
        self._open[key] = (CALLS, worst_case)
        return True

    def abort(self, key: str) -> None:
        """An admitted attempt failed without its finishing record (the
        episode raised): it is no longer in flight, and it is charged its
        reservation now, as a reload would charge it. Nothing is written."""
        self._abort_open(key)

    def _abort_open(self, key: str) -> None:
        """Admitting a key again in the same process: its earlier attempt
        failed without a record, so it is aborted and charged now, as a
        reload would charge it."""
        entry = self._open.pop(key, None)
        if entry is not None:
            self._aborted.append({"type": "attempt", "key": key, "reserved": entry[1]})
            self._model_calls += entry[1]

    def record_calls(self, key: str, *, model_calls: int) -> None:
        if self._state != "running":
            raise LedgerError(f"the run is {self._state}; nothing can be recorded")
        if key in self._charged or key in self._done:
            raise LedgerError(f"{key!r} is already recorded")
        if isinstance(model_calls, bool) or not isinstance(model_calls, int) or model_calls < 0:
            raise LedgerError(f"model_calls must be a non-negative int, got {model_calls!r}")
        record = {"type": CALLS, "key": key, "model_calls": model_calls}
        self._append(record)
        self._charged[key] = record
        self._model_calls += model_calls
        self._finish(key)

    def is_charged(self, key: str) -> bool:
        return key in self._charged

    def record(self, key: str, *, model_calls: int, outcome: Mapping) -> None:
        if self._state != "running":
            raise LedgerError(f"the run is {self._state}; nothing can be recorded")
        if key in self._done or key in self._charged:
            raise LedgerError(f"episode {key!r} is already recorded")
        if isinstance(model_calls, bool) or not isinstance(model_calls, int) or model_calls < 0:
            raise LedgerError(f"model_calls must be a non-negative int, got {model_calls!r}")
        record = {"type": EPISODE, "key": key, "model_calls": model_calls,
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
        self._append({"type": "complete", **self._terminal_counts()})


def append_note(path: Path, note: Mapping) -> dict:
    """Append a ``repair`` or ``restore`` note to a ledger that ends with a
    complete line. Notes change no count; they keep the event on record."""
    path = Path(path)
    note = dict(note)
    if note.get("type") not in NOTES:
        raise LedgerError(f"a note is one of {NOTES}, got {note.get('type')!r}")
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise LedgerError(f"{path} does not end with a complete record")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(note, sort_keys=True, separators=(",", ":"), allow_nan=False)
                     + "\n")
        handle.flush()
    return note


def repair_torn_tail(path: Path) -> dict | None:
    """Cut off a torn last line (a write a crash interrupted), keep the cut
    bytes next to the ledger, and record the cut as a ``repair`` note.
    Returns None if the ledger ends with a complete line. The record the
    torn line held is lost: an attempt it finished stays open and is
    charged its reservation, as for any crash."""
    path = Path(path)
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    if cut == 0:
        raise LedgerError(f"{path}: the header itself is torn; nothing can be kept")
    tail = data[cut:]
    digest = hashlib.sha256(tail).hexdigest()
    saved = path.with_name(f"{path.name}.torn-{digest[:12]}")
    if not saved.exists():
        saved.write_bytes(tail)
    with path.open("r+b") as handle:
        handle.truncate(cut)
    return append_note(path, {"type": "repair", "removed_bytes": len(tail),
                              "removed_sha256": digest, "saved_as": saved.name})
