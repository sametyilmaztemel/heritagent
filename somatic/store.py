"""Persistent somatic trait store (issue #10).

The somatic-memory lifecycle boundary used later by CIG (#11/#12):

    absent --add_candidate--> candidate --decide--> validated | rejected (terminal)

Hard guarantees:
- logical identity is ``(gene_id, version)``; the artifact URI, provenance
  and gene ref are immutable for that identity — a lifecycle transition may
  change ONLY the validation envelope;
- terminal decisions are immutable (no flips, no rollback, no appends);
- rejected traits are retained permanently (somatic memory, never silently
  discarded);
- ``validated`` still carries ``origin="acquired"`` — assimilation into a
  germline genome is NOT this store's job (#12/#13 create the separate
  germline ref with origin="assimilation");
- persistence is an append-only hash-chained journal (see somatic/journal.py).
  The JOURNAL FILE is the source of truth: ``verify()`` re-reads it from
  disk, and every persistent append first checks that the on-disk record
  count + tail digest still match this handle's expected state (stale-writer
  protection). Replay uses the SAME centralized transition validation as the
  live write path — a semantically invalid record is rejected on replay even
  when its hashes were recomputed.

Explicit create/open semantics (issue #8 lesson): ``SomaticStore.create``
uses atomic exclusive file creation; ``SomaticStore.open`` replays and fully
verifies an existing journal before it becomes append-capable; neither path
truncates or overwrites somatic memory. ``SomaticStore(path=None)`` is the
in-memory variant (same lifecycle, no persistence).

Close lifecycle: ``close()`` is idempotent; after close, mutation APIs
(``add_candidate``/``decide``) raise a typed error while read APIs
(``get``/``state``/``all``/``by_state``/``history``/``candidate_ref``/
``verify``/``verify_strict``) remain usable — documented and tested.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from genome.validation.errors import RecordConsistencyError
from genome.validation.loader import load_somatic
from genome.validation.registry import TraitRegistry
from trajectory.recorder.errors import RecorderError
from trajectory.storage.canonical import canonical_json
from somatic.journal import (
    JournalAppender,
    build_record,
    compute_record_digest,
    journal_record_validator,
    read_journal_records,
)

TERMINAL_STATES = ("validated", "rejected")
_VERDICTS = ("validated", "rejected")


@dataclass(frozen=True)
class SomaticEntry:
    """Read-only view over one stored trait (deep copies handed out)."""

    gene_id: str
    version: int
    envelope: dict
    history: tuple[dict, ...]


class SomaticStore:
    """Persistent append-only somatic-memory store.

    ``SomaticStore(path=None)`` → in-memory (compat mode);
    ``SomaticStore.create(path, registry)`` → new journal, atomic exclusive;
    ``SomaticStore.open(path, registry)`` → replay + verify an existing
    journal before it becomes append-capable."""

    def __init__(self, path: Path | None = None, *, registry: TraitRegistry | None = None,
                 mode: str = "create"):
        if mode not in ("create", "open"):
            raise RecorderError(f"unknown store mode {mode!r}; use create|open")
        self.path = Path(path) if path is not None else None
        self._registry = registry
        self._envelopes: dict[tuple[str, int], dict] = {}
        self._history: dict[tuple[str, int], list[dict]] = {}
        self._records: list[dict] = []
        self._writer = None
        self._closed = True  # flipped after successful setup

        if self.path is None:
            if mode == "open":
                raise RecorderError("cannot open an in-memory store; use create")
            self._closed = False
            self._expected_count = 0
            self._expected_tail = None
            return

        if mode == "create":
            # atomic exclusive creation (issue #8 lesson): never overwrite
            self._writer = JournalAppender(self.path, exclusive=True)
            self._closed = False
            self._expected_count = 0
            self._expected_tail = None
        else:
            self._reload_from_disk()  # full replay + verification, fail closed
            self._writer = JournalAppender(self.path, exclusive=False)
            self._closed = False
            self._expected_count = len(self._records)
            self._expected_tail = self._records[-1]["record_sha256"] \
                if self._records else None

    # -- explicit constructors ---------------------------------------------------
    @classmethod
    def create(cls, path: Path | None, registry: TraitRegistry | None = None) -> "SomaticStore":
        return cls(path, registry=registry, mode="create")

    @classmethod
    def open(cls, path: Path, registry: TraitRegistry | None = None) -> "SomaticStore":
        if path is None:
            raise RecorderError("open requires a journal path")
        return cls(path, registry=registry, mode="open")

    # -- centralized transition validation (live write AND replay) ----------------
    @staticmethod
    def _candidate_insert_issues(envelope: dict) -> list[str]:
        """Full candidate contract for `candidate_added` / add_candidate."""
        issues: list[str] = []
        state = envelope["validation"]["state"]
        origin = envelope["candidate"]["origin"]
        if state != "candidate":
            issues.append(f"candidate_added requires state 'candidate', got {state!r}")
        if origin != "acquired":
            issues.append(f"candidate_added requires origin 'acquired', got {origin!r}")
        return issues

    @staticmethod
    def _decision_transition_issues(current: dict, terminal: dict) -> list[str]:
        """Full decision contract for `decision_recorded` / decide: previous
        state must be candidate, resulting state must be a terminal verdict,
        gate_reports must be non-empty, and the candidate gene ref/artifact/
        provenance must be byte-identical to the stored candidate."""
        issues: list[str] = []
        if current["validation"]["state"] != "candidate":
            issues.append(
                f"decision requires previous state 'candidate', got "
                f"{current['validation']['state']!r}")
        resulting = terminal["validation"]["state"]
        if resulting not in _VERDICTS:
            issues.append(
                f"decision requires resulting state in {list(_VERDICTS)}, got {resulting!r}")
        if not terminal["validation"].get("gate_reports"):
            issues.append("decision requires non-empty gate_reports")
        if canonical_json(terminal["candidate"]) != canonical_json(current["candidate"]):
            issues.append("decision mutated the candidate gene ref/artifact/provenance")
        return issues

    # -- lifecycle: candidate ----------------------------------------------------
    def add_candidate(self, envelope: dict) -> dict:
        """Insert a NEW acquired candidate."""
        self._append_guard()
        stored = copy.deepcopy(envelope)
        issues = self._candidate_insert_issues(stored)
        if issues:
            raise RecordConsistencyError(sorted(issues))
        key = self._key(stored)
        if key in self._envelopes:
            raise RecordConsistencyError(
                f"somatic candidate {key[0]!r}@{key[1]} already exists; "
                f"no silent overwrite")
        # existing boundary: schema + registry binding (no parallel validator)
        load_somatic(copy.deepcopy(stored), self._registry)
        self._commit("candidate_added", stored, key=key)
        return copy.deepcopy(stored)

    # -- lifecycle: decision -------------------------------------------------------
    def decide(self, gene_id: str, version: int, verdict: str,
               gate_reports: list[str]) -> dict:
        """Record one terminal CIG decision for a candidate. Only the
        validation envelope changes — gene ref, artifact URI and provenance
        are carried over byte-identically from the stored candidate."""
        self._append_guard()
        if verdict not in _VERDICTS:
            raise RecordConsistencyError(
                f"verdict must be one of {list(_VERDICTS)}, got {verdict!r}")
        if not gate_reports:
            raise RecordConsistencyError(
                "terminal decisions require non-empty gate_reports (aggregate "
                "CIG ids; #11/#12 own the evidence)")
        if len(set(gate_reports)) != len(gate_reports):
            raise RecordConsistencyError(
                "gate_reports contains duplicates; treated as a set by contract")
        key = (gene_id, version)
        if key not in self._envelopes:
            raise RecordConsistencyError(f"unknown somatic trait {gene_id!r}@{version}")
        current = self._envelopes[key]
        terminal = copy.deepcopy(current)
        terminal["validation"] = {
            "state": verdict,
            "gate_reports": sorted(set(gate_reports)),
        }
        issues = self._decision_transition_issues(current, terminal)
        if issues:
            raise RecordConsistencyError(sorted(issues))
        load_somatic(copy.deepcopy(terminal), self._registry)  # schema re-validation
        self._commit("decision_recorded", terminal, key=key)
        return copy.deepcopy(terminal)

    # -- read API (defensive copies; usable after close) ----------------------------
    def get(self, gene_id: str, version: int) -> dict | None:
        entry = self._envelopes.get((gene_id, version))
        return copy.deepcopy(entry) if entry else None

    def state(self, gene_id: str, version: int) -> str | None:
        entry = self._envelopes.get((gene_id, version))
        return entry["validation"]["state"] if entry else None

    def all(self) -> dict[tuple[str, int], dict]:
        return {key: copy.deepcopy(env) for key, env in self._envelopes.items()}

    def by_state(self, state: str) -> dict[tuple[str, int], dict]:
        return {key: copy.deepcopy(env) for key, env in self._envelopes.items()
                if env["validation"]["state"] == state}

    def candidate_ref(self, gene_id: str, version: int) -> dict | None:
        entry = self._envelopes.get((gene_id, version))
        return copy.deepcopy(entry["candidate"]) if entry else None

    def history(self, gene_id: str, version: int) -> tuple[dict, ...]:
        return tuple(copy.deepcopy(env) for env in
                      self._history.get((gene_id, version), []))

    # -- integrity (journal file is the source of truth) ------------------------------
    def verify(self) -> list[str]:
        """Re-read the journal (from disk for persistent stores) and replay it
        with the same transition validation as the live write path; returns
        consistency issues (empty = sound). Read-safe after close."""
        records = read_journal_records(self.path) if self.path is not None \
            else self._records
        issues: list[str] = []
        if self.path is not None and len(records) != self._expected_count:
            issues.append(
                f"journal on disk has {len(records)} records but this handle's "
                f"expected state is {self._expected_count} (external append or "
                f"truncation)")
        saved = (self._envelopes, self._history)
        self._replay_records(records, self._registry, issues)
        self._envelopes, self._history = saved
        return issues

    def verify_strict(self) -> None:
        issues = self.verify()
        if issues:
            raise RecordConsistencyError(sorted(issues))

    def close(self) -> None:
        """Idempotent resource cleanup. After close, mutation APIs raise a
        typed error; read APIs remain usable."""
        if self._writer is not None:
            self._writer.close()
        self._closed = True

    # -- internals ---------------------------------------------------------------------
    @staticmethod
    def _key(envelope: dict) -> tuple[str, int]:
        candidate = envelope["candidate"]
        return candidate["gene_id"], candidate["version"]

    def _append_guard(self) -> None:
        """Fail closed on closed stores and on stale handles (another writer
        appended, or the journal was modified/truncated externally)."""
        if self._closed:
            raise RecorderError("somatic store is closed")
        if self.path is None:
            return
        disk = read_journal_records(self.path)
        disk_count = len(disk)
        disk_tail = disk[-1]["record_sha256"] if disk else None
        if disk_count != self._expected_count or disk_tail != self._expected_tail:
            raise RecorderError(
                f"stale somatic store handle for {str(self.path)!r}: journal on "
                f"disk has {disk_count} records (tail {str(disk_tail)[:16]}...) "
                f"but this handle expected {self._expected_count} (tail "
                f"{str(self._expected_tail)[:16]}...); another writer appended "
                f"or the journal was modified externally")

    def _commit(self, op: str, envelope: dict, *, key: tuple[str, int]) -> None:
        """Append one journal record (schema-checked) and apply the state
        transition. Replay applies transitions directly in _replay_records."""
        prev = self._records[-1]["record_sha256"] if self._records else None
        record = build_record(seq=len(self._records) + 1, op=op,
                               gene_id=key[0], version=key[1],
                               envelope=envelope, prev_sha256=prev)
        errors = [e.message for e in journal_record_validator().iter_errors(record)]
        if errors:
            raise RecordConsistencyError(sorted(errors))
        if self._writer is not None:  # in-memory stores keep records only
            self._writer.append(record)
        self._records.append(record)
        self._expected_count = len(self._records)
        self._expected_tail = record["record_sha256"]
        self._envelopes[key] = copy.deepcopy(envelope)
        self._history.setdefault(key, []).append(copy.deepcopy(envelope))

    def _reload_from_disk(self) -> None:
        self._records = read_journal_records(self.path)
        issues: list[str] = []
        self._envelopes, self._history = {}, {}
        self._replay_records(self._records, self._registry, issues)
        if issues:
            # a failed replay leaves the store unusable (fail closed)
            raise RecordConsistencyError(f"{self.path}: {sorted(issues)}")
        self._expected_count = len(self._records)
        self._expected_tail = self._records[-1]["record_sha256"] if self._records else None

    def _replay_records(self, records: list[dict], registry, issues_sink: list[str]) -> None:
        """Replay journal records through the SAME centralized transition
        validation as the live write path; rebuilds live state only when the
        replay was clean."""
        replay_envelopes: dict[tuple[str, int], dict] = {}
        replay_history: dict[tuple[str, int], list[dict]] = {}
        prev: str | None = None
        for index, record in enumerate(records):
            seq = record["seq"]
            if seq != index + 1:
                issues_sink.append(f"record {index + 1}: seq {seq!r} is not contiguous")
                continue
            errors = [e.message for e in journal_record_validator().iter_errors(record)]
            if errors:
                issues_sink.append(f"record {seq}: schema violations {sorted(errors)}")
                continue
            expected = compute_record_digest(
                seq=seq, op=record["op"], trait=record["trait"],
                envelope=record["envelope"], prev_sha256=record["prev_sha256"])
            if record["record_sha256"] != expected:
                issues_sink.append(f"record {seq}: record_sha256 mismatch "
                                    f"(record payload was edited)")
            if record["prev_sha256"] != prev:
                issues_sink.append(f"record {seq}: prev_sha256 chain broken "
                                    f"(expected {prev!r}, got {record['prev_sha256']!r})")
            prev = record["record_sha256"]

            key = (record["trait"]["gene_id"], record["trait"]["version"])
            envelope = record["envelope"]
            envelope_key = (envelope["candidate"]["gene_id"],
                             envelope["candidate"]["version"])
            if envelope_key != key:
                issues_sink.append(f"record {seq}: envelope identity {envelope_key} "
                                    f"disagrees with record trait {key}")
                continue
            op = record["op"]
            # centralized transition validation FIRST (same contract as the
            # live write path), then the existing schema/registry boundary
            if op == "candidate_added":
                transition_issues = self._candidate_insert_issues(envelope)
                if key in replay_envelopes:
                    transition_issues.append(
                        f"duplicate candidate insertion for {key}")
                if transition_issues:
                    issues_sink.extend(f"record {seq}: {issue}"
                                        for issue in transition_issues)
                    continue
            elif op == "decision_recorded":
                current = replay_envelopes.get(key)
                if current is None:
                    issues_sink.append(f"record {seq}: decision for unknown trait {key}")
                    continue
                transition_issues = self._decision_transition_issues(current, envelope)
                if transition_issues:
                    issues_sink.extend(f"record {seq}: {issue}"
                                        for issue in transition_issues)
                    continue
            else:
                issues_sink.append(f"record {seq}: unknown op {op!r}")
                continue

            try:
                load_somatic(copy.deepcopy(envelope), registry)
            except Exception as exc:  # any envelope violation is a replay issue
                issues_sink.append(f"record {seq}: envelope fails somatic validation: {exc}")
                continue

            # only reached when both transition and envelope validation passed
            replay_envelopes[key] = copy.deepcopy(envelope)
            replay_history.setdefault(key, []).append(copy.deepcopy(envelope))

        if not issues_sink:
            self._envelopes = replay_envelopes
            self._history = replay_history
