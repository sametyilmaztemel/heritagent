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
- persistence is an append-only hash-chained journal (see somatic/journal.py);
  replay reconstructs state deterministically and every resulting envelope
  is validated through the existing ``load_somatic(..., registry)`` boundary.

Explicit create/open semantics (issue #8 lesson): ``SomaticStore.create``
uses atomic exclusive file creation; ``SomaticStore.open`` replays and fully
verifies an existing journal before it becomes append-capable; neither path
truncates or overwrites somatic memory. ``SomaticStore(path=None)`` is the
in-memory variant (same lifecycle, no persistence).
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
        self._closed = False

        self._writer = None
        self._closed = True
        if self.path is None:
            if mode == "open":
                raise RecorderError("cannot open an in-memory store; use create")
            self._closed = False
            return

        if mode == "create":
            # atomic exclusive creation (issue #8 lesson): never overwrite
            self._writer = JournalAppender(self.path, exclusive=True)
            self._closed = False
        else:
            self._replay_and_verify()
            self._writer = JournalAppender(self.path, exclusive=False)
            self._closed = False

    # -- explicit constructors -------------------------------------------------
    @classmethod
    def create(cls, path: Path | None, registry: TraitRegistry | None = None) -> "SomaticStore":
        return cls(path, registry=registry, mode="create")

    @classmethod
    def open(cls, path: Path, registry: TraitRegistry | None = None) -> "SomaticStore":
        if path is None:
            raise RecorderError("open requires a journal path")
        return cls(path, registry=registry, mode="open")

    # -- lifecycle: candidate ----------------------------------------------------
    def add_candidate(self, envelope: dict) -> dict:
        """Insert a NEW acquired candidate. Only ``validation.state ==
        "candidate"`` is accepted; acquired semantics (origin="acquired",
        no cig_record, no gate_reports) are enforced by the existing somatic
        schema through ``load_somatic``."""
        stored = copy.deepcopy(envelope)
        if stored["validation"]["state"] != "candidate":
            raise RecordConsistencyError(
                "add_candidate accepts only validation.state == 'candidate'; "
                "use decide() for terminal transitions")
        if stored["candidate"]["origin"] != "acquired":
            raise RecordConsistencyError(
                "somatic candidates must be lifetime-acquired (origin='acquired')")
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
        if current["validation"]["state"] != "candidate":
            raise RecordConsistencyError(
                f"somatic trait {gene_id!r}@{version} is terminal "
                f"({current['validation']['state']!r}); decisions are immutable")

        terminal = copy.deepcopy(current)
        terminal["validation"] = {
            "state": verdict,
            "gate_reports": sorted(set(gate_reports)),  # deterministic canonical order
        }
        # lifecycle invariant: only the validation envelope may change
        if canonical_json(terminal["candidate"]) != canonical_json(current["candidate"]):
            raise RecordConsistencyError(
                "decision mutated the candidate gene ref/provenance; only the "
                "validation envelope may change")
        load_somatic(copy.deepcopy(terminal), self._registry)  # schema re-validation
        self._commit("decision_recorded", terminal, key=key)
        return copy.deepcopy(terminal)

    # -- read API (defensive copies; #11/#12 surface) ----------------------------
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
        """Convenient read-only access to the immutable gene ref."""
        entry = self._envelopes.get((gene_id, version))
        return copy.deepcopy(entry["candidate"]) if entry else None

    def history(self, gene_id: str, version: int) -> tuple[dict, ...]:
        """Ordered envelope snapshots for one trait (candidate → decision)."""
        return tuple(copy.deepcopy(env) for env in
                      self._history.get((gene_id, version), []))

    # -- integrity ------------------------------------------------------------------
    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    # -- internals ---------------------------------------------------------------------
    @staticmethod
    def _key(envelope: dict) -> tuple[str, int]:
        candidate = envelope["candidate"]
        return candidate["gene_id"], candidate["version"]

    def _commit(self, op: str, envelope: dict, *, key: tuple[str, int]) -> None:
        """Append one journal record (schema-checked) and apply the state
        transition. Replay applies transitions directly in _replay_records."""
        prev = self._records[-1]["record_sha256"] if self._records else None
        record = build_record(seq=len(self._records) + 1, op=op,
                               gene_id=key[0], version=key[1],
                               envelope=envelope, prev_sha256=prev)
        # schema validation of the assembled record (defense in depth)
        errors = [e.message for e in journal_record_validator().iter_errors(record)]
        if errors:
            raise RecordConsistencyError(sorted(errors))
        if self._writer is not None:  # in-memory stores keep records only
            self._writer.append(record)
        self._records.append(record)
        self._envelopes[key] = copy.deepcopy(envelope)
        self._history.setdefault(key, []).append(copy.deepcopy(envelope))

    def _replay_and_verify(self) -> None:
        self._records = read_journal_records(self.path)
        issues: list[str] = []
        self._envelopes, self._history = {}, {}
        self._replay_records(self._records, self._registry, issues)
        if issues:
            # a failed replay leaves the store unusable (fail closed)
            self.close()
            self._envelopes, self._history = {}, {}
            raise RecordConsistencyError(f"{self.path}: {sorted(issues)}")

    def verify(self) -> list[str]:
        """Full replay from the journal source of truth; returns consistency
        issues (empty = sound). Cached state is never trusted over replay."""
        issues: list[str] = []
        self._replay_records(self._records, self._registry, issues)
        return issues

    def verify_strict(self) -> None:
        issues = self.verify()
        if issues:
            raise RecordConsistencyError(sorted(issues))

    def _replay_records(self, records: list[dict], registry, issues_sink: list[str]) -> None:
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
            try:
                load_somatic(copy.deepcopy(envelope), registry)
            except Exception as exc:  # noqa: BLE001 — any envelope violation is a replay issue
                issues_sink.append(f"record {seq}: envelope fails somatic validation: {exc}")
                continue

            op = record["op"]
            if op == "candidate_added":
                if key in replay_envelopes:
                    issues_sink.append(f"record {seq}: duplicate candidate insertion "
                                        f"for {key}")
                    continue
                if envelope["validation"]["state"] != "candidate":
                    issues_sink.append(f"record {seq}: candidate_added with non-candidate "
                                        f"state {envelope['validation']['state']!r}")
                    continue
            else:  # decision_recorded
                current = replay_envelopes.get(key)
                if current is None:
                    issues_sink.append(f"record {seq}: decision for unknown trait {key}")
                    continue
                if current["validation"]["state"] != "candidate":
                    issues_sink.append(f"record {seq}: decision on terminal trait {key} "
                                        f"({current['validation']['state']!r})")
                    continue
                if canonical_json(envelope["candidate"]) != \
                        canonical_json(current["candidate"]):
                    issues_sink.append(f"record {seq}: decision mutated the candidate "
                                        f"gene ref/provenance")
                    continue
            replay_envelopes[key] = copy.deepcopy(envelope)
            replay_history.setdefault(key, []).append(copy.deepcopy(envelope))
        # deterministic state reconstruction (only when the replay was clean)
        if not issues_sink:
            self._envelopes = replay_envelopes
            self._history = replay_history
        return None

    # replace the stub with a bound wrapper
    def verify(self) -> list[str]:  # noqa: F811 — final implementation
        issues: list[str] = []
        saved = (self._envelopes, self._history)
        try:
            self._envelopes, self._history = {}, {}
            self._replay_records(self._records, self._registry, issues)
        finally:
            self._envelopes, self._history = saved
        return issues
