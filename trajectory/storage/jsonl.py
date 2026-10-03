"""Append-only JSONL storage + reload (issue #8 criterion 5; hardened per
critic review).

One JSON line per record, canonical JSON (sorted keys, compact separators,
UTF-8, `allow_nan=False`). The writer refuses to start on an existing
non-empty file (exclusive create, fail closed — no overwrite, no truncate)
and flushes on every record: a crash can never rewrite previously persisted
valid lines. The reader parses strict JSON per line — no pickle, no code
deserialization, no non-standard NaN/Infinity constants — and supports an
explicit `allow_incomplete=True` recovery path for a truncated trailing
line.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from trajectory.recorder.context import BudgetSnapshot, ModelProvenance, TrajectoryContext
from trajectory.recorder.errors import (
    MalformedTrajectoryError,
    RecorderError,
    TrajectoryIntegrityError,
)
from trajectory.storage.canonical import canonical_json
from trajectory.verification import build_document, verify_document


class JsonlWriter:
    """Append-only canonical-JSONL writer; flushes on every record.

    Fail-closed exclusive create: refuses to open a path that already
    contains data (evidence is never overwritten, truncated, or appended
    onto). Closes cleanly on finalization."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists() and self.path.stat().st_size > 0:
            raise RecorderError(
                f"refusing to append: {str(self.path)!r} already contains trajectory data")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8")
        self._closed = False

    def append(self, record: dict) -> None:
        if self._closed:
            raise RecorderError("writer is closed")
        self._handle.write(canonical_json(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


@dataclass(frozen=True)
class LoadedTrajectory:
    """Reloaded trajectory evidence. `complete=False` marks an explicitly
    recovered incomplete/crashed stream."""

    trajectory_id: str
    status: str
    complete: bool
    context: TrajectoryContext
    events: tuple[dict, ...]
    outcome: dict | None
    content_sha256: str
    document: dict
    source_path: Path


def _reject_non_finite_constant(name: str):
    raise ValueError(f"non-finite JSON constant {name!r} is not allowed in evidence files")


def read_records(path: Path, *, allow_incomplete: bool = False) -> list[dict]:
    """Parse JSONL records strictly; a malformed TRAILING line is tolerated
    only on the explicit recovery path (treated as a crash artifact).
    Non-standard JSON constants (NaN/Infinity/-Infinity) are rejected."""
    records: list[dict] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line, parse_constant=_reject_non_finite_constant))
        except json.JSONDecodeError as exc:
            is_trailing = index == len(lines) - 1
            if allow_incomplete and is_trailing:
                break  # truncated crash artifact: keep the valid prefix
            raise MalformedTrajectoryError(
                f"{path}: malformed JSONL at line {index + 1}: {exc}") from None
        except ValueError as exc:
            raise MalformedTrajectoryError(
                f"{path}: non-standard JSON at line {index + 1}: {exc}") from None
    return records


def _split_records(records: list[dict], path: Path) -> tuple[dict, list[dict], dict | None]:
    if not records:
        raise MalformedTrajectoryError(f"{path}: empty trajectory file")
    header = records[0]
    if header.get("record") != "header":
        raise MalformedTrajectoryError(f"{path}: first record must be the header")
    if header.get("schema_version") != "0.1":
        raise MalformedTrajectoryError(
            f"{path}: unsupported header schema_version {header.get('schema_version')!r}; "
            f"this loader speaks 0.1 only")
    if not header.get("trajectory_id"):
        raise MalformedTrajectoryError(f"{path}: header carries no trajectory_id")
    if any(r.get("record") == "header" for r in records[1:]):
        raise MalformedTrajectoryError(f"{path}: multiple header records")
    finalization = next((r for r in records if r.get("record") == "finalization"), None)
    if finalization is not None and records[-1] is not finalization:
        raise MalformedTrajectoryError(f"{path}: finalization record must be last")
    events = [r for r in records[1:] if r.get("record") == "event"]
    unknown = [r.get("record") for r in records[1:]
               if r.get("record") not in ("event", "finalization")]
    if unknown:
        raise MalformedTrajectoryError(f"{path}: unknown record types {sorted(set(unknown))}")
    return header, events, finalization


def load_trajectory(path: Path, *, allow_incomplete: bool = False) -> LoadedTrajectory:
    """Reload a trajectory file: strict JSONL parse, header/version checks,
    structural invariants, envelope coherence, and integrity-digest
    verification — through the same shared verification path used at
    finalization."""
    path = Path(path)
    records = read_records(path, allow_incomplete=allow_incomplete)
    header, events, finalization = _split_records(records, path)
    trajectory_id = header["trajectory_id"]

    status = (finalization or {}).get("status")
    if finalization is None:
        if not allow_incomplete:
            raise MalformedTrajectoryError(
                f"{path}: no finalization record (crashed run?); "
                f"reload with allow_incomplete=True to recover the prefix")
        # crashed run: no recorded digest exists — rebuild verifiable evidence
        # for exactly the prefix that survived, under the SAME trajectory id
        document = build_document(trajectory_id, header.get("context") or {}, events,
                                   status="incomplete", outcome=None,
                                   incomplete_reason="no finalization record (crashed run)")
    else:
        document = {"trajectory_id": trajectory_id, "schema_version": "0.1",
                    "status": status,
                    "complete": finalization.get("complete", status != "incomplete"),
                    "context": header.get("context"), "events": events,
                    "finalization": {k: v for k, v in finalization.items()
                                      if k not in ("record", "content_sha256", "complete")},
                    }
        document["finalization"]["content_sha256"] = finalization["content_sha256"]
    try:
        verify_document(document)
    except TrajectoryIntegrityError as exc:
        raise TrajectoryIntegrityError(f"{path}: {exc}") from None

    header_context = header.get("context") or {}
    context = TrajectoryContext(
        genome_id=header_context["genome_id"],
        run_id=header_context.get("run_id"),
        experiment_id=header_context.get("experiment_id"),
        model=ModelProvenance(**header_context["model"]) if header_context.get("model") else None,
        seed=header_context.get("seed"),
        code_commit=header_context.get("code_commit"),
        environment=header_context.get("environment"),
        benchmark=header_context.get("benchmark"),
        split=header_context.get("split"),
        budget=BudgetSnapshot(**header_context["budget"]) if header_context.get("budget") else None,
        tags=header_context.get("tags") or {},
        metadata=header_context.get("metadata") or {})
    return LoadedTrajectory(
        trajectory_id=document["trajectory_id"], status=document["status"],
        complete=document["complete"], context=context,
        events=tuple(copy.deepcopy(e) for e in document["events"]),
        outcome=document["finalization"].get("outcome"),
        content_sha256=document["finalization"]["content_sha256"],
        document=document, source_path=path)
