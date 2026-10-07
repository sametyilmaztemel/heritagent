"""Append-only hash-chained journal for the somatic trait store (issue #10).

Record digest domain (v0.1, locked)::

    record_sha256 = sha256(canonical_json({
        "schema_version": "0.1",
        "seq": <int>,
        "op": <"candidate_added" | "decision_recorded">,
        "trait": {"gene_id": ..., "version": ...},
        "envelope": <full resulting somatic envelope>,
        "prev_sha256": <previous record digest or null for seq 1>,
    }))

The hash chain makes any edit/reorder/removal of prior records detectable
on replay. Canonical strict JSON (UTF-8, sorted keys, compact separators,
``allow_nan=False``); no pickle/arbitrary deserialization anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from trajectory.storage.canonical import canonical_json, sha256_hex

SCHEMA_VERSION = "0.1"

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / \
    "somatic-journal-record-0.1.schema.json"
_VALIDATOR: Draft202012Validator | None = None


def journal_record_validator() -> Draft202012Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        _VALIDATOR = Draft202012Validator(schema)
    return _VALIDATOR


def compute_record_digest(*, seq: int, op: str, trait: dict, envelope: dict,
                          prev_sha256: str | None) -> str:
    """Deterministic record digest over the full semantic record excluding
    its own digest (domain locked above)."""
    return sha256_hex(canonical_json({
        "schema_version": SCHEMA_VERSION,
        "seq": seq,
        "op": op,
        "trait": trait,
        "envelope": envelope,
        "prev_sha256": prev_sha256,
    }).encode("utf-8"))


def build_record(*, seq: int, op: str, gene_id: str, version: int,
                 envelope: dict, prev_sha256: str | None) -> dict:
    """Assemble a chained journal record with its digest."""
    trait = {"gene_id": gene_id, "version": version}
    digest = compute_record_digest(seq=seq, op=op, trait=trait,
                                    envelope=envelope, prev_sha256=prev_sha256)
    return {
        "record": "somatic-journal",
        "schema_version": SCHEMA_VERSION,
        "seq": seq,
        "op": op,
        "trait": trait,
        "envelope": envelope,
        "prev_sha256": prev_sha256,
        "record_sha256": digest,
    }


def _reject_non_finite_constant(name: str):
    raise ValueError(f"non-finite JSON constant {name!r} is not allowed in journal files")


def read_journal_records(path: Path) -> list[dict]:
    """Strictly parse journal lines; corruption fails closed (no recovery
    path in v0.1 — somatic memory is never silently salvaged)."""
    records: list[dict] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line, parse_constant=_reject_non_finite_constant))
        except json.JSONDecodeError as exc:
            from trajectory.recorder.errors import MalformedTrajectoryError
            raise MalformedTrajectoryError(
                f"{path}: malformed journal JSON at line {index + 1}: {exc}") from None
        except ValueError as exc:
            from trajectory.recorder.errors import MalformedTrajectoryError
            raise MalformedTrajectoryError(
                f"{path}: non-standard JSON at line {index + 1}: {exc}") from None
    return records


class JournalAppender:
    """Append-only journal writer; flushes every committed record."""

    def __init__(self, path: Path, *, exclusive: bool):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if exclusive:
            try:
                self._handle = open(self.path, "x", encoding="utf-8")  # O_CREAT|O_EXCL
            except FileExistsError as exc:
                from trajectory.recorder.errors import RecorderError
                raise RecorderError(
                    f"refusing to create somatic journal {str(self.path)!r}: "
                    f"path already exists (append-only evidence store; no resume "
                    f"protocol in v0.1)") from None
        else:
            self._handle = open(self.path, "a", encoding="utf-8")
        self._closed = False

    def append(self, record: dict) -> None:
        if self._closed:
            from trajectory.recorder.errors import RecorderError
            raise RecorderError("journal writer is closed")
        self._handle.write(canonical_json(record) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True
