"""TrajectoryRecorder v0.1 (issue #8): hook-only capture, append-only JSONL
persistence, finalization with an integrity digest.

The recorder is a plain callable — pass it as one of `run_agent(hooks=...)`.
It never rewrites or forks the agent loop, never reads ambient state, and
deep-copies every incoming event so later caller mutation cannot alter
stored evidence.

Identity: a stable opaque `trajectory_id` is allocated at recorder creation
(caller-supplied for deterministic fixtures, otherwise `T-<uuid hex>`) and
written into the JSONL header, so crash-recovered prefixes retain the same
identity. The id is integrity-bound (inside the digest domain) but never
derived from the digest: post-run outcome changes alter the digest, not the
trajectory identity.
"""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from runtime.events import RuntimeEvent
from trajectory.recorder.context import OutcomeAnnotation, TrajectoryContext
from trajectory.recorder.errors import (
    RecorderError,
    StreamInvariantError,
    TrajectoryIntegrityError,
)
from trajectory.storage.canonical import canonical_json
from trajectory.storage.jsonl import JsonlWriter
from trajectory.verification import (TERMINAL_KINDS, build_document, validate_event_stream,
                                      verify_document)

_TRAJECTORY_ID_PATTERN = re.compile(r"^T-[0-9a-f]{12}$")


@dataclass(frozen=True)
class FinalizedTrajectory:
    """In-memory finalized trajectory document (schema trajectory-0.1)."""

    document: dict

    @property
    def trajectory_id(self) -> str:
        return self.document["trajectory_id"]

    @property
    def content_sha256(self) -> str:
        return self.document["finalization"]["content_sha256"]

    @property
    def status(self) -> str:
        return self.document["status"]

    @property
    def outcome(self) -> dict | None:
        return self.document["finalization"].get("outcome")


class TrajectoryRecorder:
    """Capture RuntimeEvents through the #7 hook contract and persist them
    as append-only canonical JSONL."""

    def __init__(self, context: TrajectoryContext, path: Path,
                 trajectory_id: str | None = None):
        if trajectory_id is not None and not _TRAJECTORY_ID_PATTERN.match(trajectory_id):
            raise RecorderError(
                f"trajectory_id {trajectory_id!r} must match T-<12 lowercase hex chars>")
        self.context = context  # caller-owned; NEVER re-serialized after this point
        self.path = Path(path)
        self.trajectory_id = trajectory_id or f"T-{uuid.uuid4().hex[:12]}"
        # creation-time deep snapshot: header AND digest are built from this
        # exact snapshot, so later mutation of the caller's nested
        # tags/metadata dicts cannot invalidate recorder-produced evidence
        self._context_snapshot = copy.deepcopy(context.to_dict())
        self._outcome_snapshot: dict | None = None
        self._events: list[dict] = []
        self._finalized = False
        self._closed = False
        # atomic exclusive creation happens inside JsonlWriter (mode "x"):
        # fail closed BEFORE writing any bytes, no check-then-open race
        self._writer = JsonlWriter(self.path)
        self._writer.append({"record": "header", "schema_version": "0.1",
                             "trajectory_id": self.trajectory_id,
                             "context": self._context_snapshot})

    # -- hook contract ------------------------------------------------------
    def _require_open(self) -> None:
        if self._closed:
            raise RecorderError("trajectory recorder is closed")

    def close(self) -> None:
        """Idempotent resource cleanup: releases the evidence file handle.
        Further capture/finalization is rejected until the process ends
        (an unclosed stream is recovered via the allow_incomplete path)."""
        if not self._closed:
            self._writer.close()
            self._closed = True

    def abort(self) -> None:
        """Close the recorder and mark it aborted: no further capture or
        finalization. The captured prefix stays recoverable through the
        allow_incomplete reload path."""
        self.close()

    def __call__(self, event: RuntimeEvent) -> None:
        """Subscribe via `run_agent(hooks=(recorder,))`. Deep-copies the
        incoming payload; validates incremental invariants fail-closed."""
        if self._finalized:
            raise RecorderError("event captured after finalization")
        self._require_open()
        record = {
            "record": "event",
            "seq": event.seq,
            "kind": event.kind,
            "step": event.step,
            "monotonic_s": event.monotonic_s,
            "data": copy.deepcopy(event.data),
        }
        try:
            serialized = canonical_json(record)
        except ValueError as exc:  # non-finite floats etc. — strict JSON
            raise RecorderError(f"event payload is not strict JSON: {exc}") from None
        incremental_issues = validate_event_stream(self._events + [record],
                                                    require_terminal=False)
        if incremental_issues:
            raise StreamInvariantError(sorted(set(incremental_issues)))
        del serialized
        self._events.append(record)
        self._writer.append(record)

    # -- finalization ---------------------------------------------------------
    def attach_outcome(self, outcome: OutcomeAnnotation) -> None:
        """Attach a benchmark-agnostic outcome annotation before finalization;
        never mutates historical events. Changing the outcome changes the
        integrity digest but NOT the trajectory identity. The annotation is
        deep-snapshotted at attach time: later mutation of the caller's
        outcome.metadata cannot silently change the attached annotation."""
        if self._finalized:
            raise RecorderError("outcome attached after finalization")
        self._require_open()
        self._outcome_snapshot = copy.deepcopy(outcome.to_dict())

    def finalize(self) -> FinalizedTrajectory:
        """Finalize a COMPLETED run (terminal event present). Computes the
        integrity digest over id + header + ordered events + finalization and
        appends the finalization record, then closes the writer."""
        if self._finalized:
            raise RecorderError("trajectory already finalized")
        self._require_open()
        issues = validate_event_stream(self._events, require_terminal=True)
        if issues:
            raise StreamInvariantError(sorted(set(issues)))
        terminal = self._events[-1]
        return self._persist(status=terminal["kind"])

    def finalize_incomplete(self, reason: str) -> FinalizedTrajectory:
        """Explicit crash/interruption recovery: persist the captured prefix
        as an INCOMPLETE trajectory. Never silently labels an incomplete
        stream as a completed failure."""
        if self._finalized:
            raise RecorderError("trajectory already finalized")
        self._require_open()
        if any(event["kind"] in TERMINAL_KINDS for event in self._events):
            # not a recovery path: release the evidence handle and stop
            self.close()
            raise RecorderError("run already has a terminal event; use finalize()")
        if not reason or not reason.strip():
            raise RecorderError("finalize_incomplete requires a reason")
        issues = validate_event_stream(self._events, require_terminal=False)
        if issues:
            raise StreamInvariantError(sorted(set(issues)))
        return self._persist(status="incomplete", incomplete_reason=reason)

    def _persist(self, *, status: str, incomplete_reason: str | None = None) -> FinalizedTrajectory:
        if self._finalized:
            raise RecorderError("trajectory already finalized")
        self._require_open()
        document = build_document(self.trajectory_id, self._context_snapshot, self._events,
                                   status=status, outcome=self._outcome_snapshot,
                                   incomplete_reason=incomplete_reason)
        # shared validation path: the footer is written ONLY for evidence
        # that passes the full document verification (schema + invariants +
        # digest coherence); reload verification remains defense in depth
        try:
            verify_document(document)
        except TrajectoryIntegrityError:
            # unrecoverable finalization failure: release the handle; the
            # captured prefix stays recoverable via allow_incomplete reload
            self.close()
            raise
        self._writer.append({"record": "finalization",
                             "status": status,
                             "outcome": self._outcome_snapshot,
                             "incomplete_reason": incomplete_reason,
                             "complete": status != "incomplete",
                             "content_sha256": document["finalization"]["content_sha256"]})
        self._finalized = True
        self._writer.close()
        return FinalizedTrajectory(document=document)


__all__ = ["FinalizedTrajectory", "TrajectoryRecorder", "TERMINAL_KINDS"]
