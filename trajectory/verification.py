"""Event-stream structural invariants + finalized-document verification
(issue #8 criterion 4, 6; schema enforcement per critic re-review). Shared
by the recorder (finalization) and the storage loader (reload) so both run
the SAME validation path: the versioned JSON Schema plus the stream
invariants plus integrity-digest verification."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from trajectory.recorder.errors import TrajectoryIntegrityError
from trajectory.storage.canonical import (
    SCHEMA_VERSION,
    content_digest,
)

TERMINAL_KINDS = ("finished", "budget_exhausted")

# event kinds that must reference a step that has already started
_STEP_BOUND_KINDS = frozenset({
    "skills_expressed", "model_called", "step_decision", "plan_parse_failed",
    "tool_called", "tool_rejected", "tool_result", "retry_scheduled",
})

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "trajectory-0.1.schema.json"
_SCHEMA_VALIDATOR: Draft202012Validator | None = None


def _schema_validator() -> Draft202012Validator:
    global _SCHEMA_VALIDATOR
    if _SCHEMA_VALIDATOR is None:
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        _SCHEMA_VALIDATOR = Draft202012Validator(schema)
    return _SCHEMA_VALIDATOR


def validate_event_stream(events: list[dict], *, require_terminal: bool) -> list[str]:
    """Validate structural invariants over ordered event records
    ({seq, kind, step, monotonic_s, data}); returns a list of violations
    (empty = valid). `require_terminal=False` supports crash/incomplete
    streams, which still must satisfy every other invariant."""
    issues: list[str] = []
    if not events:
        return ["event stream is empty"]
    if events[0]["kind"] != "run_started":
        issues.append(f"first event must be run_started, got {events[0]['kind']!r}")
    started_steps: set[int] = set()
    terminal_seen_at: int | None = None
    previous_monotonic: float | None = None
    previous_step: int | None = None

    for index, event in enumerate(events):
        seq, kind, step = event["seq"], event["kind"], event.get("step")
        monotonic = event.get("monotonic_s")
        if seq != index + 1:
            issues.append(f"seq {seq!r} at position {index} is not the contiguous value {index + 1}")
        if terminal_seen_at is not None:
            issues.append(f"event {seq} ({kind!r}) occurs after the terminal event at seq {terminal_seen_at}")
        if not isinstance(monotonic, (int, float)) or isinstance(monotonic, bool):
            issues.append(f"event {seq} monotonic_s {monotonic!r} is not a number")
        elif previous_monotonic is not None and monotonic < previous_monotonic:
            issues.append(f"event {seq} monotonic_s {monotonic!r} moves backwards "
                          f"(previous {previous_monotonic!r})")
        if kind == "run_started" and index != 0:
            issues.append(f"event {seq}: duplicate run_started")
        if kind == "step_started":
            if not isinstance(step, int) or isinstance(step, bool) or step < 1:
                issues.append(f"event {seq}: step_started has invalid step {step!r}")
            else:
                started_steps.add(step)
        if previous_step is not None and step is not None and step < previous_step:
            issues.append(f"event {seq}: step {step} moves backwards (previous {previous_step})")
        if kind in _STEP_BOUND_KINDS:
            if not isinstance(step, int) or isinstance(step, bool) or step < 1:
                issues.append(f"event {seq} ({kind!r}) must reference a positive step, got {step!r}")
            elif step not in started_steps:
                issues.append(f"event {seq} ({kind!r}) references step {step} which has not started")
        if step is not None:
            previous_step = step
        if isinstance(monotonic, (int, float)) and not isinstance(monotonic, bool):
            previous_monotonic = monotonic
        if kind in TERMINAL_KINDS:
            if terminal_seen_at is not None:
                issues.append(f"event {seq}: second terminal event "
                              f"(first at seq {terminal_seen_at})")
            terminal_seen_at = seq

    if require_terminal and terminal_seen_at is None:
        issues.append("completed trajectory requires exactly one terminal event "
                      "(finished or budget_exhausted)")
    return issues


def build_document(trajectory_id: str, context: dict, events: list[dict], *, status: str,
                   outcome: dict | None, incomplete_reason: str | None = None) -> dict:
    """Assemble the finalized trajectory document with its integrity digest
    (digest domain per trajectory/storage/canonical.py; the opaque
    trajectory_id is integrity-bound but never derived from the digest)."""
    if status == "incomplete" and not incomplete_reason:
        raise ValueError("incomplete finalization requires a reason")
    finalization = {"status": status, "outcome": outcome,
                    "incomplete_reason": incomplete_reason}
    digest = content_digest(trajectory_id, context, events, finalization)
    return {
        "trajectory_id": trajectory_id,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "complete": status != "incomplete",
        "context": copy.deepcopy(context),
        "events": copy.deepcopy(events),
        "finalization": {**finalization, "content_sha256": digest},
    }


def verify_document(document: dict) -> None:
    """Verify a loaded/finalized trajectory document through the single
    shared validation path: (1) the versioned JSON Schema, (2) event-stream
    invariants, (3) envelope coherence (complete/status, header identity is
    checked by the loader), (4) integrity-digest verification. Raises
    TrajectoryIntegrityError on any mismatch."""
    issues: list[str] = []
    schema_errors = [e.message for e in _schema_validator().iter_errors(document)]
    if schema_errors:
        raise TrajectoryIntegrityError(sorted(f"schema: {m}" for m in schema_errors))

    status = document["status"]
    events = document["events"]
    issues.extend(validate_event_stream(events, require_terminal=status != "incomplete"))

    # envelope coherence: complete flag must agree with the status
    if document["complete"] != (status != "incomplete"):
        issues.append(f"complete={document['complete']!r} is inconsistent with "
                      f"status={status!r}")
    if status == "incomplete" and not document["finalization"].get("incomplete_reason"):
        issues.append("incomplete trajectory must carry an incomplete_reason")
    if status in ("finished", "budget_exhausted"):
        terminal = next((e for e in events if e["kind"] in TERMINAL_KINDS), None)
        if terminal is not None and terminal["kind"] != status:
            issues.append(f"finalization status {status!r} disagrees with terminal event "
                          f"{terminal['kind']!r}")
        if status == "finished" and terminal is not None and \
                not terminal.get("data", {}).get("answer"):
            issues.append("finished terminal event carries no answer")

    finalization_raw = document.get("finalization") or {}
    # digest domain excludes the storage envelope keys (record type marker,
    # the digest itself, and the derived `complete` flag)
    finalization = {k: v for k, v in finalization_raw.items()
                    if k not in ("record", "content_sha256", "complete")}
    recomputed = content_digest(document["trajectory_id"], document["context"], events,
                                 finalization)
    recorded = finalization_raw.get("content_sha256")
    if recorded != recomputed:
        issues.append(f"integrity digest mismatch: recorded {recorded!r}, "
                      f"recomputed {recomputed!r}")
    if issues:
        raise TrajectoryIntegrityError(sorted(issues))
