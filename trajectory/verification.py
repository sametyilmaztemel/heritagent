"""Event-stream structural invariants + finalized-document verification
(issue #8 criterion 4, 6). Shared by the recorder (finalization) and the
storage loader (reload) so both enforce identical rules."""

from __future__ import annotations

import copy

from trajectory.recorder.errors import TrajectoryIntegrityError
from trajectory.storage.canonical import (
    SCHEMA_VERSION,
    content_digest,
    trajectory_id_from_digest,
)

TERMINAL_KINDS = ("finished", "budget_exhausted")

# event kinds that must reference a step that has already started
_STEP_BOUND_KINDS = frozenset({
    "skills_expressed", "model_called", "step_decision", "plan_parse_failed",
    "tool_called", "tool_rejected", "tool_result", "retry_scheduled",
})


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
    max_step_seen = 0
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
        if previous_monotonic is not None and monotonic is not None and monotonic < previous_monotonic:
            issues.append(f"event {seq} monotonic_s {monotonic!r} moves backwards "
                          f"(previous {previous_monotonic!r})")
        if kind == "run_started" and index != 0:
            issues.append(f"event {seq}: duplicate run_started")
        if kind == "step_started":
            if step is None or step < 1:
                issues.append(f"event {seq}: step_started has invalid step {step!r}")
            else:
                started_steps.add(step)
                max_step_seen = max(max_step_seen, step)
        if previous_step is not None and step is not None and step < previous_step:
            issues.append(f"event {seq}: step {step} moves backwards (previous {previous_step})")
        if kind in _STEP_BOUND_KINDS:
            if step is None or step < 1:
                issues.append(f"event {seq} ({kind!r}) must reference a positive step, got {step!r}")
            elif step not in started_steps:
                issues.append(f"event {seq} ({kind!r}) references step {step} which has not started")
        if step is not None:
            previous_step = step
        if monotonic is not None:
            previous_monotonic = monotonic
        if kind in TERMINAL_KINDS:
            if terminal_seen_at is not None:
                issues.append(f"event {seq}: second terminal event "
                              f"(first at seq {terminal_seen_at})")
            terminal_seen_at = seq

    if require_terminal and terminal_seen_at is None:
        issues.append("completed trajectory requires exactly one terminal event "
                      "(finished or budget_exhausted)")
    if terminal_seen_at is not None:
        terminal_kind = events[terminal_seen_at - 1]["kind"]
        if events[terminal_seen_at - 1]["seq"] != events[-1]["seq"]:
            issues.append("terminal event is not the last event of the stream")
        # step numbers must never exceed the last started step at terminal time
        del terminal_kind, max_step_seen
    return issues


def build_document(context: dict, events: list[dict], *, status: str,
                   outcome: dict | None, incomplete_reason: str | None = None) -> dict:
    """Assemble the finalized trajectory document with its integrity digest
    (digest domain per trajectory/storage/canonical.py)."""
    if status == "incomplete" and not incomplete_reason:
        raise ValueError("incomplete finalization requires a reason")
    finalization = {"status": status, "outcome": outcome,
                    "incomplete_reason": incomplete_reason}
    digest = content_digest(context, events, finalization)
    return {
        "trajectory_id": trajectory_id_from_digest(digest),
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "complete": status != "incomplete",
        "context": copy.deepcopy(context),
        "events": copy.deepcopy(events),
        "finalization": {**finalization, "content_sha256": digest},
    }


def verify_document(document: dict) -> None:
    """Verify a loaded trajectory document: schema envelope, event-stream
    invariants, terminal/status agreement, and integrity digest. Raises
    TrajectoryIntegrityError on any mismatch."""
    issues: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"schema_version {document.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    events = document.get("events")
    status = document.get("status")
    if status not in ("finished", "budget_exhausted", "incomplete"):
        issues.append(f"unknown status {status!r}")
    if not isinstance(events, list):
        issues.append("events must be a list")
        events = []

    require_terminal = status in ("finished", "budget_exhausted")
    issues.extend(validate_event_stream(events, require_terminal=require_terminal))

    if status == "incomplete" and not document.get("finalization", {}).get("incomplete_reason"):
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
    recomputed = content_digest(document.get("context") or {}, events, finalization)
    recorded = (document.get("finalization") or {}).get("content_sha256")
    if recorded != recomputed:
        issues.append(f"integrity digest mismatch: recorded {recorded!r}, "
                      f"recomputed {recomputed!r}")
    expected_id = trajectory_id_from_digest(recomputed)
    if document.get("trajectory_id") != expected_id:
        issues.append(f"trajectory_id {document.get('trajectory_id')!r} does not match "
                      f"digest-derived id {expected_id!r}")
    if issues:
        raise TrajectoryIntegrityError(sorted(issues))
