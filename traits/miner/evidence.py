"""Deterministic differential evidence renderer (issue #9 criterion 3).

Pure reducer from a MiningPair to compact canonical-JSON evidence for the
teacher prompt. Preserves failures in full (malformed attempts, rejected
calls, retries) — never only the successful path. Excludes provenance such
as git/backend identifiers (those belong to the mining record, not the
teacher prompt). `evidence_sha256` covers the exact rendered evidence.
"""

from __future__ import annotations

from trajectory.storage.canonical import canonical_bytes, canonical_json, sha256_hex
from traits.miner.inputs import MinerInputError, MiningPair


def _render_member(nt, label: str) -> dict:
    steps = []
    for step in nt.steps:
        steps.append({
            "step": step.step,
            "expressed_skills": list(step.expressed_skills),
            "thoughts": list(step.thoughts),
            "model_calls": [{
                "parse_ok": call.parse_ok,
                "parse_error": call.parse_error,
                "raw_response": call.raw_response,
            } for call in step.model_calls],
            "tool_calls": [{
                "tool": call.tool,
                "arguments": call.arguments,
                "rejected": call.rejected,
                "reject_reason": call.reject_reason,
                "ok": call.ok,
                "result_content": call.result_content,
            } for call in step.tool_calls],
            "retries": [{"attempt": r.attempt, "delay_s": r.delay_s, "reason": r.reason}
                         for r in step.retries],
        })
    return {
        "label": label,
        "runtime_status": nt.status,
        "final_answer": nt.answer,
        "outcome": {"success": (nt.outcome or {}).get("success"),
                     "reward": (nt.outcome or {}).get("reward")},
        "expressed_skills": list(nt.expressed_skill_ids),
        "steps": steps,
        "cost": dict(nt.action_cost,
                      prompt_tokens=nt.token_cost["prompt"],
                      completion_tokens=nt.token_cost["completion"],
                      total_tokens=nt.token_cost["total"]),
    }


def render_pair_evidence(pair: MiningPair) -> tuple[str, str]:
    """Render canonical differential evidence for one pair.

    Returns `(evidence_text, evidence_sha256)` where the digest covers the
    exact canonical bytes of the rendered document."""
    if pair.success.task != pair.failure.task:  # defensive; MiningPair enforces this
        raise MinerInputError("pair members must share the same task")
    document = {
        "kind": "heritagent-differential-evidence",
        "version": "0.1",
        "task": pair.success.task,
        "task_family": pair.task_family,
        "successful_trajectory": _render_member(pair.success, "success"),
        "failed_trajectory": _render_member(pair.failure, "failure"),
    }
    text = canonical_json(document)
    return text, sha256_hex(canonical_bytes(document))


__all__ = ["render_pair_evidence"]
