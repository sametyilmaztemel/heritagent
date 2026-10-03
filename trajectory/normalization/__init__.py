"""Deterministic NormalizedTrajectory reducer (issue #8 criteria 7, 8, 10).

Pure function: raw event stream → NormalizedTrajectory. Preserves ALL raw
evidence (including malformed attempts and rejections) while deriving the
phenotype view #9 (Trait Miner) consumes — without knowing EventLog
internals. No mining or ranking happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trajectory.recorder.context import TrajectoryContext


@dataclass(frozen=True)
class ModelCall:
    """One model invocation. `parse_ok=False` marks a malformed attempt —
    preserved deliberately as mining evidence (never discarded)."""

    step: int
    messages: tuple[dict, ...]
    raw_response: str
    parse_ok: bool | None          # None while unresolved; resolved by the next decision/parse-failure
    parse_error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    prompt_tokens_exact: bool | None
    completion_tokens_exact: bool | None
    latency_ms: float | None


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation (or rejection: `rejected=True`, env never called)."""

    step: int
    tool: str
    arguments: dict
    rejected: bool = False
    reject_reason: str | None = None
    ok: bool | None = None         # None for rejected calls
    result_content: str | None = None
    latency_ms: float | None = None


@dataclass(frozen=True)
class Retry:
    step: int
    attempt: int
    delay_s: float
    reason: str


@dataclass(frozen=True)
class StepView:
    step: int
    expressed_skills: tuple[str, ...] = ()
    thoughts: tuple[str, ...] = ()
    model_calls: tuple[ModelCall, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    retries: tuple[Retry, ...] = ()


@dataclass(frozen=True)
class NormalizedTrajectory:
    trajectory_id: str
    task: str
    status: str                       # finished | budget_exhausted | incomplete
    answer: str | None
    outcome: dict | None
    steps: tuple[StepView, ...]
    parse_failures: tuple[ModelCall, ...]
    failed_tool_calls: tuple[ToolCall, ...]     # executed but not ok
    rejected_tool_calls: tuple[ToolCall, ...]   # never reached the environment
    final_successful_path: tuple[ToolCall, ...]  # ordered successful tool calls
    expressed_skill_ids: tuple[str, ...]
    token_cost: dict                  # {prompt, completion, total, prompt_exact, completion_exact, exact}
    latency: dict                     # {model_total_ms, tool_total_ms}
    action_cost: dict                 # step/call/retry counts (H3 adaptation-cost inputs)
    metrics: dict                     # derived metric totals (criterion 8)
    provenance: dict                  # genome/model/seed/commit/benchmark context
    event_count: int


def normalize_trajectory(*, trajectory_id: str, context: TrajectoryContext | None,
                         events, status: str, outcome: dict | None = None) -> NormalizedTrajectory:
    """Reduce ordered event records ({seq, kind, step, monotonic_s, data})
    into the normalized phenotype view. Deterministic: identical evidence
    yields an identical NormalizedTrajectory."""
    task = None
    answer = None
    steps: dict[int, dict] = {}
    expressed_order: list[str] = []
    parse_failures: list[ModelCall] = []
    failed_tool_calls: list[ToolCall] = []
    rejected_tool_calls: list[ToolCall] = []
    successful_tool_calls: list[ToolCall] = []
    total_prompt = 0
    total_completion = 0
    exact_prompts: list[bool] = []
    exact_completions: list[bool] = []
    model_latency_total = 0.0
    tool_latency_total = 0.0
    model_call_count = 0
    tool_call_count = 0
    tool_successes = 0
    tool_failures = 0
    tool_rejections = 0
    retry_count = 0
    parse_failure_count = 0
    expressed_events = 0

    def step_bucket(step: int | None) -> dict:
        step = step or 0
        return steps.setdefault(step, {"expressed": [], "thoughts": [], "models": [],
                                        "tools": [], "retries": []})

    for event in events:
        kind, step, data = event["kind"], event.get("step"), event.get("data", {})
        bucket = step_bucket(step)
        if kind == "run_started":
            task = data.get("task")
        elif kind == "skills_expressed":
            expressed_events += 1
            skills = tuple(data.get("skills") or ())
            bucket["expressed"].extend(skills)
            for skill_id in skills:
                if skill_id not in expressed_order:
                    expressed_order.append(skill_id)
        elif kind == "model_called":
            model_call_count += 1
            total_prompt += data.get("prompt_tokens") or 0
            total_completion += data.get("completion_tokens") or 0
            exact_prompts.append(bool(data.get("prompt_tokens_exact")))
            exact_completions.append(bool(data.get("completion_tokens_exact")))
            model_latency_total += data.get("latency_ms") or 0.0
            bucket["models"].append(ModelCall(
                step=step, messages=tuple(data.get("messages") or ()),
                raw_response=data.get("raw_response", ""), parse_ok=None,
                parse_error=None,
                prompt_tokens=data.get("prompt_tokens"),
                completion_tokens=data.get("completion_tokens"),
                prompt_tokens_exact=data.get("prompt_tokens_exact"),
                completion_tokens_exact=data.get("completion_tokens_exact"),
                latency_ms=data.get("latency_ms")))
        elif kind == "plan_parse_failed":
            parse_failure_count += 1
            pending = _last_unresolved(bucket["models"])
            if pending is not None:
                failed_call = _replaced(bucket["models"][pending], parse_ok=False,
                                         parse_error=data.get("error"))
                bucket["models"][pending] = failed_call
                parse_failures.append(failed_call)
            else:
                parse_failures.append(ModelCall(
                    step=step, messages=(), raw_response=data.get("raw_response", ""),
                    parse_ok=False, parse_error=data.get("error"),
                    prompt_tokens=None, completion_tokens=None,
                    prompt_tokens_exact=None, completion_tokens_exact=None,
                    latency_ms=None))
        elif kind == "step_decision":
            pending = _last_unresolved(bucket["models"])
            if pending is not None:
                bucket["models"][pending] = _replaced(bucket["models"][pending], parse_ok=True)
            if data.get("thought"):
                bucket["thoughts"].append(data["thought"])
        elif kind == "tool_called":
            tool_call_count += 1
            bucket["tools"].append(ToolCall(step=step, tool=data.get("tool", ""),
                                             arguments=dict(data.get("arguments") or {})))
        elif kind == "tool_rejected":
            tool_rejections += 1
            pending = _last_unresolved_tool(bucket["tools"])
            if pending is not None:
                pending_call = bucket["tools"][pending]
                rejected = ToolCall(step=pending_call.step, tool=pending_call.tool,
                                     arguments=pending_call.arguments, rejected=True,
                                     reject_reason=data.get("reason"))
                bucket["tools"][pending] = rejected
            else:
                # the runtime never emits tool_called for a rejected action —
                # the rejection event is self-describing
                rejected = ToolCall(step=step, tool=data.get("tool", ""),
                                     arguments=dict(data.get("arguments") or {}),
                                     rejected=True, reject_reason=data.get("reason"))
                bucket["tools"].append(rejected)
            rejected_tool_calls.append(rejected)
        elif kind == "tool_result":
            tool_latency_total += data.get("latency_ms") or 0.0
            pending = _last_unresolved_tool(bucket["tools"])
            if pending is not None:
                pending_call = bucket["tools"][pending]
                resolved = ToolCall(step=pending_call.step, tool=pending_call.tool,
                                     arguments=pending_call.arguments,
                                     ok=bool(data.get("ok")),
                                     result_content=data.get("content"),
                                     latency_ms=data.get("latency_ms"))
                bucket["tools"][pending] = resolved
            else:
                # a retry re-executes an already-resolved call: record it as a
                # separate attempt carrying the same tool/arguments
                last_call = bucket["tools"][-1]
                resolved = ToolCall(step=step, tool=data.get("tool", last_call.tool),
                                     arguments=dict(last_call.arguments),
                                     ok=bool(data.get("ok")),
                                     result_content=data.get("content"),
                                     latency_ms=data.get("latency_ms"))
                bucket["tools"].append(resolved)
            if resolved.ok:
                tool_successes += 1
                successful_tool_calls.append(resolved)
            else:
                tool_failures += 1
                failed_tool_calls.append(resolved)
        elif kind == "retry_scheduled":
            retry_count += 1
            bucket["retries"].append(Retry(step=step, attempt=data.get("attempt"),
                                            delay_s=data.get("delay"), reason=data.get("reason")))
        elif kind == "finished":
            answer = data.get("answer")

    step_views = tuple(StepView(
        step=number,
        expressed_skills=tuple(bucket["expressed"]),
        thoughts=tuple(bucket["thoughts"]),
        model_calls=tuple(bucket["models"]),
        tool_calls=tuple(bucket["tools"]),
        retries=tuple(bucket["retries"]),
    ) for number, bucket in sorted(steps.items()) if number >= 1)

    provenance = {"genome_id": context.genome_id if context else None,
                  "model": vars(context.model) if context and context.model else None,
                  "seed": context.seed if context else None,
                  "code_commit": context.code_commit if context else None,
                  "run_id": context.run_id if context else None,
                  "experiment_id": context.experiment_id if context else None,
                  "environment": context.environment if context else None,
                  "benchmark": context.benchmark if context else None,
                  "split": context.split if context else None}

    def _exactness(flags: list[bool]) -> str:
        if not flags:
            return "no_data"
        if all(flags):
            return "exact"
        if not any(flags):
            return "estimated"
        return "mixed"

    prompt_exactness = _exactness(exact_prompts)
    completion_exactness = _exactness(exact_completions)
    if prompt_exactness == "exact" and completion_exactness == "exact":
        exactness = "exact"
    elif prompt_exactness == "estimated" and completion_exactness == "estimated":
        exactness = "estimated"
    else:
        exactness = "mixed"
    metrics = {
        "steps": len(step_views),
        "model_calls": model_call_count,
        "tool_calls": tool_call_count,
        "tool_successes": tool_successes,
        "tool_failures": tool_failures,
        "tool_rejections": tool_rejections,
        "retry_count": retry_count,
        "parse_failures": parse_failure_count,
        "expressed_skill_events": expressed_events,
        "unique_expressed_skills": len(expressed_order),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "model_latency_total_ms": model_latency_total,
        "tool_latency_total_ms": tool_latency_total,
        "terminal_status": status,
    }
    action_cost = {k: metrics[k] for k in
                   ("steps", "model_calls", "tool_calls", "tool_successes",
                    "tool_failures", "tool_rejections", "retry_count", "parse_failures")}

    return NormalizedTrajectory(
        trajectory_id=trajectory_id, task=task or "", status=status, answer=answer,
        outcome=outcome, steps=step_views,
        parse_failures=tuple(parse_failures),
        failed_tool_calls=tuple(failed_tool_calls),
        rejected_tool_calls=tuple(rejected_tool_calls),
        final_successful_path=tuple(successful_tool_calls),
        expressed_skill_ids=tuple(expressed_order),
        token_cost={"prompt": total_prompt, "completion": total_completion,
                     "total": total_prompt + total_completion,
                     "prompt_exactness": prompt_exactness,
                     "completion_exactness": completion_exactness,
                     "exactness": exactness},
        latency={"model_total_ms": model_latency_total, "tool_total_ms": tool_latency_total},
        action_cost=action_cost, metrics=metrics, provenance=provenance,
        event_count=len(events))


def _last_unresolved(models: list[ModelCall]) -> int | None:
    for index in range(len(models) - 1, -1, -1):
        if models[index].parse_ok is None:
            return index
    return None


def _last_unresolved_tool(tools: list[ToolCall]) -> int | None:
    for index in range(len(tools) - 1, -1, -1):
        if tools[index].ok is None and not tools[index].rejected:
            return index
    return None


def _replaced(call: ModelCall, **overrides) -> ModelCall:
    values = vars(call).copy()
    values.update(overrides)
    return ModelCall(**values)


def normalize_loaded(loaded) -> NormalizedTrajectory:
    """Convenience reducer over a LoadedTrajectory (#9 entry point)."""
    return normalize_trajectory(trajectory_id=loaded.trajectory_id,
                                 context=loaded.context, events=loaded.events,
                                 status=loaded.status, outcome=loaded.outcome)


def load_normalized(path, *, allow_incomplete: bool = False) -> NormalizedTrajectory:
    """Stable #9 read API: load + verify + reduce in one call, without
    exposing EventLog internals."""
    from trajectory.storage.jsonl import load_trajectory
    return normalize_loaded(load_trajectory(Path(path), allow_incomplete=allow_incomplete))
