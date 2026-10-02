"""Minimal single-agent execution loop (issue #7 criterion 7; hardened per critic review).

Generic observe → reason/plan → act/tool-call → observe result →
retry/replan → finish cycle over a provider-neutral tool/environment
boundary. No benchmark logic, no ALFWorld code, no trajectory persistence:
the loop only emits ordered events (#8 records them via hooks). All
budgets are explicit; the clock is injectable for deterministic latency
measurement.

Tool boundary (fail closed): the offered tool catalog comes from
`env.list_tools()` and is rendered into the planner prompt; a parsed
action is validated BEFORE the environment is called — unknown tools and
arguments violating the tool's `parameters_schema` never reach the
environment (rejected calls are agent failures, deterministic-wrong, so
they are not retried).

Counter semantics (agent-observable only, ADR-0001 D5; explicit because
#11 consumes them):
- `consecutive_failures` — failed/rejected tool calls in a row (reset on success);
- `failing_tools` — cumulative count of failed/rejected tool calls (NOT distinct tools);
- `repeated_tool_error` — consecutive failures of the SAME tool (reset on
  success or on a different tool failing);
- `steps_without_progress` — steps that ended with no successful tool call
  (reset on any successful tool call).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from jsonschema import Draft202012Validator

from runtime.events import EventLog, RuntimeEvent
from runtime.expression import RuntimeConfig
from runtime.model_adapters import GenerationSettings, ModelAdapter, ModelMessage, ToolSpec
from runtime.policies.decisions import StepDecision, ToolAction
from runtime.policies.errors import PlanParseError
from runtime.policies.planner_react import ReActPlanner
from runtime.policies.retry_backoff import RetryPolicy
from runtime.skills import expressed_skills


class ToolEnvironment(Protocol):
    """Provider-neutral tool boundary; concrete environments (e.g. ALFWorld,
    later) implement this without touching the loop."""

    def list_tools(self) -> list[ToolSpec]: ...

    def call(self, name: str, arguments: dict) -> "ToolObservation": ...


@dataclass(frozen=True)
class ToolObservation:
    ok: bool
    content: str


@dataclass(frozen=True)
class Budgets:
    max_steps: int
    max_total_retries: int
    max_tokens_per_request: int = 512
    max_total_tokens: int | None = None


@dataclass
class RuntimeCounters:
    consecutive_failures: int = 0
    failing_tools: int = 0
    repeated_tool_error: int = 0
    steps_without_progress: int = 0
    last_failed_tool: str | None = None  # internal; not an observable counter

    def as_dict(self) -> dict:
        return {
            "consecutive_failures": self.consecutive_failures,
            "failing_tools": self.failing_tools,
            "repeated_tool_error": self.repeated_tool_error,
            "steps_without_progress": self.steps_without_progress,
        }

    def after_tool(self, tool: str, ok: bool) -> None:
        if ok:
            self.consecutive_failures = 0
            self.repeated_tool_error = 0
            self.last_failed_tool = None
            return
        self.consecutive_failures += 1
        self.failing_tools += 1
        self.repeated_tool_error = self.repeated_tool_error + 1 if tool == self.last_failed_tool else 1
        self.last_failed_tool = tool


@dataclass(frozen=True)
class RunResult:
    status: str            # "finished" | "budget_exhausted"
    answer: str | None
    steps: int
    counters: RuntimeCounters
    events: EventLog


def _validate_action(action: ToolAction,
                     tools: list[ToolSpec]) -> tuple[ToolSpec | None, list[str]]:
    """Fail-closed action validation BEFORE the environment is called:
    the tool must exist and the arguments must satisfy its JSON Schema."""
    spec = next((t for t in tools if t.name == action.name), None)
    if spec is None:
        return None, [f"unknown tool {action.name!r}: not in the offered tool catalog"]
    errors = [e.message for e in Draft202012Validator(spec.parameters_schema).iter_errors(action.arguments)]
    return spec, errors


def run_agent(*, config: RuntimeConfig, model: ModelAdapter, env: ToolEnvironment, task: str,
              budgets: Budgets, hooks: tuple[Callable[[RuntimeEvent], None], ...] = (),
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.time) -> RunResult:
    planner = ReActPlanner(config.policies["cognition.planner"].config)
    retry = RetryPolicy(config.policies["execution.retry_policy"].config)
    log = EventLog(hooks)
    counters = RuntimeCounters()
    tools = list(env.list_tools())

    # heritable planner policy limit vs external loop budget (criterion 2):
    effective_max_steps = min(budgets.max_steps, planner.max_plan_steps)
    log.emit("run_started", task=task, genome_id=config.genome_id,
             max_steps=budgets.max_steps, effective_max_steps=effective_max_steps,
             planner_max_plan_steps=planner.max_plan_steps, tools=[t.name for t in tools])

    observations: list[str] = [task]
    total_retries = 0
    total_tokens = 0

    for step in range(1, effective_max_steps + 1):
        log.emit("step_started", step=step)
        active = expressed_skills(config, counters.as_dict())
        log.emit("skills_expressed", step=step, skills=[s.gene_id for s in active])
        messages = planner.build_messages(task=task, observations=observations, skills=active,
                                          tools=tools, step=step, max_steps=effective_max_steps)

        # -- plan (parse failures are retried via the retry policy) ----------
        attempts = 0
        decision = None
        while True:
            settings = GenerationSettings(temperature=0.0, max_tokens=budgets.max_tokens_per_request)
            started = clock()
            result = model.generate(messages, settings)
            model_latency_ms = (clock() - started) * 1000.0
            completion_tokens = result.completion_tokens \
                if result.completion_tokens is not None else model.estimate_tokens(result.text)
            total_tokens += completion_tokens
            log.emit("model_called", step=step, call_kind="generate",
                     messages=[{"role": m.role, "content": m.content} for m in messages],
                     raw_response=result.text,
                     prompt_tokens=result.prompt_tokens, completion_tokens=completion_tokens,
                     total_tokens=total_tokens, latency_ms=model_latency_ms,
                     temperature=settings.temperature, max_tokens=settings.max_tokens)
            if budgets.max_total_tokens is not None and total_tokens > budgets.max_total_tokens:
                log.emit("budget_exhausted", step=step, budget="max_total_tokens", used=total_tokens)
                return RunResult("budget_exhausted", None, step, counters, log)
            try:
                decision = planner.parse(result.text)
            except PlanParseError as exc:  # fail closed; the retry policy decides reprompting
                log.emit("plan_parse_failed", step=step, error=str(exc), raw_response=result.text)
                attempts += 1
                total_retries += 1
                if not retry.should_retry(attempts) or total_retries > budgets.max_total_retries:
                    log.emit("budget_exhausted", step=step, budget="max_total_retries",
                             retries=total_retries)
                    return RunResult("budget_exhausted", None, step, counters, log)
                delay = retry.compute_delay(attempts)
                log.emit("retry_scheduled", step=step, attempt=attempts, delay=delay, reason="plan_parse")
                sleep(delay)
                messages = [*messages,
                            ModelMessage(role="assistant", content=result.text),
                            ModelMessage(role="user",
                                         content="Your previous reply was not valid ReAct format. "
                                                 "Reply again using exactly one required format.")]
                continue
            log.emit("step_decision", step=step, thought=decision.thought,
                     final=decision.final_answer is not None,
                     tool=decision.action.name if decision.action else None)
            break

        if decision.final_answer is not None:
            log.emit("finished", step=step, answer=decision.final_answer)
            return RunResult("finished", decision.final_answer, step, counters, log)

        # -- act (fail-closed validation BEFORE the environment is called) ---
        action = decision.action
        spec, errors = _validate_action(action, tools)
        if spec is None or errors:
            reason = "unknown_tool" if spec is None else "invalid_arguments"
            log.emit("tool_rejected", step=step, tool=action.name, reason=reason, errors=errors,
                     arguments=action.arguments)
            counters.after_tool(action.name, False)  # deterministic-wrong call: not retried
            counters.steps_without_progress += 1
            observations.append(f"rejected tool call {action.name!r}: {'; '.join(errors)}")
            continue

        log.emit("tool_called", step=step, tool=action.name, arguments=action.arguments)
        tool_attempts = 0
        while True:
            started = clock()
            observation = env.call(action.name, action.arguments)
            tool_latency_ms = (clock() - started) * 1000.0
            log.emit("tool_result", step=step, tool=action.name, ok=observation.ok,
                     content=observation.content, latency_ms=tool_latency_ms)
            counters.after_tool(action.name, observation.ok)
            if observation.ok:
                break
            tool_attempts += 1
            if retry.should_retry(tool_attempts) and total_retries < budgets.max_total_retries:
                total_retries += 1
                delay = retry.compute_delay(tool_attempts)
                log.emit("retry_scheduled", step=step, attempt=tool_attempts, delay=delay,
                         reason="tool_failure")
                sleep(delay)
                continue
            break

        if observation.ok:
            counters.steps_without_progress = 0
        else:
            counters.steps_without_progress += 1
        observations.append(observation.content)

    exhausted_budget = "planner_max_plan_steps" if effective_max_steps < budgets.max_steps else "max_steps"
    log.emit("budget_exhausted", step=effective_max_steps, budget=exhausted_budget,
             max_steps=budgets.max_steps, effective_max_steps=effective_max_steps)
    return RunResult("budget_exhausted", None, effective_max_steps, counters, log)
