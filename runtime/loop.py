"""Minimal single-agent execution loop (issue #7 criterion 7).

Generic observe → reason/plan → act/tool-call → observe result →
retry/replan → finish cycle over a provider-neutral tool/environment
boundary. No benchmark logic, no ALFWorld code, no trajectory persistence:
the loop only emits ordered events (#8 records them via hooks). All
budgets are explicit.

Counter semantics (agent-observable only, ADR-0001 D5):
- `consecutive_failures` — failed tool calls in a row (reset on success);
- `failing_tools` — cumulative count of failed tool calls;
- `repeated_tool_error` — consecutive failures of the SAME tool (reset on
  success or on a different tool failing);
- `steps_without_progress` — steps that ended with no successful tool call
  (reset on any successful tool call).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from runtime.events import EventLog, RuntimeEvent
from runtime.expression import RuntimeConfig
from runtime.model_adapters import GenerationSettings, ModelAdapter, ModelMessage, ToolSpec
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


def run_agent(*, config: RuntimeConfig, model: ModelAdapter, env: ToolEnvironment, task: str,
              budgets: Budgets, hooks: tuple[Callable[[RuntimeEvent], None], ...] = (),
              sleep: Callable[[float], None] = time.sleep) -> RunResult:
    planner = ReActPlanner(config.policies["cognition.planner"].config)
    retry = RetryPolicy(config.policies["execution.retry_policy"].config)
    log = EventLog(hooks)
    counters = RuntimeCounters()
    log.emit("run_started", genome_id=config.genome_id, max_steps=budgets.max_steps)

    observations: list[str] = [task]
    total_retries = 0
    total_tokens = 0

    for step in range(1, budgets.max_steps + 1):
        log.emit("step_started", step=step)
        active = expressed_skills(config, counters.as_dict())
        log.emit("skills_expressed", step=step, skills=[s.gene_id for s in active])
        messages = planner.build_messages(task=task, observations=observations, skills=active,
                                          step=step, max_steps=budgets.max_steps)

        # -- plan (parse failures are retried via the retry policy) ----------
        attempts = 0
        decision = None
        while True:
            settings = GenerationSettings(temperature=0.0, max_tokens=budgets.max_tokens_per_request)
            result = model.generate(messages, settings)
            completion_tokens = result.completion_tokens \
                if result.completion_tokens is not None else model.estimate_tokens(result.text)
            total_tokens += completion_tokens
            log.emit("model_called", step=step, call_kind="generate",
                     prompt_tokens=result.prompt_tokens, completion_tokens=completion_tokens,
                     temperature=settings.temperature, max_tokens=settings.max_tokens)
            if budgets.max_total_tokens is not None and total_tokens > budgets.max_total_tokens:
                log.emit("budget_exhausted", step=step, budget="max_total_tokens", used=total_tokens)
                return RunResult("budget_exhausted", None, step, counters, log)
            try:
                decision = planner.parse(result.text)
            except PlanParseError as exc:  # fail closed; the retry policy decides reprompting
                log.emit("plan_parse_failed", step=step, error=str(exc))
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

        # -- act (failed tool calls retried via the retry policy) ------------
        action = decision.action
        log.emit("tool_called", step=step, tool=action.name, arguments=action.arguments)
        tool_attempts = 0
        while True:
            observation = env.call(action.name, action.arguments)
            log.emit("tool_result", step=step, tool=action.name, ok=observation.ok,
                     content=observation.content)
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

    log.emit("budget_exhausted", step=budgets.max_steps, budget="max_steps")
    return RunResult("budget_exhausted", None, budgets.max_steps, counters, log)
