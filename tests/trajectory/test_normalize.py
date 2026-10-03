"""Normalizer tests: deterministic reduction, derived metrics, #9 read API
(issue #8 criteria 7, 8, 9, 10, 11) — ScriptedAdapter, no GPU/live model."""

import json

import pytest

from runtime.loop import Budgets, ToolObservation, run_agent
from trajectory.normalization import load_normalized, normalize_loaded, normalize_trajectory
from trajectory.recorder.recorder import TrajectoryRecorder
from trajectory.storage.jsonl import load_trajectory
from tests.runtime.conftest import (
    ScriptedEnv,
    SKILL_GENE_ID,
    react_action,
    react_final,
    seed_with_skill,
)
from tests.trajectory.conftest import SteppedClock, run_captured

GOLDEN_ENV = {"heat_object": [ToolObservation(ok=True, content="The plate is now heated.")]}
GOLDEN_SCRIPT = [react_action("heat_object", {"object": "plate"}, thought="Inspect then heat."),
                 react_final("The plate is heated.", thought="Success reported.")]
GOLDEN_BUDGETS = Budgets(max_steps=4, max_total_retries=10, max_tokens_per_request=64)


def test_normalizer_golden_successful_run(tmp_path, registry, context):
    _, recorder, path, env, _ = run_captured(tmp_path, registry, context, name="golden",
                                              script=GOLDEN_SCRIPT, env_results=GOLDEN_ENV,
                                              budgets=GOLDEN_BUDGETS, clock=SteppedClock())
    recorder.finalize()
    normalized = load_normalized(path)

    assert normalized.trajectory_id.startswith("T-")
    assert normalized.task == "Heat the plate."
    assert normalized.status == "finished"
    assert normalized.answer == "The plate is heated."
    assert len(normalized.steps) == 2
    step1 = normalized.steps[0]
    assert step1.thoughts == ("Inspect then heat.",)
    assert len(step1.model_calls) == 1 and step1.model_calls[0].parse_ok is True
    assert step1.tool_calls[0].tool == "heat_object"
    assert step1.tool_calls[0].arguments == {"object": "plate"}
    assert step1.tool_calls[0].ok is True
    assert step1.tool_calls[0].result_content == "The plate is now heated."
    assert normalized.final_successful_path == step1.tool_calls
    assert normalized.provenance["genome_id"] == "G-0f1e2d3c"
    assert normalized.provenance["model"]["model_id"] == "scripted-test-model"


def test_parse_failure_replan_success_preserved(tmp_path, registry, context):
    script = ["this is not valid ReAct output at all",
              react_action("heat_object", {"object": "plate"}),
              react_final("recovered")]
    _, recorder, path, _, _ = run_captured(tmp_path, registry, context, name="replan",
                                            script=script,
                                            env_results={"heat_object": [ToolObservation(
                                                ok=True, content="ok")]})
    recorder.finalize()
    normalized = load_normalized(path)
    assert len(normalized.parse_failures) == 1
    failed = normalized.parse_failures[0]
    assert failed.parse_ok is False
    assert failed.raw_response == "this is not valid ReAct output at all"
    assert failed.parse_error
    # the failed attempt stays inside its step's model_calls as well
    assert normalized.steps[0].model_calls[0].parse_ok is False
    assert normalized.steps[0].model_calls[1].parse_ok is True


def test_tool_retry_then_success(tmp_path, registry, context):
    script = [react_action("heat_object", {"object": "plate"}),
              react_final("heated after retry")]
    _, recorder, path, _, _ = run_captured(tmp_path, registry, context, name="retry",
                                            script=script,
                                            env_results={"heat_object": [
                                                ToolObservation(ok=False, content="transient"),
                                                ToolObservation(ok=True, content="heated")]})
    recorder.finalize()
    normalized = load_normalized(path)
    assert len(normalized.steps[0].retries) == 1
    assert normalized.steps[0].retries[0].reason == "tool_failure"
    assert len(normalized.failed_tool_calls) == 1
    assert len(normalized.final_successful_path) == 1
    assert normalized.metrics["tool_failures"] == 1
    assert normalized.metrics["tool_successes"] == 1
    assert normalized.metrics["retry_count"] == 1


def test_tool_rejection_preserved(tmp_path, registry, context):
    script = [react_action("nonexistent_tool", {"x": 1}),
              react_final("recovered")]
    _, recorder, path, env, _ = run_captured(tmp_path, registry, context, name="reject",
                                              script=script,
                                              env_results={"heat_object": [ToolObservation(
                                                  ok=True, content="unused")]})
    recorder.finalize()
    normalized = load_normalized(path)
    assert env.calls == []  # environment was never reached
    assert len(normalized.rejected_tool_calls) == 1
    rejected = normalized.rejected_tool_calls[0]
    assert rejected.rejected is True
    assert rejected.reject_reason == "unknown_tool"
    assert rejected.ok is None and rejected.result_content is None


def test_budget_exhaustion_trajectory(tmp_path, registry, context):
    script = [react_action("heat_object", {"object": "plate"}) for _ in range(4)]
    _, recorder, path, _, _ = run_captured(tmp_path, registry, context, name="budget",
                                            script=script,
                                            env_results={"heat_object": [
                                                ToolObservation(ok=True, content="ok")] * 4},
                                            budgets=Budgets(max_steps=2, max_total_retries=0,
                                                             max_tokens_per_request=64))
    recorder.finalize()
    normalized = load_normalized(path)
    assert normalized.status == "budget_exhausted"
    assert normalized.answer is None
    assert normalized.metrics["steps"] == 2


def test_derived_metrics_match_raw_events(tmp_path, registry, context):
    script = ["malformed output",
              react_action("heat_object", {"object": "plate"}),
              react_final("done")]
    env_results = {"heat_object": [ToolObservation(ok=False, content="transient"),
                                    ToolObservation(ok=True, content="ok")]}
    _, recorder, path, _, _ = run_captured(tmp_path, registry, context, name="metrics",
                                            script=script, env_results=env_results)
    recorder.finalize()
    loaded = load_trajectory(path)
    normalized = load_normalized(path)
    raw = loaded.events

    assert normalized.metrics["steps"] == len([e for e in raw if e["kind"] == "step_started"])
    assert normalized.metrics["model_calls"] == len([e for e in raw if e["kind"] == "model_called"])
    assert normalized.metrics["tool_calls"] == len([e for e in raw if e["kind"] == "tool_called"])
    assert normalized.metrics["tool_rejections"] == len([e for e in raw if e["kind"] == "tool_rejected"])
    assert normalized.metrics["retry_count"] == len([e for e in raw if e["kind"] == "retry_scheduled"])
    assert normalized.metrics["parse_failures"] == len([e for e in raw if e["kind"] == "plan_parse_failed"])
    assert normalized.metrics["prompt_tokens"] == sum(e["data"]["prompt_tokens"]
                                                       for e in raw if e["kind"] == "model_called")
    assert normalized.metrics["completion_tokens"] == sum(e["data"]["completion_tokens"]
                                                           for e in raw if e["kind"] == "model_called")
    assert normalized.metrics["total_tokens"] == (normalized.metrics["prompt_tokens"]
                                                   + normalized.metrics["completion_tokens"])
    assert normalized.metrics["model_latency_total_ms"] == pytest.approx(
        sum(e["data"]["latency_ms"] for e in raw if e["kind"] == "model_called"))
    assert normalized.metrics["tool_latency_total_ms"] == pytest.approx(
        sum(e["data"]["latency_ms"] for e in raw if e["kind"] == "tool_result"))
    assert normalized.action_cost["steps"] == normalized.metrics["steps"]


def test_estimated_tokens_are_labeled_not_exact(tmp_path, registry, context):
    # ScriptedAdapter.generate() returns completion_tokens=None -> the loop
    # estimates; the trajectory must NOT label estimates as exact
    script = [react_final("done")]
    _, recorder, path, _, _ = run_captured(tmp_path, registry, context, name="estimate",
                                            script=script, env_results={})
    recorder.finalize()
    normalized = load_normalized(path)
    assert normalized.token_cost["completion_exact"] is False
    assert normalized.token_cost["exact"] is False
    called = normalized.steps[0].model_calls[0]
    assert called.completion_tokens_exact is False


def test_exact_tokens_labeled_when_adapter_reports(tmp_path, registry, context):
    script = [(react_final("done"), 42)]  # (text, completion_tokens)
    _, recorder, path, _, _ = run_captured(tmp_path, registry, context, name="exact",
                                            script=script, env_results={})
    recorder.finalize()
    normalized = load_normalized(path)
    assert normalized.token_cost["completion_exact"] is True
    assert normalized.token_cost["completion"] == 42


def test_expressed_skills_recorded(tmp_path, registry, context):
    from runtime.expression import compile_runtime_config
    genome = seed_with_skill(registry)  # constitutive skill carrying evaluation-only metadata
    config = compile_runtime_config(genome, registry)
    from runtime.model_adapters import ScriptedAdapter
    adapter = ScriptedAdapter([react_final("done")])
    path = tmp_path / "skills.jsonl"
    recorder = TrajectoryRecorder(context, path)
    run_agent(config=config, model=adapter, env=ScriptedEnv({}), task="t",
              budgets=Budgets(max_steps=1, max_total_retries=1, max_tokens_per_request=64),
              hooks=(recorder,), clock=SteppedClock())
    recorder.finalize()
    normalized = load_normalized(path)
    assert normalized.expressed_skill_ids == (SKILL_GENE_ID,)
    assert normalized.steps[0].expressed_skills == (SKILL_GENE_ID,)
    # evaluation-only metadata absent from STORED runtime model messages
    stored = json.dumps(normalized.steps[0].model_calls[0].messages)
    for forbidden in ("applicability", "task_families"):
        assert forbidden not in stored


def test_outcome_round_trip_through_normalizer(tmp_path, registry, context):
    from trajectory.recorder.context import OutcomeAnnotation
    _, recorder, path, _, _ = run_captured(tmp_path, registry, context, name="outcome",
                                            script=GOLDEN_SCRIPT, env_results=GOLDEN_ENV,
                                            clock=SteppedClock())
    recorder.attach_outcome(__import__("trajectory.recorder.context",
                                        fromlist=["OutcomeAnnotation"]).OutcomeAnnotation(
        success=True, reward=1.0, evaluator_id="e", evaluator_version="1"))
    recorder.finalize()
    normalized = load_normalized(path)
    assert normalized.outcome["success"] is True
    assert normalized.outcome["reward"] == 1.0


def test_normalizer_is_deterministic(tmp_path, registry, context):
    normalized_runs = []
    for name in ("a", "b"):
        _, recorder, _, _, _ = run_captured(tmp_path / name, registry, context, name=name,
                                             script=GOLDEN_SCRIPT, env_results=GOLDEN_ENV,
                                             clock=SteppedClock())
        recorder.finalize()
        path = tmp_path / name / f"{name}.jsonl"
        normalized_runs.append(load_normalized(path))
    from dataclasses import asdict
    assert asdict(normalized_runs[0]) == asdict(normalized_runs[1])
