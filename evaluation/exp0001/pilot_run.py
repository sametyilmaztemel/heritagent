"""The EXP-0001 calibration pilot, run (issue #13 Phase B; ADR-0003
"Experiment needed"; ADR-0007 E2; the frozen pilot manifest).

G0 plays every pilot game once at the evaluation settings (temperature 0,
no seed). Then the first game of each family (game_id order) runs a second
time, and the arm-B rendering smoke test plays the first pilot game once
with one fixed synthetic skill rendered into G0's prompt. The pilot
measures; it tunes nothing. Its ledger and trajectories live in their own
directory, apart from any EXP-0001 run, and it plays no EXP-0001 pool game.

The report holds the frozen measures (``PILOT_MEASURES``) and a projection
of the EXP-0001 cost from them. The projection is a model estimate (status
M), not a measurement: the lifetime runs at temperature 0.7, which the
pilot does not measure, and the gate's size depends on how many candidates
the miner proposes and how many become eligible."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from math import comb
from pathlib import Path

from evaluation.benchmarks.alfworld.splits import ALFWORLD_TASK_TYPES
from evaluation.exp0001.episodes import EpisodeExecutor, EpisodeResult, EpisodeSpec
from evaluation.exp0001.pilot import PILOT_MEASURES
from evaluation.exp0001.protocol import (
    CANDIDATE_CAP,
    CAPS,
    LIFETIME_ATTEMPTS_PER_TASK,
    MAX_TOKENS_PER_REQUEST,
    MINING_SEEDS,
    T_REG_SIZE,
    max_model_calls_per_episode,
)
from evaluation.exp0001.provenance import canonical_sha256
from inheritance.overlay import evaluation_settings
from runtime.expression import RuntimeConfig, RuntimeSkill, validate_runtime_skill_payload
from trajectory.normalization import load_normalized

PILOT_REPORT_SCHEMA = "heritagent/exp0001-pilot-report/0.1"
PILOT_RUN_ID = "EXP-0001-calibration-pilot"
SMOKE_GENE_ID = "pilot_smoke_skill"  # never mined, never in a genome
SMOKE_SKILL = {"name": "search likely places first",
               "principle": "Before taking or using an object, look in the receptacles where it "
                            "is most likely to be, and open the closed ones.",
               "when_to_apply": "household tasks that need an object you cannot see yet",
               "procedure": []}

# gate episodes per candidate (inheritance.stages): replay 1 + 4 tasks, ablation
# 20, generalization 10 + 10, each with and without the trait; regression 40
# with it only; interaction 10 tasks per eligible pair, both and neither
GATE_EPISODES_PER_CANDIDATE = 2 * 5 + 2 * 20 + 2 * 20 + T_REG_SIZE
INTERACTION_EPISODES_PER_PAIR = 2 * 10


def smoke_config(g0_config: RuntimeConfig) -> RuntimeConfig:
    payload = validate_runtime_skill_payload(dict(SMOKE_SKILL))
    skill = RuntimeSkill(gene_id=SMOKE_GENE_ID, name=payload["name"],
                         principle=payload["principle"], when_to_apply=payload["when_to_apply"],
                         procedure=tuple(payload.get("procedure") or ()))
    return RuntimeConfig(genome_id=g0_config.genome_id, policies=g0_config.policies,
                         skills=g0_config.skills + (skill,),
                         regulation=dict(g0_config.regulation))


def pilot_specs(pilot_manifest: Mapping, g0_config: RuntimeConfig) -> dict[str, list[EpisodeSpec]]:
    games = sorted(pilot_manifest["games"], key=lambda g: g["game_id"])
    settings = evaluation_settings(max_tokens=MAX_TOKENS_PER_REQUEST, seed=None)

    def spec(kind: str, game: Mapping, config: RuntimeConfig) -> EpisodeSpec:
        return EpisodeSpec(key=f"pilot/{kind}/{game['game_id']}", task_id=game["game_id"],
                           family=game["family"], split="train", config=config,
                           settings=settings, tags={"stage": "pilot", "pass": kind})

    firsts = [next(g for g in games if g["family"] == family) for family in ALFWORLD_TASK_TYPES]
    return {"run": [spec("run", g, g0_config) for g in games],
            "repeat": [spec("repeat", g, g0_config) for g in firsts],
            "smoke": [spec("arm-b-smoke", games[0], smoke_config(g0_config))]}


def trajectory_fingerprint(path: str) -> str:
    """What two runs of the same episode must share: every model reply and
    every tool call with its result, the status, the answer and the outcome."""
    trajectory = load_normalized(path)
    steps = [[[call.raw_response for call in step.model_calls],
              [[call.tool, call.arguments, call.rejected, call.ok, call.result_content]
               for call in step.tool_calls]]
             for step in trajectory.steps]
    return canonical_sha256({"status": trajectory.status, "answer": trajectory.answer,
                             "success": (trajectory.outcome or {}).get("success"),
                             "steps": steps})


def _fraction(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _summary(values: Sequence[float]) -> dict:
    values = list(values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def measure(runs: Sequence[EpisodeResult], repeats: Sequence[EpisodeResult],
            smoke: EpisodeResult, *, elapsed_seconds: float, workers: int) -> dict:
    calls = [r.model_calls for r in runs]
    successes = [r for r in runs if r.success]
    fresh = [r for r in runs if not r.resumed]
    by_game = {r.task_id: r for r in runs}
    determinism = []
    for repeat in repeats:
        original = by_game[repeat.task_id]
        determinism.append({"game_id": repeat.task_id, "family": repeat.family,
                            "identical": trajectory_fingerprint(original.trajectory_path)
                            == trajectory_fingerprint(repeat.trajectory_path),
                            "success": [original.success, repeat.success]})
    smoke_trajectory = load_normalized(smoke.trajectory_path)
    first_call = smoke_trajectory.steps[0].model_calls[0] if smoke_trajectory.steps and \
        smoke_trajectory.steps[0].model_calls else None
    in_prompt = first_call is not None and any(
        SMOKE_SKILL["principle"] in message.get("content", "") for message in first_call.messages)
    smoke_checks = {"expressed": SMOKE_GENE_ID in smoke.expressed_gene_ids,
                    "skill_in_prompt": in_prompt,
                    "first_reply_parsed": bool(first_call is not None and first_call.parse_ok),
                    "parse_failures": smoke.parse_failures}
    measures = {
        "g0_success_rate_p0": _fraction(len(successes), len(runs)),
        "invalid_command_rate": _fraction(sum(r.invalid_commands for r in runs),
                                          sum(r.steps for r in runs)),
        "parse_failure_rate": _fraction(sum(r.parse_failures for r in runs), sum(calls)),
        "retry_rate": _fraction(sum(r.retries for r in runs), sum(calls)),
        "steps_to_success": _summary([r.steps for r in successes]),
        "model_calls_per_episode": {**_summary(calls), "at_worst_case": sum(
            c >= max_model_calls_per_episode() for c in calls)},
        "throughput_episodes_per_hour": {
            "episodes": len(fresh), "seconds": elapsed_seconds, "workers": workers,
            "value": len(fresh) / (elapsed_seconds / 3600) if fresh and elapsed_seconds > 0
            else None},
        "arm_b_rendering_smoke_test": {
            **smoke_checks, "passed": smoke_checks["expressed"] and in_prompt
            and smoke_checks["first_reply_parsed"]},
        "temperature_0_repeat_determinism": {
            "games": determinism, "all_identical": all(d["identical"] for d in determinism)},
    }
    assert tuple(measures) == PILOT_MEASURES
    prompt_tokens = [call.prompt_tokens for r in runs
                     for step in load_normalized(r.trajectory_path).steps
                     for call in step.model_calls if call.prompt_tokens is not None]
    extra = {
        "per_family_success": {f: _fraction(sum(r.success for r in runs if r.family == f),
                                            sum(1 for r in runs if r.family == f))
                               for f in ALFWORLD_TASK_TYPES},
        "prompt_tokens_per_call": _summary(prompt_tokens),
        "tokens_per_episode": _summary([r.prompt_tokens + r.completion_tokens for r in runs]),
        "trajectory_bytes_per_episode": _summary([Path(r.trajectory_path).stat().st_size
                                                  for r in runs]),
        "statuses": {s: sum(r.status == s for r in runs) for s in sorted({r.status for r in runs})},
    }
    return {"measures": measures, "extra": extra}


def project_exp0001(measured: Mapping, *, t_mine_size: int, t_eval_size: int,
                    transfer_size: int, candidates_per_seed: Sequence[int] = (0, 2, 5, CANDIDATE_CAP),
                    caps: Mapping = CAPS) -> dict:
    """The EXP-0001 episode count per number of candidates per seed (none
    eligible for an interaction pair, and every pair eligible), with the
    calls and active hours at the pilot's means. A model estimate (M)."""
    seeds = len(MINING_SEEDS)
    lifetime = seeds * t_mine_size * LIFETIME_ATTEMPTS_PER_TASK
    final = (1 + 2 * seeds) * (t_eval_size + T_REG_SIZE + transfer_size)
    teacher_calls_max = seeds * t_mine_size  # one call per mining pair at most
    calls_mean = measured["measures"]["model_calls_per_episode"]["mean"]
    throughput = measured["measures"]["throughput_episodes_per_hour"]["value"]
    scenarios = []
    for k in candidates_per_seed:
        for pairs, label in ((0, "no interaction pair"), (comb(k, 2), "every pair eligible")):
            episodes = lifetime + final + seeds * (k * GATE_EPISODES_PER_CANDIDATE
                                                   + pairs * INTERACTION_EPISODES_PER_PAIR)
            calls = None if calls_mean is None else round(episodes * calls_mean
                                                          + teacher_calls_max)
            hours = None if not throughput else episodes / throughput
            scenarios.append({
                "candidates_per_seed": k, "interaction": label, "episodes": episodes,
                "model_calls_at_pilot_mean": calls, "active_hours_at_pilot_throughput": hours,
                "within_caps": {
                    "episodes": episodes <= caps["episodes_max"],
                    "model_calls": None if calls is None else calls <= caps["model_calls_max"],
                    "hours": None if hours is None else hours <= caps["wall_clock_hours_max"]}})
            if k < 2:
                break  # no pair is possible
    return {"status": "M (model estimate under the stated assumptions; not a measurement)",
            "assumptions": [
                "calls per episode and throughput as measured by the pilot at temperature 0; "
                "the lifetime runs at 0.7, which the pilot does not measure",
                "the throughput at the pilot's worker count holds for the whole run",
                "teacher calls at their maximum: one per T_mine task and seed"],
            "fixed_episodes": {"lifetime": lifetime, "final_evaluation": final},
            "gate_episodes_per_candidate": GATE_EPISODES_PER_CANDIDATE,
            "interaction_episodes_per_pair": INTERACTION_EPISODES_PER_PAIR,
            "caps": dict(caps), "scenarios": scenarios}


def run_pilot(executor: EpisodeExecutor, pilot_manifest: Mapping, g0_config: RuntimeConfig, *,
              clock: Callable[[], float] = time.monotonic) -> dict:
    """Run (or resume) the pilot and measure it."""
    specs = pilot_specs(pilot_manifest, g0_config)
    started = clock()
    runs = executor.run_many(specs["run"])
    elapsed = clock() - started
    repeats = executor.run_many(specs["repeat"])
    smoke = executor.run_many(specs["smoke"])[0]
    return measure(runs, repeats, smoke, elapsed_seconds=elapsed, workers=executor.workers)
