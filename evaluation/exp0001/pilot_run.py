"""The EXP-0001 calibration pilot, run (issue #13 Phase B; ADR-0003
"Experiment needed"; ADR-0007 E2; the frozen pilot manifest).

G0 plays every pilot game once at the evaluation settings (temperature 0,
no seed). Then the first game of each family (game_id order) runs a second
time, the arm-B rendering smoke test plays the first pilot game once with
one fixed synthetic skill rendered into G0's prompt, and every pilot game
runs 3 more times at the mining temperature 0.7 (ADR-0009 D3), which is
what the lifetime's pairs and T_reg's eligibility turn on. The pilot
measures; it tunes nothing. Its ledger and trajectories live in their own
directory, apart from any EXP-0001 run, and it plays no EXP-0001 pool game.

The report holds the frozen measures (``PILOT_MEASURES``) and a projection
of the EXP-0001 cost from them. The projection is a model estimate (status
M), not a measurement: its calls per episode and throughput are those of
the temperature-0 run pass (the temperature-0.7 attempts inform the pair
yield, T_reg's eligibility and the prompt sizes, not these two means), and
the gate's size depends on how many candidates the miner proposes and how
many become eligible."""

from __future__ import annotations

import json
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
    CONTEXT_WINDOW,
    LIFETIME_ATTEMPTS_PER_TASK,
    MAX_TOKENS_PER_REQUEST,
    MINER_CALLS_PER_PAIR_MAX,
    MINER_MAX_TOKENS,
    MINING_SEEDS,
    MINING_TEMPERATURE,
    T_REG_SIZE,
    attempt_sampling_seed,
    max_model_calls_per_episode,
)
from evaluation.exp0001.provenance import canonical_sha256
from inheritance.overlay import evaluation_settings
from runtime.expression import RuntimeConfig, RuntimeSkill, validate_runtime_skill_payload
from runtime.model_adapters.base import GenerationSettings
from trajectory.normalization import load_normalized
from traits.miner.evidence import EVIDENCE_FORMS, FULL, render_pair_evidence
from traits.miner.inputs import MiningPair
from traits.miner.miner import render_teacher_prompt

PILOT_REPORT_SCHEMA = "heritagent/exp0001-pilot-report/0.1"
PILOT_RUN_ID = "EXP-0001-calibration-pilot"
PILOT_ATTEMPT_SEED = 0  # the E1 attempt-seed rule with pilot seed 0 (ADR-0009 D3)
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

    def attempt(game: Mapping, index: int) -> EpisodeSpec:
        from runtime.policies.planner_react import STOP_SEQUENCES
        return EpisodeSpec(
            key=f"pilot/t07/{game['game_id']}/{index}", task_id=game["game_id"],
            family=game["family"], split="train", config=g0_config,
            settings=GenerationSettings(
                temperature=MINING_TEMPERATURE, max_tokens=MAX_TOKENS_PER_REQUEST,
                seed=attempt_sampling_seed(PILOT_ATTEMPT_SEED, game["game_id"], index),
                stop=STOP_SEQUENCES),
            tags={"stage": "pilot", "pass": "t07", "attempt": index})

    return {"run": [spec("run", g, g0_config) for g in games],
            "repeat": [spec("repeat", g, g0_config) for g in firsts],
            "smoke": [spec("arm-b-smoke", games[0], smoke_config(g0_config))],
            "t07": [attempt(g, a) for g in games for a in range(LIFETIME_ATTEMPTS_PER_TASK)]}


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


def _prompt_tokens(results: Sequence[EpisodeResult]) -> list[int]:
    return [call.prompt_tokens for r in results
            for step in load_normalized(r.trajectory_path).steps
            for call in step.model_calls if call.prompt_tokens is not None]


def miner_prompt_tokens(t07: Sequence[EpisodeResult],
                        count_prompt_tokens: Callable[[str], int] | None) -> dict:
    """ADR-0009 D8: each temperature-0.7 pair's teacher prompt, in both
    evidence forms, against the window less the miner's output tokens."""
    by_game: dict[str, list[EpisodeResult]] = {}
    for result in sorted(t07, key=lambda r: r.key):  # attempt order within a game
        by_game.setdefault(result.task_id, []).append(result)
    limit = CONTEXT_WINDOW - MINER_MAX_TOKENS
    counts: dict[str, list[int]] = {form: [] for form in EVIDENCE_FORMS}
    fits = {"full": 0, "only_without_admissible_commands": 0, "neither": 0}
    pairs = 0
    for game in sorted(by_game):
        attempts = by_game[game]
        success = next((r for r in attempts if r.success), None)
        failure = next((r for r in attempts if not r.success), None)
        if success is None or failure is None:
            continue
        pairs += 1
        if count_prompt_tokens is None:
            continue
        pair = MiningPair(success=load_normalized(success.trajectory_path),
                          failure=load_normalized(failure.trajectory_path),
                          task_family=success.family)
        tokens = {}
        for form in EVIDENCE_FORMS:
            evidence, _ = render_pair_evidence(pair, form)
            tokens[form] = count_prompt_tokens(render_teacher_prompt(  # as the run's miner:
                evidence_text=evidence, family_universe=tuple(sorted(ALFWORLD_TASK_TYPES)),
                max_proposals=CANDIDATE_CAP))  # MiningBatch sorts the universe
            counts[form].append(tokens[form])
        fit = {form: n <= limit for form, n in tokens.items()}
        fits["full" if fit[FULL] else "only_without_admissible_commands"
             if any(fit.values()) else "neither"] += 1
    counted = count_prompt_tokens is not None
    return {"pairs": pairs, "counted": counted, "prompt_limit": limit,
            "context_window": CONTEXT_WINDOW, "miner_max_tokens": MINER_MAX_TOKENS,
            **{form.replace("-", "_"): _summary(counts[form]) if counted else None
               for form in EVIDENCE_FORMS},
            "pairs_that_fit": fits if counted else None}


def measure(runs: Sequence[EpisodeResult], repeats: Sequence[EpisodeResult],
            smoke: EpisodeResult, t07: Sequence[EpisodeResult], *, elapsed_seconds: float,
            workers: int, count_prompt_tokens: Callable[[str], int] | None = None,
            timing: Mapping | None = None) -> dict:
    """``timing`` is the run pass's {episodes, seconds} over every session
    that played part of it; without it, this session's fresh episodes."""
    calls = [r.model_calls for r in runs]
    successes = [r for r in runs if r.success]
    fresh = [r for r in runs if not r.resumed]
    timing = timing or {"episodes": len(fresh), "seconds": elapsed_seconds}
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
    attempts: dict[str, list[bool]] = {}
    for result in t07:
        attempts.setdefault(result.task_id, []).append(result.success)
    solved_any = sorted(g for g, outcomes in attempts.items() if any(outcomes))
    paired = [g for g, outcomes in attempts.items() if any(outcomes) and not all(outcomes)]
    all_prompts = _prompt_tokens([*runs, *repeats, smoke, *t07])
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
            "episodes": timing["episodes"], "seconds": timing["seconds"], "workers": workers,
            "value": timing["episodes"] / (timing["seconds"] / 3600)
            if timing["episodes"] and timing["seconds"] > 0 else None},
        "arm_b_rendering_smoke_test": {
            **smoke_checks, "passed": smoke_checks["expressed"] and in_prompt
            and smoke_checks["first_reply_parsed"]},
        "temperature_0_repeat_determinism": {
            "games": determinism, "all_identical": all(d["identical"] for d in determinism)},
        "temperature_0_7_attempt_success_rate": _fraction(sum(r.success for r in t07), len(t07)),
        "temperature_0_7_pair_yield": _fraction(len(paired), len(attempts)),
        "temperature_0_7_solved_in_any_attempt": _fraction(len(solved_any), len(attempts)),
        "temperature_0_success_among_0_7_solved": _fraction(
            sum(by_game[g].success for g in solved_any), len(solved_any)),
        "prompt_tokens_per_call_max": {
            "value": max(all_prompts) if all_prompts else None,
            "context_window": CONTEXT_WINDOW,
            # a ReAct call fits if its prompt plus its 512-token request budget do (D4)
            "prompt_limit": CONTEXT_WINDOW - MAX_TOKENS_PER_REQUEST,
            "context_overflows": sum(r.exhausted_budget == "context_window"
                                     for r in [*runs, *repeats, smoke, *t07])},
        "loop_steps_per_episode": {**_summary([r.loop_steps for r in runs]),
                                   "game_steps": _summary([r.steps for r in runs])},
        "miner_prompt_tokens_t07_pairs": miner_prompt_tokens(t07, count_prompt_tokens),
    }
    assert tuple(measures) == PILOT_MEASURES
    prompt_tokens = _prompt_tokens(runs)
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
    # one pair per T_mine task at most, two calls per pair at most (ADR-0009 D8)
    teacher_calls_max = seeds * t_mine_size * MINER_CALLS_PER_PAIR_MAX
    measures = measured["measures"]
    pair_yield = measures.get("temperature_0_7_pair_yield", {}).get("value")
    solved_any = measures.get("temperature_0_7_solved_in_any_attempt", {}).get("value")
    calls_mean = measures["model_calls_per_episode"]["mean"]
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
                "calls per episode and throughput are those of the pilot's temperature-0 run "
                "pass; the temperature-0.7 attempts inform the pair yield, T_reg's eligibility "
                "and the prompt sizes, not these two means, though the lifetime runs at 0.7",
                "the throughput at the pilot's worker count holds for the whole run",
                "teacher calls at their maximum: two per T_mine task and seed"],
            "fixed_episodes": {"lifetime": lifetime, "final_evaluation": final},
            "gate_episodes_per_candidate": GATE_EPISODES_PER_CANDIDATE,
            "interaction_episodes_per_pair": INTERACTION_EPISODES_PER_PAIR,
            "caps": dict(caps), "scenarios": scenarios,
            "lifetime": {
                "pairs_per_seed_at_pilot_yield": None if pair_yield is None
                else round(t_mine_size * pair_yield, 1),
                "t_reg_eligible_lower_bound": None if solved_any is None
                else round(t_mine_size * solved_any, 1),
                "t_reg_size": T_REG_SIZE,
                "note": "pairs: T_mine size times the pilot's 3-attempt pair yield; eligible: "
                        "T_mine size times the pilot's solved-in-any-of-3 rate, a lower bound "
                        "for the lifetime's 9 attempts per task (ADR-0007 E3)"}}


def run_pilot(executor: EpisodeExecutor, pilot_manifest: Mapping, g0_config: RuntimeConfig, *,
              clock: Callable[[], float] = time.monotonic,
              count_prompt_tokens: Callable[[str], int] | None = None,
              timing_path: Path | None = None) -> dict:
    """Run (or resume) the pilot and measure it. ``count_prompt_tokens``
    counts a one-system-message prompt as the server would (the CLI passes
    the model tokenizer's); without it the miner prompts are not counted.
    ``timing_path`` keeps the run pass's timing across sessions (it lives in
    the pilot directory, so the mirror keeps it): a pilot resumed after its
    run pass still reports the throughput that pass was played at."""
    specs = pilot_specs(pilot_manifest, g0_config)
    started = clock()
    runs = executor.run_many(specs["run"])
    elapsed = clock() - started
    timing = None
    if timing_path is not None:
        timing_path = Path(timing_path)
        sessions = json.loads(timing_path.read_text(encoding="utf-8"))["sessions"] \
            if timing_path.exists() else []
        fresh = sum(not r.resumed for r in runs)
        if fresh:  # this session played part of the run pass
            sessions.append({"episodes": fresh, "seconds": elapsed,
                             "workers": executor.workers})
            tmp = timing_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"sessions": sessions}, indent=2), encoding="utf-8")
            tmp.replace(timing_path)
        timing = {"episodes": sum(s["episodes"] for s in sessions),
                  "seconds": sum(s["seconds"] for s in sessions), "sessions": sessions}
    repeats = executor.run_many(specs["repeat"])
    smoke = executor.run_many(specs["smoke"])[0]
    t07 = executor.run_many(specs["t07"])
    return measure(runs, repeats, smoke, t07, elapsed_seconds=elapsed, workers=executor.workers,
                   count_prompt_tokens=count_prompt_tokens, timing=timing)
