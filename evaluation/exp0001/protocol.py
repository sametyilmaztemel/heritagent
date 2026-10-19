"""The frozen EXP-0001 run protocol (ADR-0007; owner decisions E1-E8 of
2026-10-14, as amended by ADR-0008 D2-D3 and ADR-0012, owner-accepted
2026-10-19; plus the locked manifest values it builds on).

Everything the orchestrator decides by rule is fixed here, before any
episode, and recorded in the run-start record."""

from __future__ import annotations

import hashlib
from types import MappingProxyType

from runtime.seed_g0 import EPISODE_MAX_STEPS

MINING_SEEDS = (11, 23, 41)  # manifest evaluation.mining_seeds
LIFETIME_ATTEMPTS_PER_TASK = 3  # E1
MINING_TEMPERATURE = 0.7  # manifest model.mining_temperature
T_REG_SIZE = 40
T_REG_SEED = 0  # E3
CANDIDATE_CAP = 10  # manifest genome.candidate_set.cap (K)
MAX_TOKENS_PER_REQUEST = 512
MAX_TOTAL_RETRIES = 10
CONTEXT_WINDOW = 32768  # ADR-0009 D4, kept by ADR-0011 D5 (Qwen3.8-27B's native length is 262,144)
# ADR-0011 D3: thinking off in every request, the server's default and the token counter
CHAT_TEMPLATE_KWARGS = MappingProxyType({"enable_thinking": False})
MINER_MAX_TOKENS = 2048  # traits.miner.MiningSettings default
# ADR-0009 D8: a pair's teacher prompt in its full evidence form, then without the
# admissible-command lists; the cap check reserves both calls, the run records the forms sent
MINER_CALLS_PER_PAIR_MAX = 2
# ADR-0012 D9: 14,500 episodes and 507,500 calls (35 per episode); 96 active hours
CAPS = MappingProxyType({"episodes_max": 14_500, "model_calls_max": 507_500,
                         "wall_clock_hours_max": 96, "accelerator_nodes_max": 1})
# the caps the calibration pilot's ledgers were opened with (before ADR-0012)
PILOT_CAPS = MappingProxyType({"episodes_max": 7000, "model_calls_max": 225_000,
                               "wall_clock_hours_max": 96, "accelerator_nodes_max": 1})
# the committed pre-run artifacts (manifest run_protocol); a run binds to them
LOCKED_SPLIT_MANIFEST_SHA256 = "2f74d1882c1a0032dae3bed3bbd0b9cbc3bdd6bed7496522f984bf4028d1ed9c"
LOCKED_PILOT_SHA256 = "4fc2b6fe1766dec7bad7795a2f0169fa354b67baab4a569c3d1fcb92700daa52"
LOCKED_BENEFIT_BANK_SHA256 = "fd44bb361e7e0176f281f14b75e7381f4bbf68a0b7b73eee33d50b44323ca545"
GATE_ROUNDS = 6  # ADR-0012 D2
# ADR-0012 D9: the final evaluation's largest count, 7 distinct genomes (A once, B and C per
# seed) x (134 T_eval + 40 T_reg + 20 transfer games)
FINAL_EVALUATION_RESERVE = 7 * (134 + 40 + 20)
ROUND_EPISODES_MAX = 960  # ADR-0012 D9: a round's largest count, K = 10 (descriptive)

PROTOCOL = MappingProxyType({
    "mining_seeds": MINING_SEEDS,
    "lifetime_attempts_per_task": LIFETIME_ATTEMPTS_PER_TASK,
    "lifetime_order": "t_mine tasks in game_id order; attempts 0..2 per task",
    "mining_temperature": MINING_TEMPERATURE,
    "attempt_sampling_seed": "int(sha256('heritagent-attempt|<mining_seed>|<task_id>|<attempt>')"
                             "[:8], 16) mod 2**31",
    "pairing": "per task: the first success and the first failure, in attempt order",
    "t_reg": {"eligible": "t_mine tasks G0 solved in at least one lifetime episode of any seed",
              "size": T_REG_SIZE, "seed": T_REG_SEED,
              "draw": "the 40 eligible tasks with the smallest "
                      "sha256('heritagent-treg|<seed>|<task_id>')",
              "baseline": "the task's earliest success (seed order, then attempt order)",
              "shortfall": "fewer than 40 eligible fails closed"},
    "transfer_subset": "the 20-task ablation pool (since ADR-0012 it selects nothing)",
    "benefit_bank": {"file": "EXP-0001-benefit-bank.json", "sha256": LOCKED_BENEFIT_BANK_SHA256,
                     "per_family": 192},
    "candidate_cap": CANDIDATE_CAP,
    "candidate_ranking": ("miner-estimated-generality", "discovery-order"),
    # one source for the step budget: the loop, the game's own limit and the record
    "budgets": {"max_steps": EPISODE_MAX_STEPS, "max_total_retries": MAX_TOTAL_RETRIES,
                "max_tokens_per_request": MAX_TOKENS_PER_REQUEST},
    "context_window": CONTEXT_WINDOW,
    "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    "context_overflow": "a model call whose prompt tokens plus its max_tokens exceed the "
                        "context window ends the episode with status budget_exhausted "
                        "(budget context_window), a failure in every arm; the prompt is never "
                        "truncated and the call never retried; the rejected call generated "
                        "nothing and is not counted among the episode's model calls "
                        "(ADR-0009 D4)",
    "caps": CAPS,
    "cap_rule": "before each episode: stop if episodes + episodes in flight + 1 > "
                "episodes_max, if model_calls + the reservations in flight + (max_steps + "
                "max_total_retries) > model_calls_max, or if the active run time exceeds "
                "wall_clock_hours_max; a stopped run is incomplete; the miner's teacher calls "
                "count against model_calls_max",
    # ADR-0012 D9
    "reserve_rule": "before each gate round: stop, marked incomplete, if episodes + the "
                    "round's unrecorded episodes + final_evaluation_reserve > episodes_max, or "
                    "if model_calls + (the round's unrecorded episodes + "
                    "final_evaluation_reserve) x the run's mean calls per recorded episode > "
                    "model_calls_max; each check is a ledger record. The round's unrecorded "
                    "episodes are its largest possible count given the candidates still in it "
                    "(no request of a round depends on that round's own outcomes) less those "
                    "already recorded, which episodes counts",
    "final_evaluation_reserve": FINAL_EVALUATION_RESERVE,
    "gate_rounds": {"count": GATE_ROUNDS,
                    "R1": "stage 1 and stage-2 look 1", "R2-R4": "stage-2 looks 2-4",
                    "R5": "stage-3 out-of-scope and 4A", "R6": "4C pairs",
                    "cells": "one G0 execution per (seed, round, game), played lazily, read only "
                             "in its round",
                    "queue": "a round's with-trait episodes and cells in ascending "
                             "sha256 of their ledger keys"},
    "wall_clock_basis": "active run time: the sum of the ledger sessions' spans, each from "
                        "the process opening the ledger to its last record (ADR-0008 D2)",
    "stage2_allocation": "192 benefit_bank games over the frozen applicability families in "
                         "name order: game j is the floor(j/m)-th game, in bank order, of the "
                         "(j mod m)-th family (ADR-0012 D4)",
    "generalization_allocation": "in scope: stage 2's games, by reference; out of scope: 24 "
                                 "benefit_bank games over the omitted families in name order, "
                                 "round-robin, each family's first games in bank order; an "
                                 "all-family applicability gets a vacuous out-of-scope stage "
                                 "(ADR-0012 D5)",
    "arm_genomes": "G1 = G0 + skills in frozen rank order, each origin assimilation with its "
                   "source trajectory and aggregate CIG record; B all candidates, C promoted only",
    "run_order": ("lifetimes-all-seeds", "t_reg", "per-seed-miner-gate-arms", "final-evaluation"),
    "final_evaluation": {"temperature": 0.0, "sampling_seed": None, "arm_a": "once"},
})


def attempt_sampling_seed(mining_seed: int, task_id: str, attempt: int) -> int:
    """The pre-registered stochastic source of a mining-temperature attempt
    (manifest: future stochastic repeats need an explicit source)."""
    key = f"heritagent-attempt|{mining_seed}|{task_id}|{attempt}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)


def max_model_calls_per_episode() -> int:
    """One model call per step plus at most one per retry."""
    return PROTOCOL["budgets"]["max_steps"] + MAX_TOTAL_RETRIES
