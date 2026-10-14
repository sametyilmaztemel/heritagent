"""The frozen EXP-0001 run protocol (ADR-0007; owner decisions E1-E8 of
2026-10-14, as amended by ADR-0008 D2-D3; plus the locked manifest values
it builds on).

Everything the orchestrator decides by rule is fixed here, before any
episode, and recorded in the run-start record."""

from __future__ import annotations

import hashlib
from types import MappingProxyType

MINING_SEEDS = (11, 23, 41)  # manifest evaluation.mining_seeds
LIFETIME_ATTEMPTS_PER_TASK = 3  # E1
MINING_TEMPERATURE = 0.7  # manifest model.mining_temperature
T_REG_SIZE = 40
T_REG_SEED = 0  # E3
CANDIDATE_CAP = 10  # manifest genome.candidate_set.cap (K)
MAX_TOKENS_PER_REQUEST = 512
MAX_TOTAL_RETRIES = 10
CAPS = MappingProxyType({"episodes_max": 7000, "model_calls_max": 225_000,
                         "wall_clock_hours_max": 96, "accelerator_nodes_max": 1})
# the committed pre-run artifacts (manifest run_protocol); a run binds to them
LOCKED_SPLIT_MANIFEST_SHA256 = "2f74d1882c1a0032dae3bed3bbd0b9cbc3bdd6bed7496522f984bf4028d1ed9c"
LOCKED_PILOT_SHA256 = "6286d0fa3072f3a179fbe70d543ec3a6a3e4020f05977111fc05ddb02d57e90c"

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
    "transfer_subset": "the 20-task ablation pool",
    "candidate_cap": CANDIDATE_CAP,
    "candidate_ranking": ("miner-estimated-generality", "discovery-order"),
    "budgets": {"max_steps": 50, "max_total_retries": MAX_TOTAL_RETRIES,
                "max_tokens_per_request": MAX_TOKENS_PER_REQUEST},
    "caps": CAPS,
    "cap_rule": "before each episode: stop if episodes + episodes in flight + 1 > "
                "episodes_max, if model_calls + the reservations in flight + (max_steps + "
                "max_total_retries) > model_calls_max, or if the active run time exceeds "
                "wall_clock_hours_max; a stopped run is incomplete; the miner's teacher calls "
                "count against model_calls_max",
    "wall_clock_basis": "active run time: the sum of the ledger sessions' spans, each from "
                        "the process opening the ledger to its last record (ADR-0008 D2)",
    "generalization_allocation": "10 in-scope games split evenly over the applicability "
                                 "families and 10 out-of-scope games over the others (remainder "
                                 "to the first families by name); per family the games with the "
                                 "smallest sha256('heritagent-gen|0|<family>|<game_id>'); an "
                                 "all-family applicability gets a vacuous out-of-scope stage",
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
