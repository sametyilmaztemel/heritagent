"""Descriptive backend-noise measures from the gate's G0 cells (ADR-0012 D8,
owner-accepted 2026-10-19).

The server runs without batch-invariant kernels (ADR-0011 D2), so two
temperature-0 executions of one request can differ. The gate's cells give
two lower references for that disagreement, at no episode's cost:

- **cross-round:** a seed's cells of one game in two rounds are identical
  requests (same prompt, request seed and settings) in different batches;
- **cross-seed:** two seeds' cells of one game share the prompt, and only
  the request seed differs, which greedy decoding should not use (U). Each
  seed contributes its earliest-round cell of the game.

They decide nothing. They are lower references for the changed-prompt
disagreement a with/without contrast carries, not estimates of it. Each
comes with an exact (Clopper-Pearson) 95 % interval."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from itertools import combinations

CONFIDENCE = 0.95


def _log_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _tail(ks, n: int, p: float) -> float:
    logs = [_log_pmf(k, n, p) for k in ks]
    top = max(logs)
    if top == -math.inf:
        return 0.0
    return math.exp(top) * sum(math.exp(v - top) for v in logs)


def _solve(f, target: float) -> float:
    """The p in [0, 1] where the increasing function f crosses target."""
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2
        if f(middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def exact_interval(k: int, n: int, confidence: float = CONFIDENCE) -> list[float]:
    """The Clopper-Pearson interval for k successes in n trials."""
    if isinstance(k, bool) or isinstance(n, bool) or not 0 <= k <= n or n < 1:
        raise ValueError(f"need 0 <= k <= n and n >= 1, got k={k!r}, n={n!r}")
    alpha = 1 - confidence
    lower = 0.0 if k == 0 else _solve(lambda p: _tail(range(k, n + 1), n, p), alpha / 2)
    upper = 1.0 if k == n else _solve(lambda p: 1 - _tail(range(0, k + 1), n, p), 1 - alpha / 2)
    return [lower, upper]


def gate_cells(seed: int, gate: Mapping) -> dict[tuple[int, int, str], bool]:
    """(seed, round, game) -> the cell's success, from every record that read
    a cell; two readers of one cell must agree."""
    cells: dict[tuple[int, int, str], bool] = {}

    def add(gate_round: int, task_id: str, success: bool) -> None:
        key = (seed, gate_round, task_id)
        if cells.setdefault(key, success) != success:
            raise ValueError(f"the R{gate_round} cell of {task_id!r} (seed {seed}) is recorded "
                             f"with two outcomes")

    for child in gate["children"].values():
        m = child.get("measurements") or {}
        stage = child.get("stage")
        if stage in ("ablation", "generalization_out_of_scope"):
            for outcome in m.get("paired_outcomes", ()):
                add(outcome["cell"]["round"], outcome["task_id"], outcome["cell"]["success"])
        elif stage == "interaction_regression":
            baseline = (m.get("regression") or {}).get("paired_g0_run") or {}
            for episode in baseline.get("episodes", ()):
                add(episode["round"], episode["task_id"], episode["success"])
            for pair in (m.get("interaction") or {}).get("pair_evidence", ()):
                for outcome in pair["canonical_evidence"]["paired_outcomes"]:
                    if outcome["condition"] == "neither":
                        add(6, outcome["task_id"], outcome["success"])
    return cells


def _measure(pairs: list[tuple[bool, bool]]) -> dict:
    n = len(pairs)
    disagreements = sum(a != b for a, b in pairs)
    if not n:
        return {"pairs": 0, "disagreements": 0, "rate": None, "interval_95": None}
    rate = Fraction(disagreements, n)
    return {"pairs": n, "disagreements": disagreements,
            "rate": {"exact": f"{rate.numerator}/{rate.denominator}", "float": float(rate)},
            "interval_95": exact_interval(disagreements, n)}


def cell_disagreement(gates: Mapping[str, Mapping]) -> dict:
    """ADR-0012 D8's two descriptive measures over every seed's gate."""
    cells: dict[tuple[int, int, str], bool] = {}
    for seed, gate in gates.items():
        cells.update(gate_cells(int(seed), gate))
    by_seed_game: dict[tuple[int, str], dict[int, bool]] = {}
    for (seed, gate_round, task_id), success in cells.items():
        by_seed_game.setdefault((seed, task_id), {})[gate_round] = success
    cross_round = [(rounds[a], rounds[b]) for rounds in by_seed_game.values()
                   for a, b in combinations(sorted(rounds), 2)]
    earliest: dict[str, dict[int, bool]] = {}
    for (seed, task_id), rounds in by_seed_game.items():
        earliest.setdefault(task_id, {})[seed] = rounds[min(rounds)]
    cross_seed = [(seeds[a], seeds[b]) for seeds in earliest.values()
                  for a, b in combinations(sorted(seeds), 2)]
    return {"descriptive_only": True,
            "basis": "the gate's G0 cells (ADR-0012 D8): lower references for the "
                     "changed-prompt disagreement a gate contrast carries, not estimates of it",
            "cross_round": _measure(cross_round), "cross_seed": _measure(cross_seed),
            "cells": len(cells)}
