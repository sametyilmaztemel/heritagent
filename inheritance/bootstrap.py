"""Deterministic paired cluster bootstrap (issue #11 criterion 9; ADR-0002
statistics — locked implementation).

- clusters = the distinct task instances; each paired with/without delta
  ``d_i`` is preserved as a pair;
- resample cluster indices WITH replacement, n_resamples times;
- statistic = mean of the resampled deltas;
- percentile CI: 2.5th / 97.5th percentiles of the sorted resample means;
- explicit deterministic seed via ``random.Random`` — no SciPy, no hidden
  version-dependent defaults.

This is stage-level paired intervention evidence. It is NOT global causal
utility, inheritance, or H2 support (those require #12 + EXP-0001).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

N_RESAMPLES = 10_000  # ADR-0002 locked
CONFIDENCE = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    lower_bound: float
    upper_bound: float
    n_resamples: int
    seed: int
    method: str = "percentile-paired-cluster-bootstrap"


def paired_cluster_bootstrap(deltas: list[float], *, seed: int,
                              n_resamples: int = N_RESAMPLES,
                              confidence: float = CONFIDENCE) -> BootstrapResult:
    """Percentile paired cluster bootstrap over task-level deltas."""
    if not deltas:
        raise ValueError("bootstrap requires at least one paired delta")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    n = len(deltas)
    point = sum(deltas) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    alpha = 1.0 - confidence
    lower_index = max(0, int(alpha / 2 * n_resamples))
    upper_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return BootstrapResult(point_estimate=point,
                            lower_bound=means[lower_index],
                            upper_bound=means[upper_index],
                            n_resamples=n_resamples, seed=seed)


def stage_passes(point_estimate: float, lower_bound: float, *,
                 tau_c: float = 0.05) -> bool:
    """Locked stage-2 pass rule: point estimate >= tau_c AND bootstrap 95%
    lower bound > 0."""
    return point_estimate >= tau_c and lower_bound > 0.0
