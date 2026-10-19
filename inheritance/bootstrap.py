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

``sign_test_p`` is the exact one-sided sign test of the recalibrated
stage 2 (ADR-0012 D4), computed with integers and ``Fraction``; the
bootstrap stays as the descriptive interval every stage-2 record carries.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction

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
    """Percentile paired cluster bootstrap over task-level deltas.

    Fails closed on inputs that would make the locked statistic silently
    non-deterministic or meaningless: the seed must be an explicit int
    (``random.Random(None)`` seeds from the OS), and every paired delta of
    binary outcomes must be a finite real number in [-1, 1] (bool rejected)."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"bootstrap seed must be an explicit int, got {seed!r}")
    if not deltas:
        raise ValueError("bootstrap requires at least one paired delta")
    for index, delta in enumerate(deltas):
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) \
                or not math.isfinite(delta) or not -1 <= delta <= 1:
            raise ValueError(
                f"paired delta {index} must be a finite number in [-1, 1], "
                f"got {delta!r}")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) \
            or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive int")
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


def sign_test_p(n_plus: int, n_minus: int) -> Fraction:
    """One-sided exact sign test (ADR-0012 D4): P(Bin(n_plus + n_minus, 1/2)
    >= n_plus), where n_plus counts the games the with-trait execution solved
    and its cell failed, n_minus the reverse. Exact (integers and
    ``Fraction``); 1 when no game is discordant."""
    for value in (n_plus, n_minus):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"discordant counts must be non-negative ints, got {value!r}")
    n = n_plus + n_minus
    if n == 0:
        return Fraction(1)
    return Fraction(sum(math.comb(n, i) for i in range(n_plus, n + 1)), 2 ** n)
