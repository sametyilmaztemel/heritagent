"""Deterministic paired cluster bootstrap tests (issue #11 criterion 9; the
bootstrap is descriptive under ADR-0012 D4) and the exact sign test of the
recalibrated stage 2."""

import math
import random
from fractions import Fraction

import pytest

from inheritance.bootstrap import paired_cluster_bootstrap, sign_test_p


def test_known_vector_all_positive_small_effect():
    # all deltas +0.1: point 0.1, every resample mean is 0.1 -> LB == 0.1 > 0
    result = paired_cluster_bootstrap([0.1] * 20, seed=11)
    assert result.point_estimate == pytest.approx(0.1)
    assert result.lower_bound == pytest.approx(0.1)
    assert result.upper_bound == pytest.approx(0.1)


def test_known_vector_mixed_effect_lower_bound_negative():
    # half +1, half -1: point 0 but resample means spread widely -> LB < 0
    deltas = [1.0] * 10 + [-1.0] * 10
    result = paired_cluster_bootstrap(deltas, seed=11)
    assert result.point_estimate == 0.0
    assert result.lower_bound < 0


def test_fixed_seed_deterministic():
    deltas = [0.2, -0.1, 0.3, 0.1, 0.0, 0.25, 0.05, 0.15, -0.05, 0.1,
               0.2, 0.0, 0.1, 0.3, -0.2, 0.1, 0.05, 0.1, 0.0, 0.2]
    a = paired_cluster_bootstrap(deltas, seed=4242)
    b = paired_cluster_bootstrap(deltas, seed=4242)
    assert (a.point_estimate, a.lower_bound, a.upper_bound) == \
        (b.point_estimate, b.lower_bound, b.upper_bound)
    c = paired_cluster_bootstrap(deltas, seed=7)
    # different seed -> different RNG stream -> different resample means
    assert (c.lower_bound, c.upper_bound) != (a.lower_bound, a.upper_bound)
    assert a.seed == 4242 and c.seed == 7


def test_task_pair_bootstrap_not_call_level_pseudo_replication():
    """20 paired task deltas -> exactly 20 clusters; resample means are over
    cluster deltas, not duplicated per-episode calls."""
    deltas = [1.0] + [0.0] * 19  # one strongly positive task
    result = paired_cluster_bootstrap(deltas, seed=1)
    # point estimate = 1/20 = 0.05 exactly
    assert result.point_estimate == pytest.approx(0.05)
    # a resample including the 1.0 k times has mean k/20: achievable means
    # are multiples of 1/20 -> the CI bounds must lie on that grid
    assert result.lower_bound >= 0.0
    assert result.lower_bound * 20 == pytest.approx(round(result.lower_bound * 20))


@pytest.mark.parametrize("n_plus,n_minus,p", [
    (0, 0, Fraction(1)),            # no discordant game: nothing against the null
    (1, 0, Fraction(1, 2)),
    (0, 1, Fraction(1)),
    (3, 0, Fraction(1, 8)),
    (9, 0, Fraction(1, 512)),       # ADR-0012 D4's look-4 alpha 3/1000 is met
    (11, 0, Fraction(1, 2048)),     # look 1's alpha 1/2000 is met at net 11
    (10, 0, Fraction(1, 1024)),     # and missed at net 10
    (5, 2, Fraction(29, 128)),
])
def test_sign_test_known_values(n_plus, n_minus, p):
    assert sign_test_p(n_plus, n_minus) == p


def test_sign_test_is_the_binomial_upper_tail():
    for n in range(0, 25):
        for n_plus in range(n + 1):
            expected = Fraction(sum(math.comb(n, i) for i in range(n_plus, n + 1)), 2 ** n)
            assert sign_test_p(n_plus, n - n_plus) == expected
            assert isinstance(sign_test_p(n_plus, n - n_plus), Fraction)


@pytest.mark.parametrize("bad", [-1, True, 1.0, "1", None])
def test_sign_test_counts_must_be_non_negative_ints(bad):
    with pytest.raises(ValueError, match="non-negative ints"):
        sign_test_p(bad, 0)
    with pytest.raises(ValueError, match="non-negative ints"):
        sign_test_p(0, bad)


def test_empty_and_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="at least one"):
        paired_cluster_bootstrap([], seed=1)
    with pytest.raises(ValueError, match="n_resamples"):
        paired_cluster_bootstrap([0.1], seed=1, n_resamples=0)


def test_uses_explicit_rng_not_hidden_defaults():
    """The implementation must draw from random.Random(seed): identical seed
    reproduces the exact resample stream."""
    deltas = [float(i % 5) / 10 for i in range(20)]
    rng = random.Random(99)
    expected_first_draws = [rng.randrange(20) for _ in range(20)]
    # first resample mean must equal the mean of deltas at those indices
    result = paired_cluster_bootstrap(deltas, seed=99, n_resamples=1)
    expected_mean = sum(deltas[i] for i in expected_first_draws) / 20
    assert result.point_estimate == sum(deltas) / 20  # point always exact mean
    assert result.lower_bound == pytest.approx(expected_mean)
    assert result.upper_bound == pytest.approx(expected_mean)


@pytest.mark.parametrize("seed", [None, True, False, 1.0, "0"])
def test_seed_must_be_an_explicit_int(seed):
    """seed=None would seed from the OS: identical evidence could pass or
    fail between runs."""
    with pytest.raises(ValueError, match="explicit int"):
        paired_cluster_bootstrap([1, 0, 1, 0], seed=seed, n_resamples=10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 2, -1.5, True, "1", None])
def test_deltas_must_be_finite_numbers_in_unit_range(bad):
    with pytest.raises(ValueError, match=r"finite number in \[-1, 1\]"):
        paired_cluster_bootstrap([1, 0, bad], seed=0, n_resamples=10)


@pytest.mark.parametrize("n_resamples", [True, 1.5, 0])
def test_n_resamples_must_be_a_positive_int(n_resamples):
    with pytest.raises(ValueError, match="positive int"):
        paired_cluster_bootstrap([1, 0], seed=0, n_resamples=n_resamples)


def test_valid_inputs_unchanged():
    """Ints, floats and the boundary values -1 and 1 stay accepted and give
    the same result as before the validation."""
    a = paired_cluster_bootstrap([1, 0, -1, 1.0, 0.0], seed=7, n_resamples=200)
    b = paired_cluster_bootstrap([1, 0, -1, 1, 0], seed=7, n_resamples=200)
    assert (a.point_estimate, a.lower_bound, a.upper_bound) == \
        (b.point_estimate, b.lower_bound, b.upper_bound)
