"""Deterministic paired cluster bootstrap tests (issue #11 criterion 9)."""

import random

import pytest

from inheritance.bootstrap import paired_cluster_bootstrap, stage_passes


def test_known_vector_all_positive_small_effect():
    # all deltas +0.1: point 0.1, every resample mean is 0.1 -> LB == 0.1 > 0
    result = paired_cluster_bootstrap([0.1] * 20, seed=11)
    assert result.point_estimate == pytest.approx(0.1)
    assert result.lower_bound == pytest.approx(0.1)
    assert result.upper_bound == pytest.approx(0.1)
    assert stage_passes(result.point_estimate, result.lower_bound)


def test_known_vector_mixed_effect_lower_bound_negative():
    # half +1, half -1: point 0 but resample means spread widely -> LB < 0
    deltas = [1.0] * 10 + [-1.0] * 10
    result = paired_cluster_bootstrap(deltas, seed=11)
    assert result.point_estimate == 0.0
    assert result.lower_bound < 0
    assert not stage_passes(result.point_estimate, result.lower_bound)


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


def test_tau_c_boundary():
    # point estimate exactly at tau_c with positive LB -> pass
    assert stage_passes(0.05, 0.01)
    # below tau_c -> fail even with positive LB
    assert not stage_passes(0.049, 0.01)
    # at tau_c but LB == 0 -> fail (LB must be strictly > 0)
    assert not stage_passes(0.05, 0.0)


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
