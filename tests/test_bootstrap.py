"""Unit tests for the non-parametric question-level bootstrap CI."""

from __future__ import annotations

import math

import numpy as np

from scripts.aggregate_bootstrap import bootstrap_ci


def test_empty_returns_nan_tuple() -> None:
    mean, lo, hi = bootstrap_ci([])
    assert math.isnan(mean)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_constant_array_zero_width_ci() -> None:
    mean, lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 1.0, 1.0])
    assert mean == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_balanced_binary_straddles_half() -> None:
    mean, lo, hi = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], n_boot=2000)
    assert math.isclose(mean, 0.5)
    assert lo < 0.5 < hi


def test_deterministic_with_same_seed() -> None:
    data = [0.0, 0.25, 0.5, 0.75, 1.0, 0.3, 0.6]
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    out_a = bootstrap_ci(data, n_boot=1000, rng=rng_a)
    out_b = bootstrap_ci(data, n_boot=1000, rng=rng_b)
    assert out_a == out_b


def test_95pct_wider_than_90pct() -> None:
    data = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.3, 0.7]
    _, lo95, hi95 = bootstrap_ci(data, n_boot=2000, alpha=0.05, rng=np.random.default_rng(1))
    _, lo90, hi90 = bootstrap_ci(data, n_boot=2000, alpha=0.10, rng=np.random.default_rng(1))
    assert (hi95 - lo95) >= (hi90 - lo90)
