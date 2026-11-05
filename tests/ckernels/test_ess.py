"""Tests for the ESS kernel in ``_ckernels/estimation``.

An AR(1) has ``tau = (1 + phi) / (1 - phi)`` in closed form, which is the oracle
for the estimator; the tolerance is wide because ``tau`` is itself estimated off
a finite chain. The four status codes need chains a sampler cannot produce, so
each has a deterministic generator rather than a seeded draw.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.estimation import ess, ess1d
from SymbolicDSGE._ckernels.estimation._ess import (
    ess1d_kernel_probe,
    SDSGE_ESS_EBADSHAPE,
    SDSGE_ESS_EUDEFVARIANCE,
    SDSGE_ESS_ENODECAY,
    SDSGE_ESS_ENONPOSTAU,
)

EBADSHAPE = SDSGE_ESS_EBADSHAPE
EUDEFVARIANCE = SDSGE_ESS_EUDEFVARIANCE
ENODECAY = SDSGE_ESS_ENODECAY
ENONPOSTAU = SDSGE_ESS_ENONPOSTAU

#: Lag-1 autocorrelation of -0.5 with nothing beyond it, so the pair sums never
#: turn and the scan runs out of lags.
UNDECAYED = np.tile([1.0, -1.0], 512)

#: Five alternations and a gap per period puts ``tau`` near -0.72 at any length,
#: far enough from zero that the verdict does not ride on rounding.
NONPOSITIVE_TAU = np.tile([1.0, -1.0, 1.0, -1.0, 1.0, 0.0], 512)

CONSTANT = np.ones(1024)


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(n)
    chain = np.empty(n)
    chain[0] = innovations[0] / np.sqrt(1.0 - phi**2) if phi else innovations[0]
    for t in range(1, n):
        chain[t] = phi * chain[t - 1] + innovations[t]
    return chain


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9, 0.98])
def test_ess1d_recovers_the_ar1_autocorrelation_time(phi):
    n = 20000
    tau = n / ess1d(_ar1(n, phi, seed=int(phi * 100)))
    assert tau == pytest.approx((1.0 + phi) / (1.0 - phi), rel=0.15)


@pytest.mark.parametrize(
    ("chain", "code"),
    [
        (np.array([1.0]), EBADSHAPE),
        (CONSTANT, EUDEFVARIANCE),
        (UNDECAYED, ENODECAY),
        (NONPOSITIVE_TAU, ENONPOSTAU),
    ],
    ids=["single_draw", "constant", "undecayed", "nonpositive_tau"],
)
def test_kernel_pairs_a_code_with_nan_for_degenerate_chains(chain, code):
    rc, value = ess1d_kernel_probe(chain)
    assert rc == code
    assert np.isnan(value)


@pytest.mark.parametrize(
    "chain",
    [CONSTANT, UNDECAYED, NONPOSITIVE_TAU],
    ids=["constant", "undecayed", "nonpositive_tau"],
)
def test_wrappers_forward_nan_instead_of_raising(chain):
    assert np.isnan(ess1d(chain))
    assert np.isnan(ess(chain.reshape(-1, 1))[0])


def test_a_single_draw_is_rejected_by_both_wrappers():
    with pytest.raises(ValueError, match="at least two"):
        ess1d(np.array([1.0]))
    with pytest.raises(ValueError, match="at least two"):
        ess(np.array([[1.0]]))


def test_ess_reuses_one_arena_across_columns():
    """A column that fails must leave no residue for the columns after it."""
    columns = [
        _ar1(6000, 0.0, 11),
        _ar1(6000, 0.98, 12),
        np.ones(6000),
        _ar1(6000, 0.9, 13),
    ]

    batch = ess(np.column_stack(columns))

    for value, column in zip(batch, columns):
        solo = ess1d(column)
        assert value == solo or (np.isnan(value) and np.isnan(solo))


@pytest.mark.parametrize(
    ("scale", "shift"), [(2.0, 0.0), (0.25, 100.0), (3.7, -5.5), (-1.0, 0.0)]
)
def test_ess_is_invariant_to_affine_rescaling(scale, shift):
    chain = _ar1(8000, 0.8, seed=5)
    assert ess1d(scale * chain + shift) == pytest.approx(ess1d(chain), rel=1e-12)


def test_ess_is_invariant_to_time_reversal():
    chain = _ar1(8000, 0.8, seed=5)
    assert ess1d(chain[::-1]) == pytest.approx(ess1d(chain), rel=1e-12)
