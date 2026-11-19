"""Tests for the ESS kernel in ``_ckernels/estimation``.

An AR(1) has ``tau = (1 + phi) / (1 - phi)`` in closed form, which is the oracle
for the estimator; the tolerance is wide because ``tau`` is itself estimated off
a finite chain. The four status codes need chains a sampler cannot produce, so
each has a deterministic generator rather than a seeded draw.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.estimation._mcmc_diagnostics import (
    ess_rhat,
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
half = np.tile([1.0, -1.0], 256)
UNDECAYED = np.concatenate([half + 10.0, half - 10.0]).reshape(1, 1024, 1)

#: Five alternations and a gap per period puts ``tau`` near -0.72 at any length,
#: far enough from zero that the verdict does not ride on rounding.
NONPOSITIVE_TAU = np.tile([1.0, -1.0, 1.0, -1.0, 1.0, 0.0], 170).reshape(1, 1020, 1)

CONSTANT = np.ones((1, 1024, 1))


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(n)
    chain = np.empty(n)
    chain[0] = innovations[0] / np.sqrt(1.0 - phi**2) if phi else innovations[0]
    for t in range(1, n):
        chain[t] = phi * chain[t - 1] + innovations[t]
    return chain.reshape(1, n, -1)


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9, 0.98])
def test_ess1d_recovers_the_ar1_autocorrelation_time(phi):
    n = 20000
    tau = n / ess_rhat(_ar1(n, phi, seed=int(phi * 100)))[0][0]
    assert tau == pytest.approx((1.0 + phi) / (1.0 - phi), rel=0.15)


@pytest.mark.parametrize(
    ("chain", "code"),
    [
        (np.full((1, 1, 1), 1.0), EBADSHAPE),
        (CONSTANT, EUDEFVARIANCE),
        (UNDECAYED, ENODECAY),
        (NONPOSITIVE_TAU, ENONPOSTAU),
    ],
    ids=["single_draw", "constant", "undecayed", "nonpositive_tau"],
)
def test_kernel_pairs_a_code_with_nan_for_degenerate_chains(chain, code):
    val, _, rc = ess_rhat(chain)
    assert rc[0] == code
    assert np.isnan(val[0])


@pytest.mark.parametrize(
    "chain",
    [CONSTANT, UNDECAYED, NONPOSITIVE_TAU],
    ids=["constant", "undecayed", "nonpositive_tau"],
)
def test_wrappers_forward_nan_instead_of_raising(chain):
    assert np.isnan(ess_rhat(chain)[0][0])


@pytest.mark.parametrize(
    ("scale", "shift"), [(2.0, 0.0), (0.25, 100.0), (3.7, -5.5), (-1.0, 0.0)]
)
def test_ess_is_invariant_to_affine_rescaling(scale, shift):
    chain = _ar1(8000, 0.8, seed=5)
    scaled = ess_rhat(scale * chain + shift)[0][0]
    raw = ess_rhat(chain)[0][0]
    assert scaled == pytest.approx(raw, rel=1e-12)


def test_ess_is_invariant_to_time_reversal():
    chain = _ar1(8000, 0.8, seed=5)
    forward = ess_rhat(chain)[0][0]
    reverse = ess_rhat(chain[::-1])[0][0]
    assert forward == pytest.approx(reverse, rel=1e-12)
