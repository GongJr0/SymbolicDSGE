"""Deterministic parity tests for the native MCMC port (issue #331).

Rather than argue statistical equivalence between the native chain and the
numpy-era chain (which genuinely differ: Cholesky vs SVD proposal, own draws vs
``multivariate_normal``, running vs batch covariance), these tests strip out
every source of randomness divergence and check the native pieces against an
exact numpy transcription of the *native* algorithm. Three concerns, isolated:

1. **Loop mechanics** -- native ``run_mcmc`` vs ``native_algorithm_mimic`` driven
   by the *same* native objective (a ``NativeLogpost`` handle) and the same PCG64
   stream. With adaptation off the proposal is diagonal (no Cholesky), so it is
   **bit-exact**. With adaptation on the only residual is the Cholesky ULP
   (``sdsge_chol`` vs LAPACK ``potrf``), which tracks to machine epsilon and never
   flips an accept (accept counts match exactly).
2. **Objective** -- the native objective (``NativeLogpost``) vs the Python
   ``_safe_loglik`` / ``_safe_logpost`` over a theta grid, including the
   stationarity boundary (``rho_r >= 1`` -> both ``-inf``). They share the native
   kernels, so they agree bit-for-bit; this is where the BK-detection bug lived.
3. **Adaptation covariance** -- the running Welford estimator vs numpy batch
   ``np.cov`` (holding proposal / RNG / objective fixed): the single algorithmic
   difference between the native and numpy-era chains, and it is numerically
   benign.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.estimation import run_mcmc

# NativeLogpost is a test-only single-eval objective handle; it is deliberately
# not on the package's public export, so reach it through the private compiled
# module rather than the ``_ckernels.estimation`` package surface.
from SymbolicDSGE._ckernels.estimation._estimation import NativeLogpost

from tests._oracles.estimation import (
    build_post82_estimator,
    native_algorithm_mimic,
)


@pytest.fixture(scope="module")
def parity_setup():
    """Compiled POST82 estimator + its native ctx DTO and a shared logpost
    handle. Built once; the DTO is reused (run_mcmc re-marshals it each call)."""
    est = build_post82_estimator()
    ctx, mode = est._build_native_context()
    nlp = NativeLogpost(ctx, mode)
    return est, ctx, mode, nlp


def _theta0(est, values):
    return np.ascontiguousarray(
        est.resolve_theta0(np.array(values, dtype=np.float64)), dtype=np.float64
    )


# --- 1. loop mechanics -------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize(
    "theta0, proposal_scale",
    [([2.0, 0.8], 0.1), ([1.5, 0.7], 0.05), ([2.5, 0.9], 0.2)],
)
def test_loop_bit_exact_without_adaptation(parity_setup, seed, theta0, proposal_scale):
    """Adaptation off -> diagonal proposal, no Cholesky: native and the mimic are
    bit-for-bit identical (draws, accepts, thinning, buffers, BK auto-reject)."""
    est, ctx, mode, nlp = parity_setup
    th0 = _theta0(est, theta0)
    kw = dict(n_draws=300, burn_in=100, thin=1, proposal_scale=proposal_scale)

    out = run_mcmc(ctx, mode, th0, np.random.default_rng(seed), adapt=0, **kw)
    ref = native_algorithm_mimic(
        nlp.logpost, th0, np.random.default_rng(seed), adapt=False, **kw
    )

    np.testing.assert_array_equal(out["samples_theta"], ref.kept)
    np.testing.assert_array_equal(out["logpost_trace"], ref.kept_lp)
    assert out["n_accepted"] == ref.n_accepted


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("theta0", [[2.0, 0.8], [1.8, 0.85]])
def test_loop_matches_with_adaptation(parity_setup, seed, theta0):
    """Adaptation on: the Cholesky ULP (sdsge_chol vs LAPACK) keeps this from
    bit-exact, but it must not flip accepts -- samples track to machine epsilon
    and accept counts match exactly."""
    est, ctx, mode, nlp = parity_setup
    th0 = _theta0(est, theta0)
    kw = dict(
        n_draws=500,
        burn_in=300,
        thin=1,
        adapt_start=50,
        adapt_interval=25,
        proposal_scale=0.1,
    )

    out = run_mcmc(ctx, mode, th0, np.random.default_rng(seed), adapt=1, **kw)
    ref = native_algorithm_mimic(
        nlp.logpost, th0, np.random.default_rng(seed), adapt=True, **kw
    )

    assert out["n_accepted"] == ref.n_accepted
    np.testing.assert_allclose(out["samples_theta"], ref.kept, rtol=0, atol=1e-10)
    np.testing.assert_allclose(out["logpost_trace"], ref.kept_lp, rtol=0, atol=1e-8)


# --- 2. objective parity -----------------------------------------------------


def test_objective_matches_python_over_grid(parity_setup):
    """The native objective equals the Python ``_safe_loglik`` / ``_safe_logpost``
    bit-for-bit on the interior, and BK detection agrees on the boundary
    (``rho_r >= 1`` -> both ``-inf``). Grid spans both regimes."""
    est, _, _, nlp = parity_setup
    n_finite = n_bk = 0
    for psi in np.linspace(1.2, 3.5, 10):
        for rho in np.linspace(0.5, 1.05, 12):
            th = np.array([psi, rho], dtype=np.float64)
            n_ll = nlp.loglik(th)
            n_lp = nlp.logpost(th)
            p_ll = float(est._safe_loglik(th))
            p_lp = float(est._safe_logpost(th))
            # BK / non-finite classification must agree.
            assert np.isfinite(n_ll) == np.isfinite(p_ll)
            assert np.isfinite(n_lp) == np.isfinite(p_lp)
            if np.isfinite(n_ll):
                assert n_ll == p_ll
                assert n_lp == p_lp
                n_finite += 1
            else:
                n_bk += 1
    assert n_finite > 0 and n_bk > 0


# --- 3. adaptation covariance ------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_welford_matches_batch_cov(parity_setup, seed):
    """Running Welford covariance (native) vs numpy batch ``np.cov`` (numpy-era),
    holding proposal / RNG / objective fixed. The two are the same sample
    covariance up to summation order, so over a short chain the chains track
    tightly and the accept counts match -- the option-(b) switch to Welford is
    numerically benign."""
    est, _, _, nlp = parity_setup
    th0 = _theta0(est, [2.0, 0.8])
    kw = dict(
        n_draws=300,
        burn_in=200,
        thin=1,
        adapt=True,
        adapt_start=50,
        adapt_interval=25,
        proposal_scale=0.1,
    )
    w = native_algorithm_mimic(
        nlp.logpost, th0, np.random.default_rng(seed), cov_method="welford", **kw
    )
    b = native_algorithm_mimic(
        nlp.logpost, th0, np.random.default_rng(seed), cov_method="batch", **kw
    )
    assert w.n_accepted == b.n_accepted
    np.testing.assert_allclose(w.kept, b.kept, rtol=0, atol=1e-8)
