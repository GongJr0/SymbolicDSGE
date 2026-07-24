"""End-to-end oracle for ``Estimator.mcmc`` on a real model.

Locks the current (numpy-era) adaptive random-walk Metropolis behavior before the
native-mainloop rewrite (#331). Unlike the MLE/MAP oracle, the MCMC port targets
**statistical equivalence**, not bit-parity: the native chain runs a frozen native
RNG stream (numpy draws via ``c_distributions`` + our own Cholesky proposal and
incremental covariance), so it will not reproduce this chain draw-for-draw. So the
guards here come in two layers:

* **Faithfulness (bit-exact, valid only pre-port).** ``adaptive_rwm_reference`` --
  the numpy-era loop lifted into ``tests/_oracles`` -- is asserted identical to
  ``Estimator.mcmc`` at a shared seed. This proves the frozen reference is a true
  transcription of today's lib behavior. It naturally starts failing once the lib
  goes native (expected), at which point it is retired / re-pointed at the harness.
* **Statistical golden (survives the port).** Acceptance rate and per-parameter
  posterior mean/std at a fixed seed are pinned. Post-port, the native chain is
  checked against ``adaptive_rwm_reference``'s marginals with a two-sample KS test
  plus a matched acceptance rate -- the reference regenerates the full sample cloud
  on demand, so these goldens only certify the reference itself has not drifted.

The ``post82`` fixture lives in tests/estimation/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.estimation.results import MCMCResult

from tests._oracles.estimation import adaptive_rwm_reference

_ESTIMATED = ["psi_pi", "rho_r"]
_TH0 = np.array([2.0, 0.8], dtype=np.float64)
_SEED = 20260724

# Small but non-trivial: exercises the d>=2 ``np.cov`` adaptation path the native
# port must match statistically, while staying cheap enough for CI (one Klein
# solve + Kalman filter per step).
_MCMC_KW = dict(
    n_draws=300,
    burn_in=200,
    thin=1,
    adapt=True,
    adapt_start=50,
    adapt_interval=25,
    proposal_scale=0.05,
    adapt_epsilon=1e-8,
)


def _mcmc_estimator(post82) -> Estimator:
    priors = {
        "psi_pi": Estimator.make_prior(
            distribution="normal",
            parameters={"mean": 2.0, "std": 0.5},
            transform="identity",
        ),
        "rho_r": Estimator.make_prior(
            distribution="normal",
            parameters={"mean": 0.8, "std": 0.1},
            transform="identity",
        ),
    }
    return Estimator(
        solver=post82["solver"],
        compiled=post82["compiled"],
        y=post82["y"],
        observables=post82["obs"],
        filter_mode="linear",
        estimated_params=_ESTIMATED,
        priors=priors,
        ss_seed=post82["steady"],
    )


def test_mcmc_reference_is_faithful_to_estimator(post82):
    """The lifted numpy-era reference reproduces ``Estimator.mcmc`` bit-for-bit at
    a shared seed. Load-bearing only while the lib is still on numpy; retire this
    once ``mcmc`` goes native (the two streams diverge by design)."""
    est = _mcmc_estimator(post82)

    res = est.mcmc(theta0=_TH0, random_state=_SEED, **_MCMC_KW)

    ref = adaptive_rwm_reference(
        est._safe_logpost,
        est.resolve_theta0(_TH0),
        np.random.default_rng(_SEED),
        **_MCMC_KW,
    )
    # Reference walks theta space; map to named params exactly as mcmc does.
    ref_params = np.empty_like(ref.kept)
    for i in range(ref.kept.shape[0]):
        p = est.theta_to_params(ref.kept[i])
        for j, name in enumerate(_ESTIMATED):
            ref_params[i, j] = float(p[name])

    assert res.accept_rate == pytest.approx(ref.accept_rate, abs=0.0)
    np.testing.assert_array_equal(res.samples, ref_params)
    np.testing.assert_array_equal(res.logpost_trace, ref.kept_lp)


def test_mcmc_statistical_golden(post82):
    """Pin the current chain's acceptance rate and posterior marginals. These are
    the reference numbers the native (statistically-equivalent) chain is compared
    against post-port."""
    est = _mcmc_estimator(post82)
    res = est.mcmc(theta0=_TH0, random_state=_SEED, **_MCMC_KW)

    assert isinstance(res, MCMCResult)
    assert res.samples.shape == (_MCMC_KW["n_draws"], len(_ESTIMATED))

    means = res.samples.mean(axis=0)
    stds = res.samples.std(axis=0, ddof=1)

    # GOLDEN (recorded from the numpy path; fill after first run).
    assert res.accept_rate == pytest.approx(_GOLDEN_ACCEPT, abs=1e-9)
    np.testing.assert_allclose(means, _GOLDEN_MEAN, atol=1e-6)
    np.testing.assert_allclose(stds, _GOLDEN_STD, atol=1e-6)


# --- Goldens (recorded from the numpy path, seed 20260724) -------------------
_GOLDEN_ACCEPT = 0.4
_GOLDEN_MEAN = np.array([1.940229940275, 0.835466957239], dtype=np.float64)
_GOLDEN_STD = np.array([0.426615007635, 0.029471640405], dtype=np.float64)
