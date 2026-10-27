"""Parity: the native objective vs the model Kalman loglik, one per filter mode.

``NativeLogpost`` marshals a context the same way ``Estimator`` does and calls
the same C objective the estimation and sampling drivers call, so the linear and
extended cases pin the whole native path from marshalling through solve, filter
and loglik against ``SolvedModel.kalman(...).loglik``, the oracle the backend
tests use. Each estimates one parameter and evaluates at its calibrated value,
so the resolved parameter vector is the base calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.estimation import Estimator


def _native_loglik(
    *, config, compiled, y, observables, filter_mode, estimated, ss_seed, **kwargs
) -> float:
    """The native loglik at the base calibration, through the production path."""
    est = Estimator(
        config=config,
        compiled=compiled,
        y=y,
        observables=list(observables),
        filter_mode=filter_mode,
        estimated_params=[estimated],
        ss_seed=ss_seed,
        **kwargs,
    )
    return est.loglik()


@pytest.mark.parametrize("filter_mode", ["linear", "extended"])
def test_native_objective_matches_model_kalman(solved_post82, y_post82, filter_mode):
    config = solved_post82.config
    compiled = solved_post82.compiled
    obs = ["Infl", "Rate"]
    ll = _native_loglik(
        config=config,
        compiled=compiled,
        y=y_post82,
        observables=obs,
        filter_mode=filter_mode,
        estimated="psi_pi",
        ss_seed=solved_post82.policy.steady_state,
    )
    ll_model = solved_post82.kalman(
        y=y_post82, filter_mode=filter_mode, observables=obs
    ).loglik

    # A Blanchard-Kahn failure returns -inf rather than a wrong number, so a
    # finite value is the observable form of the old bk == 0 assertion.
    assert np.isfinite(ll)
    np.testing.assert_allclose(ll, float(ll_model), rtol=1e-9, atol=1e-9)


def test_native_unscented_objective_matches_model_kalman(solved_rbc_so):
    """The second-order RBC is a levels model with a nonzero steady state, which
    the gap model above cannot reach. Both sides read R off the model's own
    kalman block, so neither is handed one."""
    obs = ["c_obs"]
    jitter, symmetrize = 1e-8, True

    T = 40
    rng = np.random.default_rng(20260303)
    # x0 defaults to the steady state.
    sim = solved_rbc_so.sim(
        T=T,
        shocks={"e": rng.normal(0.0, 0.01, size=T)},
        observables=True,
    )
    y = sim.observables["c_obs"][1:].reshape(-1, 1)

    config = solved_rbc_so.config
    compiled = solved_rbc_so.compiled

    ll = _native_loglik(
        config=config,
        compiled=compiled,
        y=y,
        observables=obs,
        filter_mode="unscented",
        estimated="rho",
        ss_seed=None,
        jitter=jitter,
        symmetrize=symmetrize,
    )
    ll_model = solved_rbc_so.kalman(
        y=y,
        filter_mode="unscented",
        observables=obs,
        jitter=jitter,
        symmetrize=symmetrize,
    ).loglik

    assert np.isfinite(ll)
    np.testing.assert_allclose(ll, float(ll_model), rtol=1e-9, atol=1e-9)
