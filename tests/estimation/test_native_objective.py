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
import pandas as pd
import pytest
from sympy import Symbol

from SymbolicDSGE import ModelParser, DSGESolver
from SymbolicDSGE.kalman.config import KalmanConfig
from SymbolicDSGE.estimation import Estimator


def _native_loglik(
    *, compiled, y, observables, filter_mode, estimated, ss_seed, **kwargs
) -> float:
    """The native loglik at the base calibration, through the production path."""
    est = Estimator(
        compiled=compiled,
        y=y,
        observables=list(observables),
        filter_mode=filter_mode,
        estimated_params=[estimated],
        ss_seed=ss_seed,
        **kwargs,
    )
    theta0 = np.ascontiguousarray(est.resolve_theta0(None), dtype=np.float64)
    return est.loglik(theta0)


@pytest.fixture(scope="module")
def bundle(post82_test_model_path):
    model, kalman = ModelParser(post82_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    steady = np.zeros((compiled.n_declared,), dtype=np.float64)
    solved = solver.solve(compiled=compiled, ss_seed=steady)

    params = model.calibration.parameters
    std_map = model.calibration.shock_std
    sig = {s: float(params[std_map[Symbol(s)]]) for s in ("e_g", "e_z", "e_r")}

    T = 24
    rng = np.random.default_rng(20260303)
    sim = solved.sim(
        T=T,
        shocks={
            "e_g": rng.normal(0.0, sig["e_g"], size=T),
            "e_z": rng.normal(0.0, sig["e_z"], size=T),
            "e_r": rng.normal(0.0, sig["e_r"], size=T),
        },
        x0=np.zeros((compiled.n_var,), dtype=np.float64),
        observables=True,
    )
    y = pd.DataFrame(
        {
            "OutGap": sim.observables["OutGap"][1:],
            "Infl": sim.observables["Infl"][1:],
            "Rate": sim.observables["Rate"][1:],
        }
    )
    return {
        "compiled": compiled,
        "kalman": compiled.kalman,
        "solver": solver,
        "solved": solved,
        "steady": steady,
        "y": y,
    }


@pytest.mark.parametrize("filter_mode", ["linear", "extended"])
def test_native_objective_matches_model_kalman(bundle, filter_mode):
    obs = ["Infl", "Rate"]
    ll = _native_loglik(
        compiled=bundle["compiled"],
        y=bundle["y"],
        observables=obs,
        filter_mode=filter_mode,
        estimated="psi_pi",
        ss_seed=bundle["steady"],
    )
    ll_model = (
        bundle["solved"]
        .kalman(y=bundle["y"], filter_mode=filter_mode, observables=obs)
        .loglik
    )

    # A Blanchard-Kahn failure returns -inf rather than a wrong number, so a
    # finite value is the observable form of the old bk == 0 assertion.
    assert np.isfinite(ll)
    np.testing.assert_allclose(ll, float(ll_model), rtol=1e-9, atol=1e-9)


@pytest.fixture(scope="module")
def rbc_bundle(rbc_second_order_test_model_path):
    """Second-order RBC (levels model, nonzero steady state) for the unscented
    parity. The fixture has no kalman section, so a scalar measurement-noise
    configuration is supplied."""
    model, _ = ModelParser(rbc_second_order_test_model_path).get_all()
    R = np.array([[1e-4]], dtype=np.float64)
    solver = DSGESolver(model, KalmanConfig(R=R))
    compiled = solver.compile()

    # The config's own ss_seed resolves Newton here, so the solve needs no help.
    solved = solver.solve(compiled=compiled, order=2)

    T = 40
    rng = np.random.default_rng(20260303)
    # x0 defaults to the steady state.
    sim = solved.sim(
        T=T,
        shocks={"e": rng.normal(0.0, 0.01, size=T)},
        observables=True,
    )
    return {
        "compiled": compiled,
        "solver": solver,
        "solved": solved,
        "y": pd.DataFrame({"c_obs": sim.observables["c_obs"][1:]}),
        "R": R,
    }


def test_native_unscented_objective_matches_model_kalman(rbc_bundle):

    obs = ["c_obs"]
    jitter, symmetrize = 1e-8, True
    ll = _native_loglik(
        compiled=rbc_bundle["compiled"],
        y=rbc_bundle["y"],
        observables=obs,
        filter_mode="unscented",
        estimated="rho",
        ss_seed=None,
        R=rbc_bundle["R"],
        jitter=jitter,
        symmetrize=symmetrize,
    )
    ll_model = (
        rbc_bundle["solved"]
        .kalman(
            y=rbc_bundle["y"],
            filter_mode="unscented",
            observables=obs,
            jitter=jitter,
            symmetrize=symmetrize,
        )
        .loglik
    )

    assert np.isfinite(ll)
    np.testing.assert_allclose(ll, float(ll_model), rtol=1e-9, atol=1e-9)
