"""Parity: native linear objective vs the model Kalman loglik.

First native-objective slice: n_theta == 0 (base calibration), constant Q/R, no
prior. The native ``obj_linear_base`` runs the full solve -> filter -> loglik in
C; it must match ``SolvedModel.kalman(...).loglik`` (the same oracle the linear
backend test uses).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

from SymbolicDSGE import ModelParser, DSGESolver
from SymbolicDSGE.estimation import backend
from SymbolicDSGE._ckernels.kalman import stationary_covariance
from SymbolicDSGE.kalman.config import KalmanConfig
from SymbolicDSGE._ckernels.estimation._estimation import (
    obj_extended_base,
    obj_linear_base,
    obj_unscented_base,
)


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
        "solved": solved,
        "steady": steady,
        "y": y,
    }


def test_obj_linear_base_matches_model_kalman(bundle):
    compiled = bundle["compiled"]
    kalman = bundle["kalman"]
    solved = bundle["solved"]
    steady = bundle["steady"]
    y = bundle["y"]
    obs = ["Infl", "Rate"]

    base = backend.extract_base_params(compiled)
    prep = backend.prepare_filter_run(
        compiled=compiled,
        kalman=kalman,
        y=y,
        observables=obs,
        filter_mode="linear",
        jitter=None,
        symmetrize=None,
    )

    cc = np.ascontiguousarray
    Q = cc(backend.build_Q(compiled, base), dtype=np.float64)
    R = cc(backend.build_R(compiled, kalman, prep.observables, base), dtype=np.float64)
    calib = cc(backend.build_calib_param_vector(compiled, base), dtype=np.float64)
    steady_c = cc(steady, dtype=np.float64)
    y_c = cc(prep.y_reordered, dtype=np.float64)
    n_var = len(compiled.var_names)
    p0_err, P0 = stationary_covariance(solved.policy.A, solved.policy.B, Q)
    assert p0_err == 0
    P0 = cc(P0, dtype=np.float64)
    assert P0.shape == (n_var, n_var)

    ll, bk = obj_linear_base(
        compiled.construct_objective_cfunc().address,
        prep.meas_addr,
        prep.jac_addr,
        compiled.n_state,
        compiled.n_exog,
        len(prep.observables),
        steady_c,
        compiled._incidence,
        calib,
        Q,
        R,
        y_c,
        P0,
        float(prep.kf_jitter),
        int(prep.kf_sym),
    )

    ll_model = solved.kalman(y=y, filter_mode="linear", observables=obs).loglik

    assert bk == 0
    assert np.isfinite(ll)
    np.testing.assert_allclose(ll, ll_model, rtol=1e-9, atol=1e-9)


def test_obj_extended_base_matches_model_kalman(bundle):
    compiled = bundle["compiled"]
    kalman = bundle["kalman"]
    solved = bundle["solved"]
    steady = bundle["steady"]
    y = bundle["y"]
    obs = ["Infl", "Rate"]

    base = backend.extract_base_params(compiled)
    prep = backend.prepare_filter_run(
        compiled=compiled,
        kalman=kalman,
        y=y,
        observables=obs,
        filter_mode="extended",
        jitter=None,
        symmetrize=None,
    )

    cc = np.ascontiguousarray
    Q = cc(backend.build_Q(compiled, base), dtype=np.float64)
    R = cc(backend.build_R(compiled, kalman, prep.observables, base), dtype=np.float64)
    calib = cc(backend.build_calib_param_vector(compiled, base), dtype=np.float64)
    steady_c = cc(steady, dtype=np.float64)
    y_c = cc(prep.y_reordered, dtype=np.float64)
    p0_err, P0 = stationary_covariance(solved.policy.A, solved.policy.B, Q)
    assert p0_err == 0
    P0 = cc(P0, dtype=np.float64)

    ll, bk = obj_extended_base(
        compiled.construct_objective_cfunc().address,
        prep.meas_addr,
        prep.jac_addr,
        compiled.n_state,
        compiled.n_exog,
        len(prep.observables),
        steady_c,
        compiled._incidence,
        calib,
        Q,
        R,
        y_c,
        P0,
        float(prep.kf_jitter),
        int(prep.kf_sym),
    )

    ll_model = solved.kalman(y=y, filter_mode="extended", observables=obs).loglik

    assert bk == 0
    assert np.isfinite(ll)
    np.testing.assert_allclose(ll, ll_model, rtol=1e-9, atol=1e-9)


@pytest.fixture(scope="module")
def rbc_bundle(rbc_second_order_test_model_path):
    """Second-order RBC (levels model, nonzero steady state) for the unscented
    parity. The fixture has no kalman section, so a scalar measurement-noise
    configuration is supplied. The test derives its P0 from the solved policy,
    matching the runtime default."""
    model, _ = ModelParser(rbc_second_order_test_model_path).get_all()
    R = np.array([[1e-4]], dtype=np.float64)
    kalman = KalmanConfig(R=R)
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    # Levels model, but the config's own ss_seed already resolves Newton to the
    # steady state, so the solve needs no help. Feeding a resolved steady state
    # back in as a dense seed would not work anyway: it comes back in canonical
    # order and a dense seed is read in declaration order.
    solved = solver.solve(compiled=compiled, order=2)

    T = 40
    rng = np.random.default_rng(20260303)
    # x0 defaults to the steady state.
    sim = solved.sim(
        T=T,
        shocks={"e": rng.normal(0.0, 0.01, size=T)},
        observables=True,
    )
    y = pd.DataFrame({"c_obs": sim.observables["c_obs"][1:]})
    return {
        "compiled": compiled,
        "solved": solved,
        # The native kernel indexes its Newton seed by canonical position, which
        # is the order a resolved steady state comes back in.
        "seed": np.asarray(solved.policy.steady_state, dtype=np.float64),
        "y": y,
        "R": R,
    }


def test_obj_unscented_base_matches_model_kalman(rbc_bundle):
    compiled = rbc_bundle["compiled"]
    solved = rbc_bundle["solved"]
    seed = rbc_bundle["seed"]
    y = rbc_bundle["y"]
    R = rbc_bundle["R"]
    obs = ["c_obs"]
    n_state = compiled.n_state
    jitter, symmetrize = 1e-8, 1

    base = backend.extract_base_params(compiled)

    cc = np.ascontiguousarray
    Q = cc(backend.build_Q(compiled, base), dtype=np.float64)
    calib = cc(backend.build_calib_param_vector(compiled, base), dtype=np.float64)
    y_c = np.array(y.to_numpy(), dtype=np.float64, copy=True)

    # UKF augments the state. Its default P0 uses the stationary first-order
    # covariance in the first block and zeros in the second-order block.
    p0_err, P0_state = stationary_covariance(
        solved.policy.hx, solved.policy.B[:n_state, :], Q
    )
    assert p0_err == 0
    P0_ukf = np.zeros((2 * n_state, 2 * n_state), dtype=np.float64)
    P0_ukf[:n_state, :n_state] = P0_state

    ll, bk = obj_unscented_base(
        compiled.construct_objective_cfunc().address,
        compiled.construct_objective_cfunc_bicomplex().address,
        compiled.construct_measurement_cfunc(obs).address,
        n_state,
        compiled.n_exog,
        len(obs),
        cc(seed, dtype=np.float64),
        compiled._incidence,
        calib,
        Q,
        cc(R, dtype=np.float64),
        y_c,
        P0_ukf,
        float(jitter),
        int(symmetrize),
    )

    ll_model = solved.kalman(
        y=y,
        filter_mode="unscented",
        observables=obs,
        jitter=jitter,
        symmetrize=bool(symmetrize),
    ).loglik

    assert bk == 0
    assert np.isfinite(ll)
    np.testing.assert_allclose(ll, ll_model, rtol=1e-9, atol=1e-9)
