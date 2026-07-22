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
from SymbolicDSGE.kalman.config import KalmanConfig
from SymbolicDSGE.kalman.interface import FilterMode, _resolve_P0
from SymbolicDSGE._ckernels.estimation._estimation import (
    obj_linear_base,
    obj_unscented_base,
)


@pytest.fixture(scope="module")
def bundle(post82_test_model_path):
    model, kalman = ModelParser(post82_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    steady = np.zeros((len(compiled.var_names),), dtype=np.float64)
    solved = solver.solve(compiled=compiled, ss_seed=steady)

    params = model.calibration.parameters
    std_map = model.calibration.shock_std
    sig = {s: float(params[std_map[Symbol(s)]]) for s in ("e_g", "e_z", "e_r")}

    T = 24
    rng = np.random.default_rng(20260303)
    sim = solved.sim(
        T=T,
        shocks={
            "g": rng.normal(0.0, sig["e_g"], size=T),
            "z": rng.normal(0.0, sig["e_z"], size=T),
            "r": rng.normal(0.0, sig["e_r"], size=T),
        },
        x0=np.zeros((len(compiled.var_names),), dtype=np.float64),
        observables=True,
    )
    y = pd.DataFrame(
        {"OutGap": sim["OutGap"][1:], "Infl": sim["Infl"][1:], "Rate": sim["Rate"][1:]}
    )
    return {
        "compiled": compiled,
        "kalman": kalman,
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
    P0 = cc(prep.P0, dtype=np.float64)

    n_var = len(compiled.var_names)
    assert P0.shape == (n_var, n_var)

    ll, bk = obj_linear_base(
        compiled.construct_objective_cfunc().address,
        prep.meas_addr,
        prep.jac_addr,
        compiled.n_state,
        compiled.n_exog,
        len(prep.observables),
        0,  # log_linear
        steady_c,
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


@pytest.fixture(scope="module")
def rbc_bundle(rbc_second_order_test_model_path):
    """Second-order RBC (levels model, nonzero steady state) for the unscented
    parity. The fixture has no kalman section, so a minimal config is supplied:
    a uniform-diagonal P0 (so ``compile``'s variable permutation is a no-op) and
    a scalar measurement noise."""
    model, _ = ModelParser(rbc_second_order_test_model_path).get_all()
    n_var = 3  # z, k, c
    R = np.array([[1e-4]], dtype=np.float64)
    kalman = KalmanConfig(R=R, P0=np.eye(n_var, dtype=np.float64) * 0.1)
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    # Levels model: seed Newton from the resolved steady state, not zeros.
    seed = np.asarray(
        solver.solve(compiled=compiled, order=2).policy.steady_state, dtype=np.float64
    )
    solved = solver.solve(compiled=compiled, ss_seed=seed, order=2)

    T = 40
    rng = np.random.default_rng(20260303)
    sim = solved.sim(
        T=T,
        shocks={"z": rng.normal(0.0, 0.01, size=T)},
        x0=np.asarray(solved.policy.steady_state, dtype=np.float64),
        observables=True,
    )
    y = pd.DataFrame({"c_obs": sim["c_obs"][1:]})
    return {
        "compiled": compiled,
        "solved": solved,
        "seed": seed,
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

    # UKF augments the state: the native filter reads a 2*n_state P0, the block
    # expansion the interface applies for the oracle.
    P0_base = np.eye(n_state, dtype=np.float64) * 0.1
    P0_ukf = cc(_resolve_P0(FilterMode.UNSCENTED, n_state, P0_base), dtype=np.float64)

    ll, bk = obj_unscented_base(
        compiled.construct_objective_cfunc().address,
        compiled.construct_objective_cfunc_bicomplex().address,
        compiled.construct_measurement_cfunc(obs).address,
        n_state,
        compiled.n_exog,
        len(obs),
        cc(seed, dtype=np.float64),
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
