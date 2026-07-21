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
from SymbolicDSGE._ckernels.estimation._estimation import obj_linear_base


@pytest.fixture(scope="module")
def bundle(post82_test_model_path):
    model, kalman = ModelParser(post82_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    steady = np.zeros((len(compiled.var_names),), dtype=np.float64)
    solved = solver.solve(compiled=compiled, steady_state=steady)

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
    zero_state = cc(prep.zero_state, dtype=np.float64)

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
        zero_state,
        float(prep.kf_jitter),
        int(prep.kf_sym),
    )

    ll_model = solved.kalman(y=y, filter_mode="linear", observables=obs).loglik

    assert bk == 0
    assert np.isfinite(ll)
    np.testing.assert_allclose(ll, ll_model, rtol=1e-9, atol=1e-9)
