"""Native second-order assembly (#248): sdsge_second_order vs the numpy oracle.

The C kernel is the allocation-free row-major transcription of
core.second_order.solve_second_order; here it is checked to reproduce that oracle
to machine precision on models of different (n, nx, ny) shapes. The numpy side is
itself pinned to Dynare in tests/core/test_second_order.py, so parity here plus
that golden chains the native path to the independent solver.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.core._core import (
    bicomplex_hessian,
    klein_preprocess,
    second_order,
    second_order_risk,
)
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.klein import klein_solve
from SymbolicDSGE.core.second_order import (
    _solve_second_order_numpy,
    _solve_second_order_risk_numpy,
)


def _drive(path, ss_map=None):
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    n_eq = len(compiled.var_names)
    n_state = compiled.n_state
    calib = compiled.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)
    cf = compiled.construct_objective_cfunc()
    cf_bc = compiled.construct_objective_cfunc_bicomplex()
    eq = compiled.construct_objective_vector_func()

    if ss_map is not None:
        ss = np.array([ss_map[nm] for nm in compiled.var_names], dtype=np.float64)
    else:
        ss = np.zeros(n_eq, dtype=np.float64)

    a, b = klein_preprocess(cf.address, ss, par, n_eq, False)
    sol = klein_solve(eq, par, ss, n_state, residual_cfunc=cf)
    gx, hx = np.real(sol.f), np.real(sol.p)
    f_xx = bicomplex_hessian(cf_bc.address, ss, par, n_eq)
    eta = DSGESolver._build_eta(compiled)
    return a, b, f_xx, gx, hx, n_state, eta


def _rbc_ss():
    model, _ = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    calib = model.calibration.parameters
    return {
        "z": 0.0,
        "k": float(calib[sp.Symbol("k_ss")]),
        "c": float(calib[sp.Symbol("c_ss")]),
    }


@pytest.mark.parametrize(
    "path, ss_map",
    [
        ("tests/fixtures/models/rbc_second_order.yaml", _rbc_ss()),  # n=3, nx=2, ny=1
        ("MODELS/test.yaml", None),  # n=6, nx=3, ny=3
        ("MODELS/POST82.yaml", None),  # n=5, nx=3, ny=2
    ],
)
def test_native_second_order_matches_numpy(path, ss_map):
    a, b, f_xx, gx, hx, n_state, eta = _drive(path, ss_map)

    gxx_np, hxx_np = _solve_second_order_numpy(a, b, f_xx, gx, hx, n_state)
    gxx_c, hxx_c = second_order(a, b, f_xx, gx, hx, n_state)
    np.testing.assert_allclose(gxx_c, gxx_np, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(hxx_c, hxx_np, rtol=0.0, atol=1e-12)

    # Risk correction shares the same gxx input so parity isolates the risk step.
    gss_np, hss_np = _solve_second_order_risk_numpy(
        a, b, f_xx, gx, gxx_np, eta, n_state
    )
    gss_c, hss_c = second_order_risk(a, b, f_xx, gx, gxx_np, eta, n_state)
    np.testing.assert_allclose(gss_c, gss_np, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(hss_c, hss_np, rtol=0.0, atol=1e-12)
