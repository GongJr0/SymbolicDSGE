"""Piece C (#248): the SGU second-order assembly transpile.

1. the first-order FOC residual is ~0 -- guards the adapter (block slicing +
   the -b sign) independently of gxx;
2. a (log-)linear model has an identically-zero second-order solution -- guards
   the assembly + solve plumbing;
3. the RBC g_xx/h_xx match Dynare's ghxx (with the [k,z]->[z,k] reorder and the
   (1/rho)^m timing map) -- the independent-solver check on the actual math.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.core._core import bicomplex_hessian, klein_preprocess
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.klein import klein_solve
from SymbolicDSGE.core.residual_printer import ResidualLayout
from SymbolicDSGE.core.second_order import first_order_residual, solve_second_order

# Dynare stoch_simul(order=2) on tests/fixtures/models/rbc_second_order.mod,
# untouched full precision. Rows are DR order [k', z (linear -> 0), c]; the four
# columns are the state-pair second derivatives in Dynare's state order [k, z],
# i.e. [kk, kz, zk, zz].
_DYNARE_GHXX_KPRIME = [
    -0.00020831558951371761,
    0.029121140800089713,
    0.029121140800089713,
    2.2629131783555407,
]
_DYNARE_GHXX_C = [
    -0.0006212783838050477,
    0.004224819272851745,
    0.004224819272851745,
    0.45842005940556829,
]
# ghs2 (sigma^2 risk correction), DR order [k', z, c] -- for the future gss_hss.
_DYNARE_GHS2 = [0.0010614857740643515, 0.0, -0.0010614857740643515]


def _dynare_to_our_convention(dyn_row: list[float], rho: float) -> np.ndarray:
    """Map a Dynare ghxx row ([kk, kz, zk, zz] in state order [k, z]) to our 2x2
    Hessian in state order [z, k]. Our offset-0/+1 timing puts the innovation on
    z(t+1) vs Dynare's z(t), so our z-state = rho * Dynare's -> every z-derivative
    carries 1/rho (k is unaffected: same predetermined stock in both datings)."""
    kk, kz, _zk, zz = dyn_row
    return np.array([[zz / rho**2, kz / rho], [kz / rho, kk]])


def _drive(path):
    """Compile a model and run the full second-order preproc chain at ss = 0."""
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    layout = ResidualLayout.from_compiled(compiled)
    n_eq, n_state = layout.n_eq, compiled.n_state

    ss = np.zeros(layout.n_var, dtype=np.float64)
    par = np.array(
        [
            float(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=np.float64,
    )
    cf = compiled.construct_objective_cfunc()
    cf_bc = compiled.construct_objective_cfunc_bicomplex()
    eq = compiled.construct_objective_vector_func()

    a, b = klein_preprocess(cf.address, ss, par, n_eq, False)
    sol = klein_solve(eq, par, ss, n_state, residual_cfunc=cf)
    gx, hx = np.real(sol.f), np.real(sol.p)
    f_xx = bicomplex_hessian(cf_bc.address, ss, par, n_eq)
    return a, b, f_xx, gx, hx, n_state


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_first_order_foc_holds(path):
    a, b, _f_xx, gx, hx, n_state = _drive(path)
    foc = first_order_residual(a, b, gx, hx, n_state)
    np.testing.assert_allclose(foc, 0.0, atol=1e-8)


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_linear_model_has_zero_second_order(path):
    a, b, f_xx, gx, hx, n_state = _drive(path)
    gxx, hxx = solve_second_order(a, b, f_xx, gx, hx, n_state)

    nx = n_state
    ny = gx.shape[0]
    assert gxx.shape == (ny, nx, nx)
    assert hxx.shape == (nx, nx, nx)
    np.testing.assert_allclose(gxx, 0.0, atol=1e-6)
    np.testing.assert_allclose(hxx, 0.0, atol=1e-6)


def test_rbc_second_order_matches_dynare_ghxx():
    """The golden: our g_xx/h_xx on the RBC match Dynare's ghxx (after the
    [k,z]->[z,k] reorder + the (1/rho)^m timing map). This is the check that
    exercises the actual second-order math against an independent solver."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    compiled = DSGESolver(model, kalman).compile()
    assert list(compiled.var_names) == ["z", "k", "c"]  # states [z, k], control c

    layout = ResidualLayout.from_compiled(compiled)
    n_eq, n_state = layout.n_eq, compiled.n_state
    calib = compiled.config.calibration.parameters
    rho = float(calib[sp.Symbol("rho")])
    ss_map = {
        "z": 0.0,
        "k": float(calib[sp.Symbol("k_ss")]),
        "c": float(calib[sp.Symbol("c_ss")]),
    }
    ss = np.array([ss_map[nm] for nm in compiled.var_names], dtype=np.float64)
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)

    cf = compiled.construct_objective_cfunc()
    cf_bc = compiled.construct_objective_cfunc_bicomplex()
    eq = compiled.construct_objective_vector_func()

    # Steady state actually clears the residual.
    resid = eq(
        ss.astype(np.complex128), ss.astype(np.complex128), par.astype(np.complex128)
    )
    np.testing.assert_allclose(np.real(resid), 0.0, atol=1e-7)

    a, b = klein_preprocess(cf.address, ss, par, n_eq, False)
    sol = klein_solve(eq, par, ss, n_state, residual_cfunc=cf)
    assert sol.stab == 0
    gx, hx = np.real(sol.f), np.real(sol.p)
    f_xx = bicomplex_hessian(cf_bc.address, ss, par, n_eq)
    gxx, hxx = solve_second_order(a, b, f_xx, gx, hx, n_state)

    # gxx[0] = c (the single control); hxx[1] = k' (state index 1); hxx[0] = z'.
    np.testing.assert_allclose(
        gxx[0], _dynare_to_our_convention(_DYNARE_GHXX_C, rho), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        hxx[1],
        _dynare_to_our_convention(_DYNARE_GHXX_KPRIME, rho),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(hxx[0], 0.0, atol=1e-6)  # z' is linear
