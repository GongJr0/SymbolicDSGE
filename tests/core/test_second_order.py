"""Piece C (#248): the SGU second-order assembly transpile.

1. the first-order FOC residual is ~0 -- guards the adapter (block slicing +
   the -b sign) independently of gxx;
2. a (log-)linear model has an identically-zero second-order solution -- guards
   the assembly + solve plumbing;
3. the RBC g_xx/h_xx match Dynare's ghxx (with the [k,z]->[z,k] reorder and the
   (1/rho)^m timing map), and the risk correction g_ss/h_ss match ghs2 directly
   (a constant -> no timing factor) -- the independent-solver check on the math.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.core._core import bicomplex_hessian, klein_preprocess
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.klein import klein_solve
from SymbolicDSGE.core.residual_printer import ResidualLayout
from SymbolicDSGE.core.second_order import (
    first_order_residual,
    solve_second_order,
    solve_second_order_risk,
)

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


def test_rbc_second_order_matches_dynare():
    """The golden: our g_xx/h_xx match Dynare's ghxx (after the [k,z]->[z,k]
    reorder + the (1/rho)^m timing map), and the risk correction g_ss/h_ss matches
    ghs2 directly (a constant, so no timing factor). The independent-solver check
    on the actual second-order math."""
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

    # Risk correction vs ghs2: eta loads the single shock (std sig) on z (state 0);
    # x' = h(x) + eta @ eps. No timing factor -- g_ss/h_ss are constants.
    sig = float(calib[sp.Symbol("sig")])
    eta = np.zeros((n_state, 1), dtype=np.float64)
    eta[0, 0] = sig
    gss, hss = solve_second_order_risk(a, b, f_xx, gx, gxx, eta, n_state)
    # ours [g_ss(c); h_ss(z', k')] -> Dynare DR order [k', z, c]
    np.testing.assert_allclose(
        [hss[1], hss[0], gss[0]], _DYNARE_GHS2, rtol=1e-4, atol=1e-8
    )


def test_solve_order2_wiring():
    """The .solve(order=2) public path end to end: it resolves + cross-checks the
    nonlinear steady state, builds eta from the shock calibration, and returns a
    PerturbationSolution whose tensors match the Dynare goldens. order=1 is
    unchanged (a plain KleinSolution with no second-order fields)."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    solved = solver.solve(compiled, order=2)
    pol = solved.policy
    assert pol.order == 2

    calib = compiled.config.calibration.parameters
    rho = float(calib[sp.Symbol("rho")])
    k_ss, c_ss = float(calib[sp.Symbol("k_ss")]), float(calib[sp.Symbol("c_ss")])
    # steady state was solved/validated to the nonlinear point (order [z, k, c]).
    np.testing.assert_allclose(
        pol.steady_state, [0.0, k_ss, c_ss], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        pol.gxx[0], _dynare_to_our_convention(_DYNARE_GHXX_C, rho), rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        pol.hxx[1],
        _dynare_to_our_convention(_DYNARE_GHXX_KPRIME, rho),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [pol.hss[1], pol.hss[0], pol.gss[0]], _DYNARE_GHS2, rtol=1e-4, atol=1e-8
    )
    # First order path is untouched: KleinSolution, no second-order tensors.
    # (levels model -> the expansion point must be supplied; zeros would fail BK.)
    first = solver.solve(compiled, order=1, steady_state=[0.0, k_ss, c_ss])
    assert first.policy.order == 1
    assert not hasattr(first.policy, "gxx")


def test_solve_order2_rejects_inconsistent_steady_state():
    """A steady_state= that does not clear F(ss, ss) is rejected rather than used
    as a bad expansion point."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    with pytest.raises(ValueError, match="disagrees with the numerically solved"):
        solver.solve(compiled, order=2, steady_state=[0.0, 1.0, 1.0])
