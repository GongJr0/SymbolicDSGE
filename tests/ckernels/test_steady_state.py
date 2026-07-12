"""Native steady-state Newton (#248): F(ss, ss) = 0 via klein_preproc's Jacobian.

The driver reuses the complex-step Jacobian (a - b) and the f64 LU solve; it is
the expansion-point solver for the native order-2 path. Validated against the
known RBC steady state (a stronger oracle than a same-math twin) and the trivial
zero steady state of a linear model.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.core._core import steady_state_newton
from SymbolicDSGE.core import DSGESolver, ModelParser


def _rbc():
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    compiled = DSGESolver(model, kalman).compile()
    calib = compiled.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)
    cf = compiled.construct_objective_cfunc()
    eq = compiled.equations
    # var order [z, k, c]; deterministic ss is (0, k_ss, c_ss).
    true_ss = np.array(
        [0.0, float(calib[sp.Symbol("k_ss")]), float(calib[sp.Symbol("c_ss")])]
    )
    return compiled, par, cf, eq, true_ss


def _resid_norm(eq, ss, par):
    r = eq(
        ss.astype(np.complex128), ss.astype(np.complex128), par.astype(np.complex128)
    )
    return float(np.max(np.abs(np.real(r))))


def test_newton_rbc_from_exact_seed():
    _compiled, par, cf, eq, true_ss = _rbc()
    ss, iters = steady_state_newton(cf.address, true_ss.copy(), par)
    assert iters <= 2
    assert _resid_norm(eq, ss, par) < 1e-10
    # z is exactly linear -> stays at 0; k, c match the (rounded) config to ~1e-6.
    np.testing.assert_allclose(ss, true_ss, rtol=1e-6, atol=1e-8)


def test_newton_rbc_from_perturbed_seed():
    _compiled, par, cf, eq, true_ss = _rbc()
    seed = true_ss * np.array([0.0, 1.1, 0.9]) + np.array([0.05, 0.0, 0.0])
    ss, iters = steady_state_newton(cf.address, seed, par)
    assert 1 <= iters <= 20
    assert _resid_norm(eq, ss, par) < 1e-10
    np.testing.assert_allclose(ss, true_ss, rtol=1e-6, atol=1e-8)


def test_newton_non_convergence_raises():
    # One iteration from a far seed cannot reach tol -> the driver reports failure
    # rather than returning a bogus point.
    _compiled, par, cf, _eq, true_ss = _rbc()
    seed = true_ss * np.array([0.0, 2.0, 0.5]) + np.array([0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="did not converge"):
        steady_state_newton(cf.address, seed, par, max_iter=1)


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_newton_linear_model_zero_steady_state(path):
    # A (log-)linear model clears at ss = 0; Newton seeded there converges at once.
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    calib = compiled.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)
    cf = compiled.construct_objective_cfunc()
    seed = np.zeros(len(compiled.var_names), dtype=np.float64)
    ss, iters = steady_state_newton(cf.address, seed, par)
    assert iters == 0
    np.testing.assert_allclose(ss, 0.0, atol=1e-12)
