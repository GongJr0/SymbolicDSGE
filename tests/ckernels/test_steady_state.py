"""Native steady-state Newton (#248): F(ss, ss) = 0 via klein_preproc's Jacobian.

The driver reuses the complex-step Jacobian (a - b) and the f64 LU solve; it is
the expansion-point solver for the native order-2 path. Validated against the
known RBC steady state (a stronger oracle than a same-math twin) and the trivial
zero steady state of a linear model.
"""

from __future__ import annotations

import re

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
    # The seed sizes the solve, so it spans the compiled layout: a lag aux sits
    # where its origin does, the lifted shock and tfp at 0.
    levels = {
        "k": float(calib[sp.Symbol("k_ss")]),
        "c": float(calib[sp.Symbol("c_ss")]),
    }
    true_ss = np.array(
        [levels.get(re.sub(r"_lag\d+$", "", name), 0.0) for name in compiled.var_names],
        dtype=np.float64,
    )
    return compiled, par, cf, eq, true_ss


def _perturbed(compiled, true_ss, capital, consumption, tfp):
    """A seed off the steady state, scaling every copy of a variable alike."""
    seed = true_ss.copy()
    for name in ("k", "k_lag1"):
        seed[compiled.idx[name]] *= capital
    seed[compiled.idx["c"]] *= consumption
    for name in ("z", "z_lag1"):
        seed[compiled.idx[name]] += tfp
    return seed


def _resid_norm(eq, ss, par, n_exog):
    # The steady state holds every date at the same point, with no innovation.
    point = ss.astype(np.complex128)
    r = eq(
        point,
        point,
        point,
        np.zeros(n_exog, np.complex128),
        par.astype(np.complex128),
    )
    return float(np.max(np.abs(np.real(r))))


def test_newton_rbc_from_exact_seed():
    _compiled, par, cf, eq, true_ss = _rbc()
    ss, iters = steady_state_newton(cf.address, true_ss.copy(), par, _compiled.n_exog)
    assert iters <= 2
    assert _resid_norm(eq, ss, par, _compiled.n_exog) < 1e-10
    # z is exactly linear -> stays at 0; k, c match the (rounded) config to ~1e-6.
    np.testing.assert_allclose(ss, true_ss, rtol=1e-6, atol=1e-8)


def test_newton_rbc_from_perturbed_seed():
    compiled, par, cf, eq, true_ss = _rbc()
    seed = _perturbed(compiled, true_ss, capital=1.1, consumption=0.9, tfp=0.05)
    ss, iters = steady_state_newton(cf.address, seed, par, compiled.n_exog)
    assert 1 <= iters <= 20
    assert _resid_norm(eq, ss, par, compiled.n_exog) < 1e-10
    np.testing.assert_allclose(ss, true_ss, rtol=1e-6, atol=1e-8)


def test_newton_non_convergence_raises():
    # One iteration from a far seed cannot reach tol -> the driver reports failure
    # rather than returning a bogus point.
    compiled, par, cf, _eq, true_ss = _rbc()
    seed = _perturbed(compiled, true_ss, capital=2.0, consumption=0.5, tfp=0.5)
    with pytest.raises(ValueError, match="did not converge"):
        steady_state_newton(cf.address, seed, par, compiled.n_exog, max_iter=1)


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_newton_linear_model_zero_steady_state(path):
    # A (log-)linear model clears at ss = 0; Newton seeded there converges at once.
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    calib = compiled.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)
    cf = compiled.construct_objective_cfunc()
    seed = np.zeros(len(compiled.var_names), dtype=np.float64)
    ss, iters = steady_state_newton(cf.address, seed, par, compiled.n_exog)
    assert iters == 0
    np.testing.assert_allclose(ss, 0.0, atol=1e-12)
