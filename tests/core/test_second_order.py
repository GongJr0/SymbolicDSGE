"""Second order assembly, independent of any Dynare golden.

First order residual consistency and zero tensors for a linear model, over the
two shipped MODELS. Dynare parity lives in test_second_order_rbc.py and
test_second_order_multishock.py, each next to its own golden module.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core._core import bicomplex_hessian, klein_preprocess
from SymbolicDSGE._ckernels.core import second_order
from SymbolicDSGE._symbolic_printers import ResidualLayout
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solver_backend import klein_solve
from _oracles.core import first_order_residual


def _drive(path):
    """Compile a model and run the full second-order preproc chain at ss = 0."""
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    layout = ResidualLayout.from_compiled(compiled)
    n_eq, n_state = layout.n_var, compiled.n_state

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

    a, b = klein_preprocess(cf.address, ss, par, n_eq)
    sol = klein_solve(cf, par, ss, n_state)
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
    gxx, hxx = second_order(a, b, f_xx, gx, hx, n_state)

    nx = n_state
    ny = gx.shape[0]
    assert gxx.shape == (ny, nx, nx)
    assert hxx.shape == (nx, nx, nx)
    np.testing.assert_allclose(gxx, 0.0, atol=1e-6)
    np.testing.assert_allclose(hxx, 0.0, atol=1e-6)
