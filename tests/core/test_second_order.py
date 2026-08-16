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

    a, b, c, _ = klein_preprocess(cf.address, ss, par, n_eq, compiled.n_exog)
    sol = klein_solve(cf, par, ss, compiled._incidence, n_state, n_exog=compiled.n_exog)
    f_xx = bicomplex_hessian(cf_bc.address, ss, par, compiled.n_exog, n_eq)
    return (
        a,
        b,
        c,
        f_xx,
        np.real(sol.f),
        np.real(sol.p),
        np.real(sol.B),
        DSGESolver._build_Q(compiled),
        n_state,
    )


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_first_order_foc_holds(path):
    a, b, c, _f_xx, gx, hx, _bu, _Q, n_state = _drive(path)
    foc = first_order_residual(a, b, c, gx, hx, n_state)
    np.testing.assert_allclose(foc, 0.0, atol=1e-8)


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_linear_model_has_zero_second_order(path):
    a, b, _c, f_xx, gx, hx, bu, Q, n_state = _drive(path)
    gxx, hxx, gxu, hxu, guu, huu, gss, hss = second_order(
        a, b, f_xx, gx, hx, bu, Q, n_state
    )

    nx, ny, ne = n_state, gx.shape[0], bu.shape[1]
    assert (gxx.shape, hxx.shape) == ((ny, nx, nx), (nx, nx, nx))
    assert (gxu.shape, hxu.shape) == ((ny, nx, ne), (nx, nx, ne))
    assert (guu.shape, huu.shape) == ((ny, ne, ne), (nx, ne, ne))
    assert (gss.shape, hss.shape) == ((ny,), (nx,))
    for block in (gxx, hxx, gxu, hxu, guu, huu, gss, hss):
        np.testing.assert_allclose(block, 0.0, atol=1e-6)
