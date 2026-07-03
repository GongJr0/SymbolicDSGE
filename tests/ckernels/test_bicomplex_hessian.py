"""Piece B (#248): native ``bicomplex_hessian`` sweep vs analytic Hessians.

The residual Hessian is validated against *known* second derivatives (a stronger
oracle than a same-math numba twin): a polynomial (exact) and a transcendental
(within the bicomplex accuracy budget). The stacked arg order is z = (fwd, cur),
so ``F_xx[eq]`` is 2n x 2n with the fwd block first.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.core._core import bicomplex_hessian
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.residual_printer import (
    BicomplexOps,
    ResidualLayout,
    build_cfunc,
)


def test_bicomplex_hessian_polynomial_is_exact():
    # F = 2*fwd^2 + 3*fwd*cur + cur^3 (1 var). Hessian over z = (fwd, cur):
    #   d2/dfwd2 = 4, d2/dfwd dcur = 3, d2/dcur2 = 6*cur.  At ss = 1 -> [[4,3],[3,6]].
    fwd_x, cur_x = sp.symbols("fwd_x cur_x")
    expr = 2 * fwd_x**2 + 3 * fwd_x * cur_x + cur_x**3
    layout = ResidualLayout(
        slot={fwd_x: ("fwd", 0), cur_x: ("cur", 0)}, n_var=1, n_par=0, n_eq=1
    )
    cf = build_cfunc([expr], layout, BicomplexOps())  # hold: keep .address valid

    H = bicomplex_hessian(cf.address, np.array([1.0]), np.zeros(0), 1)

    expected = np.array([[[4.0, 3.0], [3.0, 6.0]]])
    np.testing.assert_allclose(H, expected, rtol=1e-6, atol=1e-7)


def test_bicomplex_hessian_transcendental():
    # F = fwd * exp(cur) (1 var). d2/dfwd2 = 0, d2/dfwd dcur = exp(cur),
    # d2/dcur2 = fwd*exp(cur). At ss = 1 -> [[0, e], [e, e]].
    fwd_x, cur_x = sp.symbols("fwd_x cur_x")
    expr = fwd_x * sp.exp(cur_x)
    layout = ResidualLayout(
        slot={fwd_x: ("fwd", 0), cur_x: ("cur", 0)}, n_var=1, n_par=0, n_eq=1
    )
    cf = build_cfunc([expr], layout, BicomplexOps())

    H = bicomplex_hessian(cf.address, np.array([1.0]), np.zeros(0), 1)

    e = float(np.exp(1.0))
    expected = np.array([[[0.0, e], [e, e]]])
    np.testing.assert_allclose(H, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_bicomplex_hessian_linear_model_is_zero(path):
    # A (log-)linear model has an identically-zero residual Hessian. Exercises
    # the CompiledModel bicomplex-cfunc accessor + real model dims end-to-end.
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    layout = ResidualLayout.from_compiled(compiled)
    cf = compiled.construct_objective_cfunc_bicomplex()

    ss = np.zeros(layout.n_var, dtype=np.float64)
    par = np.array(
        [
            float(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=np.float64,
    )

    H = bicomplex_hessian(cf.address, ss, par, layout.n_eq)

    n2 = 2 * layout.n_var
    assert H.shape == (layout.n_eq, n2, n2)
    np.testing.assert_allclose(H, 0.0, atol=1e-6)
