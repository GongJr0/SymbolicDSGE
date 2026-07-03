"""Piece A (#248 second order): numba bicomplex ops + the BicomplexOps printer.

Two checks:
  1. the numba bicomplex primitives match the C ``bc256`` oracle (via ``_core``);
  2. the printer's BicomplexOps backend emits a residual whose bicomplex-step
     second derivative recovers the analytic Hessian.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE.core import bicomplex as bcn
from SymbolicDSGE.core.residual_printer import (
    BicomplexOps,
    ResidualLayout,
    build_cfunc,
    build_njit,
)
from SymbolicDSGE._ckernels.core._core import (
    bc_add as c_add,
    bc_div as c_div,
    bc_exp as c_exp,
    bc_log as c_log,
    bc_mul as c_mul,
    bc_neg as c_neg,
    bc_real_scale as c_real_scale,
    bc_sqrt as c_sqrt,
    bc_sub as c_sub,
)

C = np.complex128


def _to_numba(t):
    # (re, i, j, ij) -> (a, b) with a, b complex128
    return (complex(t[0], t[1]), complex(t[2], t[3]))


def _from_numba(v):
    return (v[0].real, v[0].imag, v[1].real, v[1].imag)


@pytest.mark.parametrize("seed", range(6))
def test_numba_bicomplex_ops_match_c_oracle(seed):
    rng = np.random.default_rng(seed)

    def rand():
        # positive-real-dominant so log/sqrt stay on the principal branch
        return (float(rng.uniform(0.5, 2.0)), *rng.uniform(-0.3, 0.3, size=3))

    x4, y4 = rand(), rand()
    xn, yn = _to_numba(x4), _to_numba(y4)

    for nfn, cfn in [
        (bcn.bc_add, c_add),
        (bcn.bc_sub, c_sub),
        (bcn.bc_mul, c_mul),
        (bcn.bc_div, c_div),
    ]:
        np.testing.assert_allclose(
            _from_numba(nfn(xn, yn)), cfn(x4, y4), rtol=1e-12, atol=1e-13
        )

    for nfn, cfn in [
        (bcn.bc_neg, c_neg),
        (bcn.bc_exp, c_exp),
        (bcn.bc_log, c_log),
        (bcn.bc_sqrt, c_sqrt),
    ]:
        np.testing.assert_allclose(
            _from_numba(nfn(xn)), cfn(x4), rtol=1e-11, atol=1e-12
        )

    s = float(rng.uniform(-2.0, 2.0))
    np.testing.assert_allclose(
        _from_numba(bcn.bc_real_scale(xn, s)),
        c_real_scale(x4, s),
        rtol=1e-12,
        atol=1e-13,
    )


def _second_derivative(expr, x0, h=1e-4):
    """f''(x0) via the printer's bicomplex residual: perturb x on i and j, read
    the ij component / h^2."""
    x = sp.Symbol("x")
    layout = ResidualLayout(slot={x: ("cur", 0)}, n_var=1, n_par=0, n_eq=1)
    res = build_njit([expr(x)], layout, BicomplexOps())
    cur = np.array([complex(x0, h), complex(h, 0.0)], dtype=C)  # x + h*i + h*j
    out = res(np.zeros(2, C), cur, np.zeros(0, C))
    return out[1].imag / h**2  # ij component


@pytest.mark.parametrize(
    "expr,analytic,x0,tol",
    [
        (lambda x: x**3 + 2 * x**2, lambda v: 6 * v + 4, 1.5, 1e-8),  # exact
        (lambda x: sp.exp(x), np.exp, 0.3, 1e-5),
        (lambda x: sp.log(x), lambda v: -1.0 / v**2, 1.4, 1e-5),
        (lambda x: x ** sp.Rational(5, 2), lambda v: 2.5 * 1.5 * v**0.5, 1.4, 1e-5),
    ],
)
def test_bicomplex_printer_second_derivative(expr, analytic, x0, tol):
    assert _second_derivative(expr, x0) == pytest.approx(analytic(x0), rel=tol)


def test_build_cfunc_bicomplex_compiles_to_address():
    x, p = sp.symbols("x p")
    layout = ResidualLayout(
        slot={x: ("cur", 0), p: ("par", 0)}, n_var=1, n_par=1, n_eq=1
    )
    cf = build_cfunc([p * sp.exp(x)], layout, BicomplexOps())
    assert isinstance(cf.address, int) and cf.address != 0
