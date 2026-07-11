"""Parity: native ``klein_preproc`` vs numba ``_approximate_system_numeric``.

The native driver (``_ckernels/core/klein_preproc.c``) runs the complex step
first order sweep in C and calls the printer residual cfunc by address. The
numba path runs the same printer residual through the njit vector function and
its own complex step loop. Same step, same arithmetic, and same output.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.core._core import klein_preprocess
from SymbolicDSGE.core import DSGESolver, ModelParser
from _oracles.core import _approximate_system_numeric
from SymbolicDSGE._symbolic_printers import (
    ResidualLayout,
    build_cfunc,
    build_njit,
)

RTOL = 1e-10
ATOL = 1e-12


def _compiled(path: str):
    model, kalman = ModelParser(path).get_all()
    return DSGESolver(model, kalman).compile()


def _params(compiled) -> np.ndarray:
    return np.array(
        [
            float(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_klein_preproc_parity(path):
    compiled = _compiled(path)
    layout = ResidualLayout.from_compiled(compiled)
    cf = build_cfunc(compiled.objective_eqs, layout)  # hold: keeps .address valid
    eq_func = build_njit(compiled.objective_eqs, layout)

    ss = np.zeros(layout.n_var, dtype=np.float64)
    par = _params(compiled)

    a_ref, b_ref = _approximate_system_numeric(eq_func, ss, par, False)
    a, b = klein_preprocess(cf.address, ss, par, layout.n_eq, False)

    assert a.shape == (layout.n_eq, layout.n_var)
    assert b.shape == (layout.n_eq, layout.n_var)
    np.testing.assert_allclose(a, a_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(b, b_ref, rtol=RTOL, atol=ATOL)


def test_klein_preproc_log_linear_parity():
    # Hand-built residual with a positive steady state, so the log-linear wrap
    # log(f(exp(.), exp(.)) + 1) stays on the principal branch (real, finite).
    fwd_x, cur_x, p = sp.symbols("fwd_x cur_x p")
    exprs = [p * fwd_x - cur_x]
    layout = ResidualLayout(
        slot={fwd_x: ("fwd", 0), cur_x: ("cur", 0), p: ("par", 0)},
        n_var=1,
        n_par=1,
        n_eq=1,
    )
    cf = build_cfunc(exprs, layout)
    eq_func = build_njit(exprs, layout)

    ss = np.array([1.0], dtype=np.float64)
    par = np.array([0.5], dtype=np.float64)

    a_ref, b_ref = _approximate_system_numeric(eq_func, ss, par, True)
    a, b = klein_preprocess(cf.address, ss, par, 1, True)

    np.testing.assert_allclose(a, a_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(b, b_ref, rtol=RTOL, atol=ATOL)
