"""Parity: native ``klein_preproc`` vs numba ``_approximate_system_numeric``.

The native driver (``_ckernels/core/klein_preproc.c``) runs the complex step
first order sweep in C and calls the printer residual cfunc by address. The
reference path drives the same cfunc from Python through its own complex step
loop. Same step, same arithmetic, and same output.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core._core import klein_preprocess
from SymbolicDSGE.core import DSGESolver, ModelParser
from _oracles.core import _approximate_system_numeric
from SymbolicDSGE._symbolic_printers import ResidualLayout, build_cfunc
from _residual_call import residual_caller

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
    eq_func = residual_caller(compiled.objective_eqs, layout)

    ss = np.zeros(layout.n_var, dtype=np.float64)
    par = _params(compiled)

    a_ref, b_ref, c_ref, d_ref = _approximate_system_numeric(
        eq_func, ss, par, layout.n_exog
    )
    a, b, c, d = klein_preprocess(
        eq_func.cfunc.address, ss, par, layout.n_var, layout.n_exog
    )

    assert a.shape == (layout.n_var, layout.n_var)
    assert b.shape == (layout.n_var, layout.n_var)
    assert c.shape == (layout.n_var, layout.n_var)
    assert d.shape == (layout.n_var, layout.n_exog)
    np.testing.assert_allclose(a, a_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(b, b_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(c, c_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(d, d_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_the_lag_and_shock_blocks_carry_the_model(path):
    """Both models lag something and are driven by shocks, so both blocks are live.

    A lag of one and a shock reach the printed residual as they were written, so
    ``c`` and ``d`` are where they land. Zeros here would mean the residual lost
    them on the way to the pencil, which the ``a``/``b`` goldens would not
    announce.
    """
    compiled = _compiled(path)
    layout = ResidualLayout.from_compiled(compiled)
    cf = build_cfunc(compiled.objective_eqs, layout)

    ss = np.zeros(layout.n_var, dtype=np.float64)
    _, _, c, d = klein_preprocess(
        cf.address, ss, _params(compiled), layout.n_var, layout.n_exog
    )

    assert c.any()
    assert d.any()
    # One column per lagged variable, one per innovation that reaches a row.
    assert np.count_nonzero(c.any(axis=0)) == compiled.n_state
    assert np.count_nonzero(d.any(axis=0)) == compiled.n_exog
