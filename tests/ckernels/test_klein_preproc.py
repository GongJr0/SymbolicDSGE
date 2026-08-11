"""Parity: native ``klein_preproc`` vs numba ``_approximate_system_numeric``.

The native driver (``_ckernels/core/klein_preproc.c``) runs the complex step
first order sweep in C and calls the printer residual cfunc by address. The
numba path runs the same printer residual through the njit vector function and
its own complex step loop. Same step, same arithmetic, and same output.
"""

from __future__ import annotations

import numpy as np
import pytest

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

    a_ref, b_ref = _approximate_system_numeric(eq_func, ss, par)
    a, b = klein_preprocess(cf.address, ss, par, layout.n_var)

    assert a.shape == (layout.n_var, layout.n_var)
    assert b.shape == (layout.n_var, layout.n_var)
    np.testing.assert_allclose(a, a_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(b, b_ref, rtol=RTOL, atol=ATOL)
