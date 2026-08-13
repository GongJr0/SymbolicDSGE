"""Parity: native ``residual_path`` vs per step numba residual evaluation.

This is the Den Haan Marcet native backend. It reuses the solve cfunc to build
the residuals matrix and avoids a numba residual compile.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core._core import residual_path
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE._symbolic_printers import ResidualLayout, build_cfunc


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_residual_path_matches_reference(path):
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    layout = ResidualLayout.from_compiled(compiled)
    cf = build_cfunc(compiled.objective_eqs, layout)  # hold: keeps .address valid

    n_var, n_steps = layout.n_var, 20
    rng = np.random.default_rng(3)
    cur = rng.normal(size=(n_steps, n_var)).astype(np.complex128)
    fwd = rng.normal(size=(n_steps, n_var)).astype(np.complex128)
    par = np.array(
        [
            float(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=np.complex128,
    )

    prev = rng.normal(size=(n_steps, n_var)).astype(np.complex128)
    eps = np.zeros((n_steps, layout.n_exog), dtype=np.complex128)

    got = residual_path(cf.address, cur, fwd, prev, eps, par, n_var)
    assert got.shape == (n_steps, n_var)

    eq = compiled.equations
    want = np.empty((n_steps, n_var), dtype=np.float64)
    for t in range(n_steps):
        want[t] = eq(fwd[t], cur[t], prev[t], eps[t], par).real

    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
