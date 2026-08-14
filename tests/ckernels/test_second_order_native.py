"""Native second-order assembly (#248): sdsge_second_order vs the numpy oracle.

The C kernel is the allocation-free row-major transcription of
core.second_order.solve_second_order; here it is checked to reproduce that oracle
to machine precision on models of different (n, nx, ny) shapes. The numpy side is
itself pinned to Dynare in tests/core/test_second_order_rbc.py and
tests/core/test_second_order_multishock.py, so parity here plus those goldens
chains the native path to the independent solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core._core import (
    bicomplex_hessian,
    klein_preprocess,
    second_order,
)
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solver_backend import klein_solve
from _oracles.core import _solve_second_order_numpy

_BLOCKS = ("gxx", "hxx", "gxu", "hxu", "guu", "huu", "gss", "hss")


def _drive(path):
    model, kalman = ModelParser(path).get_all()
    compiled = DSGESolver(model, kalman).compile()
    n_eq = len(compiled.var_names)
    n_state = compiled.n_state
    calib = compiled.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)
    cf = compiled.construct_objective_cfunc()
    cf_bc = compiled.construct_objective_cfunc_bicomplex()

    # Config steady state, resolved at run time (RBC -> [0, k_ss, c_ss];
    # deviation-form models -> 0). No file I/O at collection time.
    ss = DSGESolver._resolve_ss_seed(None, compiled)

    a, b, _, _ = klein_preprocess(cf.address, ss, par, n_eq, compiled.n_exog)
    sol = klein_solve(cf, par, ss, compiled.incidence, n_state, n_exog=compiled.n_exog)
    f_xx = bicomplex_hessian(cf_bc.address, ss, par, compiled.n_exog, n_eq)
    return (
        a,
        b,
        f_xx,
        np.real(sol.f),
        np.real(sol.p),
        np.real(sol.B),
        DSGESolver._build_Q(compiled),
        n_state,
    )


@pytest.mark.parametrize(
    "path",
    [
        "tests/fixtures/models/rbc_second_order.yaml",  # n=3, nx=2, ny=1
        "MODELS/test.yaml",  # n=6, nx=3, ny=3
        "MODELS/POST82.yaml",  # n=5, nx=3, ny=2
    ],
)
def test_native_second_order_matches_numpy(path):
    args = _drive(path)

    for name, native, ref in zip(
        _BLOCKS, second_order(*args), _solve_second_order_numpy(*args)
    ):
        np.testing.assert_allclose(native, ref, rtol=0.0, atol=1e-12, err_msg=name)
