from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import sympy as sp

from SymbolicDSGE._symbolic_printers import (
    BicomplexOps,
    MeasurementLayout,
    ResidualLayout,
    build_cfunc,
    build_measurement_cfunc,
    build_njit,
)

C = np.complex128


def test_residual_printer_build_njit_matches_sympy_values() -> None:
    cur_x, cur_y, beta = sp.symbols("cur_x cur_y beta")
    layout = ResidualLayout(
        slot={cur_x: ("cur", 0), cur_y: ("cur", 1), beta: ("par", 0)},
        n_var=2,
        n_par=1,
        n_eq=2,
    )
    exprs = [
        sp.exp(cur_x) + beta * cur_y**2 - 1 / (1 + cur_x),
        sp.log(cur_y) + sp.sqrt(cur_x),
    ]

    fn = build_njit(exprs, layout)
    cur = np.array([1.2 + 0.1j, 0.7 - 0.05j], dtype=C)
    par = np.array([0.8 + 0.0j], dtype=C)
    got = fn(np.zeros(2, dtype=C), cur, par)

    ref = sp.lambdify((cur_x, cur_y, beta), exprs, "numpy")
    want = np.asarray(ref(cur[0], cur[1], par[0]), dtype=C)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_residual_printer_build_cfunc_compiles_bicomplex() -> None:
    fwd_x, cur_x, beta = sp.symbols("fwd_x cur_x beta")
    layout = ResidualLayout(
        slot={
            fwd_x: ("fwd", 0),
            cur_x: ("cur", 0),
            beta: ("par", 0),
        },
        n_var=1,
        n_par=1,
        n_eq=1,
    )

    cf = build_cfunc([beta * sp.exp(fwd_x) + cur_x**2], layout, BicomplexOps())

    assert isinstance(cf.address, int) and cf.address != 0


def test_measurement_layout_normalizes_observables() -> None:
    alpha = sp.Symbol("alpha")
    cur_x, cur_y = sp.symbols("cur_x cur_y")
    compiled = SimpleNamespace(
        var_names=["x", "y"],
        calib_params=[alpha],
        observable_names=["obs_x", "obs_y"],
        observable_eqs=[cur_x + alpha, cur_y * alpha],
    )

    layout = MeasurementLayout.from_compiled(compiled, ["obs_y", "obs_x"])

    assert layout.observable_indices == (0, 1)
    assert layout.n_expr == 2
    assert layout.slot[cur_x] == ("vars", 0)
    assert layout.slot[cur_y] == ("vars", 1)
    assert layout.slot[alpha] == ("par", 0)


def test_measurement_cfunc_writes_outputs() -> None:
    beta = sp.Symbol("beta")
    cur_x, cur_y = sp.symbols("cur_x cur_y")
    layout = MeasurementLayout(
        slot={cur_x: ("vars", 0), cur_y: ("vars", 1), beta: ("par", 0)},
        n_var=2,
        n_par=1,
        n_obs=2,
    )
    exprs = [
        cur_x + beta * cur_y,
        sp.exp(cur_y) + sp.sqrt(cur_x**2),
    ]

    cf = build_measurement_cfunc(exprs, layout)
    state = np.array([1.25, 0.4], dtype=np.float64)
    params = np.array([0.75], dtype=np.float64)
    out = np.empty(2, dtype=np.float64)

    ptr = ctypes.POINTER(ctypes.c_double)
    cf.ctypes(
        state.ctypes.data_as(ptr),
        params.ctypes.data_as(ptr),
        out.ctypes.data_as(ptr),
    )

    expected = np.array(
        [
            state[0] + params[0] * state[1],
            np.exp(state[1]) + np.sqrt(state[0] ** 2),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
