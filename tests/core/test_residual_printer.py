"""Tests for the c128 residual printer (issue #248).

Three layers:
  1. Arithmetic breadth -- hand-built expressions covering every op path, value-
     checked against sympy's own lambdify at random (complex) points.
  2. Derivative correctness -- the reason integer powers use repeated multiply
     rather than ``**``: the complex-step derivative must be right for a *negative*
     base.
  3. Real-model parity -- the emitted residual matches the existing reference both
     in value and in the Klein complex-step linearization (a, b), and the cfunc
     form compiles to a callable address.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.klein import _approximate_system_numeric
from SymbolicDSGE.core.residual_printer import (
    ResidualLayout,
    build_cfunc,
    build_njit,
)

C = np.complex128


def _run_expr(expr: sp.Expr, order: list[sp.Symbol]):
    slot = {s: ("cur", i) for i, s in enumerate(order)}
    layout = ResidualLayout(slot=slot, n_var=len(order), n_par=0, n_eq=1)
    fn = build_njit([expr], layout)
    ref = sp.lambdify(order, expr, "numpy")

    def run(values: tuple[complex, ...]) -> tuple[complex, complex]:
        cur = np.array(values, dtype=C)
        out = fn(np.zeros(len(order), C), cur, np.zeros(0, C))
        return complex(out[0]), complex(ref(*values))

    return run


def test_expr_value_parity():
    x, y = sp.symbols("x y")
    exprs = [
        x + y,
        x - y,
        -x,
        x * y,
        x / y,
        2 * x + 3 * y,
        x - 2 * y,
        sp.Rational(3, 2) * x * y,
        x**2,
        x**3,
        x**5,
        x**-1,
        x**-2,
        sp.sqrt(x),
        x ** sp.Rational(5, 2),
        x ** sp.Rational(-1, 2),
        sp.exp(x),
        sp.log(x),
        sp.exp(x) * y - x**2,
        x**y,  # symbolic exponent -> cpow
        (x + y) ** 2,  # compound base -> bound to a temp
        1 / (1 + x**2),
    ]
    rng = np.random.default_rng(0)
    for expr in exprs:
        run = _run_expr(expr, [x, y])
        for _ in range(15):
            # Positive-real-dominant keeps log/sqrt/spow on the principal branch.
            vals = (
                rng.uniform(0.5, 2.0) + 1j * rng.uniform(-0.3, 0.3),
                rng.uniform(0.5, 2.0) + 1j * rng.uniform(-0.3, 0.3),
            )
            got, want = run(vals)
            np.testing.assert_allclose(
                got, want, rtol=1e-9, atol=1e-11, err_msg=str(expr)
            )


@pytest.mark.parametrize(
    "expr_factory,analytic",
    [
        (lambda x: x**2, lambda v: 2 * v),
        (lambda x: x**3, lambda v: 3 * v**2),
        (lambda x: 1 / x, lambda v: -1 / v**2),
    ],
)
def test_ipow_complex_step_correct_for_negative_base(expr_factory, analytic):
    # Repeated-multiply integer powers give the correct complex-step derivative
    # even where the base is negative -- the branch cut that `**` would hit.
    x = sp.Symbol("x")
    layout = ResidualLayout(slot={x: ("cur", 0)}, n_var=1, n_par=0, n_eq=1)
    fn = build_njit([expr_factory(x)], layout)
    v0, h = -0.7, 1e-100
    out = fn(np.zeros(1, C), np.array([v0 + 1j * h], dtype=C), np.zeros(0, C))
    assert out[0].imag / h == pytest.approx(analytic(v0), rel=1e-10)


def _compiled(path: str):
    model, kalman = ModelParser(path).get_all()
    return DSGESolver(model, kalman).compile()


def _param_vector(compiled, dtype):
    return np.array(
        [
            float(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=dtype,
    )


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_printer_matches_reference_residual_values(path):
    compiled = _compiled(path)
    layout = ResidualLayout.from_compiled(compiled)
    fn = build_njit(compiled.objective_eqs, layout)
    par = _param_vector(compiled, C)

    rng = np.random.default_rng(1)
    n = layout.n_var
    for _ in range(10):
        fwd = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(C)
        cur = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(C)
        got = fn(fwd, cur, par)
        want = compiled.equations(fwd, cur, par)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("path", ["MODELS/test.yaml", "MODELS/POST82.yaml"])
def test_printer_linearization_matches_reference(path):
    compiled = _compiled(path)
    layout = ResidualLayout.from_compiled(compiled)
    fn = build_njit(compiled.objective_eqs, layout)
    ref = compiled.construct_objective_vector_func()

    ss = np.zeros(layout.n_var, dtype=np.float64)
    par = _param_vector(compiled, np.float64)

    a_ref, b_ref = _approximate_system_numeric(ref, ss, par, False)
    a_new, b_new = _approximate_system_numeric(fn, ss, par, False)

    np.testing.assert_allclose(a_new, a_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(b_new, b_ref, rtol=1e-10, atol=1e-12)


def test_build_cfunc_compiles_to_address():
    compiled = _compiled("MODELS/test.yaml")
    layout = ResidualLayout.from_compiled(compiled)
    cf = build_cfunc(compiled.objective_eqs, layout)
    assert isinstance(cf.address, int) and cf.address != 0
