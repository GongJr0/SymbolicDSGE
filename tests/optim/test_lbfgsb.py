"""Parity + behavior tests for the native L-BFGS-B driver (issue #329).

The driver runs the vendored scipy L-BFGS-B kernel over a self-contained BLAS
backend; these assert result-parity with scipy on standard benchmark objectives
and pin the driver's deterministic, bounds-respecting behavior. Backend is the
module default (shim), so the suite exercises the shipped path without naming it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from SymbolicDSGE._ckernels.optim._optim import run_lbfgsb


def _scipy_rosen(x0, bounds=None):
    def f(x):
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    return minimize(f, x0, method="L-BFGS-B", bounds=bounds)


def _scipy_quad(x0, d, xs, bounds):
    def f(x):
        return 0.5 * np.sum(d * (x - xs) ** 2)

    return minimize(f, x0, method="L-BFGS-B", bounds=bounds)


def test_rosenbrock_matches_scipy():
    x0 = np.full(5, -1.2, dtype=np.float64)
    r = run_lbfgsb("rosenbrock", x0)
    rs = _scipy_rosen(x0)

    assert r["success"]
    assert r["nfev"] > 0 and r["nit"] > 0
    # Both reach the global minimum at the all-ones vector.
    np.testing.assert_allclose(r["x"], np.ones(5), atol=1e-3)
    assert r["fun"] < 1e-6
    np.testing.assert_allclose(r["x"], rs.x, atol=1e-3)


def test_bounded_quadratic_matches_scipy():
    n = 4
    d = np.array([1.0, 2.0, 4.0, 8.0])  # well-conditioned
    xs = np.array([2.0, -2.0, 0.5, -0.5])
    bounds = [(0.0, 1.0), (-1.0, 0.0), (None, None), (None, None)]
    params = np.concatenate([d, xs])
    x0 = np.zeros(n)

    r = run_lbfgsb("quad", x0, bounds=bounds, params=params)
    rs = _scipy_quad(x0, d, xs, bounds)

    assert r["success"]
    # Active bounds clip coords 0,1 to their nearest bound; 2,3 hit xstar.
    np.testing.assert_allclose(r["x"], [1.0, -1.0, 0.5, -0.5], atol=1e-6)
    np.testing.assert_allclose(r["x"], rs.x, atol=1e-6)
    np.testing.assert_allclose(r["fun"], rs.fun, rtol=1e-8, atol=1e-8)


def test_bounds_are_respected():
    n = 3
    d = np.array([1.0, 1.0, 1.0])
    xs = np.array([5.0, -5.0, 0.0])  # pulls outside the box
    bounds = [(-1.0, 1.0)] * n
    params = np.concatenate([d, xs])
    r = run_lbfgsb("quad", np.zeros(n), bounds=bounds, params=params)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    assert np.all(r["x"] >= lo - 1e-12)
    assert np.all(r["x"] <= hi + 1e-12)


def test_bimodal_deterministic_basin_selection():
    # Separable double well: minima at +/-1 per coord, saddle at 0. An off-center
    # seed must resolve to the near basin, deterministically and reproducibly.
    n = 3
    up1 = run_lbfgsb("double_well", np.full(n, 0.01))
    up2 = run_lbfgsb("double_well", np.full(n, 0.01))
    dn = run_lbfgsb("double_well", np.full(n, -0.01))

    assert np.all(up1["x"] > 0.9)
    assert np.all(dn["x"] < -0.9)
    # Determinism: identical inputs give byte-identical results.
    np.testing.assert_array_equal(up1["x"], up2["x"])


def test_result_reports_finite_fun_and_counts():
    r = run_lbfgsb("rosenbrock", np.zeros(4))
    assert np.isfinite(r["fun"])
    assert isinstance(r["message"], str) and r["message"]
    assert r["nfev"] >= r["nit"] > 0


def test_unknown_objective_raises():
    with pytest.raises(ValueError):
        run_lbfgsb("nope", np.zeros(3))
