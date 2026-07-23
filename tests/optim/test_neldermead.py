"""Parity + behavior tests for the native Nelder-Mead driver (issue #335).

The driver is a faithful transpilation of scipy's `_minimize_neldermead`; these
assert result-parity with scipy Nelder-Mead on standard benchmark objectives
(Rosenbrock, bounded quadratic) and pin the driver's contracted behavior: bound
clipping, +inf (BK-violation) tolerance, deterministic basin selection, and the
maxfun cap. Parity is tolerance-based on x/fun, never on the trajectory: scipy
re-sorts the simplex with a non-stable argsort, so tie ordering is not
reproducible across implementations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from SymbolicDSGE._ckernels.optim._optim import run_neldermead

# Generous shared budget so both sides fully converge on the slow (gradient-free)
# Rosenbrock descent; parity is meaningless if either stops on the cap.
_TOL = dict(xatol=1e-8, fatol=1e-8)
_BUDGET = dict(maxiter=8000, maxfun=8000)


def _scipy_nm(f, x0, bounds=None):
    return minimize(
        f,
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={**_TOL, "maxiter": _BUDGET["maxiter"], "maxfev": _BUDGET["maxfun"]},
    )


def _rosen(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def test_rosenbrock_matches_scipy():
    x0 = np.full(3, -1.2, dtype=np.float64)
    r = run_neldermead("rosenbrock", x0, **_TOL, **_BUDGET)
    rs = _scipy_nm(_rosen, x0)

    assert r["success"]
    assert r["nfev"] > 0 and r["nit"] > 0
    # Both reach the global minimum at the all-ones vector.
    np.testing.assert_allclose(r["x"], np.ones(3), atol=1e-4)
    assert r["fun"] < 1e-8
    np.testing.assert_allclose(r["x"], rs.x, atol=1e-4)


def test_bounded_quadratic_matches_scipy():
    n = 4
    d = np.array([1.0, 2.0, 4.0, 8.0])  # well-conditioned
    xs = np.array([2.0, -2.0, 0.5, -0.5])
    bounds = [(0.0, 1.0), (-1.0, 0.0), (None, None), (None, None)]
    params = np.concatenate([d, xs])
    # Interior seed: a start on a bound makes the zdelt-scale simplex stall there
    # (scipy does the same), so seed strictly inside the box to reach the corner.
    x0 = np.array([0.5, -0.5, 0.0, 0.0])

    def f(x):
        return 0.5 * np.sum(d * (x - xs) ** 2)

    r = run_neldermead("quad", x0, bounds=bounds, params=params, **_TOL, **_BUDGET)
    rs = _scipy_nm(f, x0, bounds=bounds)

    assert r["success"]
    # Active bounds clip coords 0,1 to their nearest bound; 2,3 hit xstar.
    np.testing.assert_allclose(r["x"], [1.0, -1.0, 0.5, -0.5], atol=1e-4)
    np.testing.assert_allclose(r["x"], rs.x, atol=1e-4)
    np.testing.assert_allclose(r["fun"], rs.fun, rtol=1e-6, atol=1e-8)


def test_bounds_are_respected():
    n = 3
    d = np.array([1.0, 1.0, 1.0])
    xs = np.array([5.0, -5.0, 0.0])  # pulls outside the box
    bounds = [(-1.0, 1.0)] * n
    params = np.concatenate([d, xs])
    r = run_neldermead("quad", np.zeros(n), bounds=bounds, params=params, **_TOL)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    assert np.all(r["x"] >= lo - 1e-12)
    assert np.all(r["x"] <= hi + 1e-12)


def test_plus_inf_returns_do_not_derail_search():
    # rosen_halfplane returns +inf for x[0] < 0 (a BK-violation analog). Starting
    # feasible, the search probes into the infeasible region but must still reach
    # the feasible minimum at all-ones.
    x0 = np.array([0.6, 0.6], dtype=np.float64)
    r = run_neldermead("rosen_halfplane", x0, **_TOL, **_BUDGET)
    assert r["success"]
    np.testing.assert_allclose(r["x"], np.ones(2), atol=1e-4)
    assert np.isfinite(r["fun"]) and r["fun"] < 1e-8


def test_all_infeasible_start_reports_failure_not_false_success():
    # Every initial-simplex vertex is +inf: NM has no signal to escape, and the
    # convergence delta is inf-inf == NaN. The NaN-propagating max must fail the
    # convergence test (matching scipy's np.max), so the driver reports
    # success=False rather than "converging" at the garbage start.
    x0 = np.array([-5.0, 1.0], dtype=np.float64)
    r = run_neldermead("rosen_halfplane", x0, **_TOL)
    assert not r["success"]
    assert not np.isfinite(r["fun"])


def test_bimodal_deterministic_basin_selection():
    # Separable double well: minima at +/-1 per coord, saddle at 0. An off-center
    # seed must resolve to the near basin, deterministically and reproducibly.
    n = 3
    up1 = run_neldermead("double_well", np.full(n, 0.05), **_TOL)
    up2 = run_neldermead("double_well", np.full(n, 0.05), **_TOL)
    dn = run_neldermead("double_well", np.full(n, -0.05), **_TOL)

    assert np.all(up1["x"] > 0.9)
    assert np.all(dn["x"] < -0.9)
    # Determinism: identical inputs give byte-identical results.
    np.testing.assert_array_equal(up1["x"], up2["x"])


def test_maxfun_cap_stops_and_flags():
    # A tight eval cap must stop the search, report failure with the maxfev
    # message, and never exceed the cap (per-eval guard, not a loop-top check).
    cap = 25
    r = run_neldermead("rosenbrock", np.full(4, -1.2), maxfun=cap, maxiter=100000)
    assert not r["success"]
    assert r["status"] == 1
    assert r["nfev"] <= cap
    assert "function evaluations" in r["message"]


def test_maxiter_cap_stops_and_flags():
    cap = 3
    r = run_neldermead("rosenbrock", np.full(4, -1.2), maxiter=cap, maxfun=100000)
    assert not r["success"]
    assert r["status"] == 2
    assert r["nit"] <= cap
    assert "iterations" in r["message"]


def test_result_reports_finite_fun_and_counts():
    r = run_neldermead("rosenbrock", np.zeros(4), **_TOL)
    assert np.isfinite(r["fun"])
    assert isinstance(r["message"], str) and r["message"]
    assert r["nfev"] >= r["nit"] > 0


def test_unknown_objective_raises():
    with pytest.raises(ValueError):
        run_neldermead("nope", np.zeros(3))
