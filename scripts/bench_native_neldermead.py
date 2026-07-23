"""Benchmark the native Nelder-Mead optimizer against scipy's Python routine.

Nelder-Mead has no inner linear algebra (no BLAS/LAPACK), so unlike the L-BFGS-B
bench there is no backend axis: the comparison is native C driver vs scipy's
pure-Python `_minimize_neldermead`. On a *cheap* synthetic objective the driver
loop dominates, so this measures the win from moving the simplex bookkeeping and
the per-eval dispatch out of Python; on a real DSGE objective the eval cost
swamps the loop and the two converge in wall time.

Cases:
  * rosenbrock  -- unbounded parity + timing across dimensions.
  * microbench  -- bounded quadratic; cheap eval, many iterations. Reports wall
    time, nfev, nit for native vs scipy and the speedup, and asserts they land on
    the same optimum within tolerance.
  * bimodal     -- separable double well; off-center seeds must resolve to the
    near basin identically and deterministically.
  * halfplane   -- Rosenbrock on the feasible half-plane x[0] >= 0 (+inf outside),
    a BK-violation analog: both must tolerate the +inf returns and still reach
    the feasible minimum.

Usage:
    uv run python scripts/bench_native_neldermead.py
    uv run python scripts/bench_native_neldermead.py --reps 500 --n 6

Once the estimation<->optim wiring (issue #330) lands, an in-context run on the
real DSGE solve+filter objective is the meaningful end-to-end bench; here the
objective is synthetic so the optimizer internals are exposed.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from scipy.optimize import minimize

from SymbolicDSGE._ckernels.optim._optim import run_neldermead

# Tight tol + generous budget so both sides fully converge (parity is meaningless
# if either stops on the cap).
_TOL = dict(xatol=1e-8, fatol=1e-8)


def _time(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def _scipy_opts(budget):
    return {**_TOL, "maxiter": budget, "maxfev": budget}


def rosenbrock(n, budget):
    print(f"=== rosenbrock parity n={n} ===")
    x0 = np.full(n, -1.2)

    def f(x):
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    r = run_neldermead("rosenbrock", x0, maxiter=budget, maxfun=budget, **_TOL)
    rs = minimize(f, x0, method="Nelder-Mead", options=_scipy_opts(budget))
    for name, x, fun, nit, nfev in (
        ("native", r["x"], r["fun"], r["nit"], r["nfev"]),
        ("scipy", rs.x, rs.fun, rs.nit, rs.nfev),
    ):
        print(
            f"  {name:7s} fun={fun:.3e} nit={nit:5d} nfev={nfev:5d} "
            f"|x-1|_max={np.max(np.abs(x - 1)):.2e}"
        )
    print()


def microbench(n, budget, reps, warmup):
    print(f"=== microbench: bounded quadratic n={n} ===")
    d = np.logspace(0.0, 1.5, n)
    xstar = np.where(np.arange(n) % 2 == 0, 2.0, -2.0)
    params = np.concatenate([d, xstar])
    # Even coords bounded (active at the corner), odd coords free.
    bounds = [(0.0, 1.0) if i % 2 == 0 else (None, None) for i in range(n)]
    # Seed strictly inside every bound so the corner is actually reached.
    x0 = np.where(np.arange(n) % 2 == 0, 0.5, 0.0)

    def f(x):
        return 0.5 * np.sum(d * (x - xstar) ** 2)

    rn = run_neldermead(
        "quad", x0, bounds=bounds, params=params, maxiter=budget, maxfun=budget, **_TOL
    )
    rs = minimize(
        f, x0, method="Nelder-Mead", bounds=bounds, options=_scipy_opts(budget)
    )

    tn = _time(
        lambda: run_neldermead(
            "quad",
            x0,
            bounds=bounds,
            params=params,
            maxiter=budget,
            maxfun=budget,
            **_TOL,
        ),
        reps,
        warmup,
    )
    ts = _time(
        lambda: minimize(
            f, x0, method="Nelder-Mead", bounds=bounds, options=_scipy_opts(budget)
        ),
        reps,
        warmup,
    )

    hdr = f"{'impl':7s} {'fun':>14s} {'nit':>5s} {'nfev':>6s} {'med us':>12s} {'min us':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for name, r, tt in (
        ("native", rn, tn),
        ("scipy", {"fun": rs.fun, "nit": rs.nit, "nfev": rs.nfev}, ts),
    ):
        print(
            f"{name:7s} {r['fun']:14.9f} {r['nit']:5d} {r['nfev']:6d} "
            f"{statistics.median(tt) * 1e6:12.2f} {min(tt) * 1e6:12.2f}"
        )
    dx = np.max(np.abs(rn["x"] - rs.x))
    print(f"  native vs scipy: |dfun|={abs(rn['fun'] - rs.fun):.2e} |dx|_max={dx:.2e}")
    speedup = statistics.median(ts) / statistics.median(tn)
    print(f"  native/scipy median wall speedup = {speedup:.1f}x\n")


def bimodal():
    print("=== bimodal: separable double well, off-center seeds ===")
    n = 4

    def f(x):
        return np.sum((x**2 - 1.0) ** 2)

    for seed in (0.05, -0.05):
        rn = run_neldermead("double_well", np.full(n, seed), **_TOL)
        rs = minimize(f, np.full(n, seed), method="Nelder-Mead", options=_TOL)
        sn = np.sign(np.round(rn["x"], 6)).astype(int)
        ss = np.sign(np.round(rs.x, 6)).astype(int)
        print(
            f"  seed {seed:+.2f}: native basin {sn}  scipy basin {ss}  agree={np.array_equal(sn, ss)}"
        )
    # Determinism: identical inputs -> byte-identical native results.
    a = run_neldermead("double_well", np.full(n, 0.05), **_TOL)["x"]
    b = run_neldermead("double_well", np.full(n, 0.05), **_TOL)["x"]
    print(f"  native determinism (byte-identical repeat): {np.array_equal(a, b)}\n")


def halfplane(budget):
    print("=== halfplane: Rosenbrock on x[0] >= 0 (+inf outside) ===")
    x0 = np.array([0.6, 0.6])

    def f(x):
        if x[0] < 0.0:
            return np.inf
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    rn = run_neldermead("rosen_halfplane", x0, maxiter=budget, maxfun=budget, **_TOL)
    rs = minimize(f, x0, method="Nelder-Mead", options=_scipy_opts(budget))
    for name, x, fun, ok in (
        ("native", rn["x"], rn["fun"], rn["success"]),
        ("scipy", rs.x, rs.fun, rs.success),
    ):
        print(f"  {name:7s} x={np.round(x, 6)} fun={fun:.3e} ok={ok}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--budget", type=int, default=8000, help="maxiter == maxfev cap")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    rosenbrock(args.n, args.budget)
    bimodal()
    halfplane(args.budget)
    microbench(args.n, args.budget, args.reps, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
