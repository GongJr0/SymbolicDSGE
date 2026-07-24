"""Benchmark end-to-end point estimation: the native pipeline vs the legacy
Python/scipy pipeline it replaces (issue #330).

`Estimator.mle` / `Estimator.map` now hand a single struct-shaped container to
Cython and run the whole theta -> params -> solve -> filter -> loglik loop plus
the optimizer in native code, never returning to Python between evaluations. The
pipeline it replaced drove `scipy.optimize.minimize` over a Python objective that
re-ran `solver.solve(...).kalman(...)` on every evaluation. This bench times both
to convergence on the same problem and reports the end-to-end speedup.

The legacy side is reconstructed self-contained here rather than called on the
shipped `Estimator`: the native cleanup evicts the Python evaluators
(`_safe_loglik`, `loglik`, `backend.evaluate_loglik`, ...) from the estimation
call chain, so a bench that leaned on them would rot. The vendored objective
depends only on stable public surface: `solver.solve(parameters=...)`, the Kalman
filter interface (`solved.kalman(...).loglik`), and `Prior.logpdf`. It is exact
for the scalar identity-transform parameters used here; it is not a general
theta<->param scatter (matrix/CPC blocks and non-identity transforms are out of
scope for this bench).

Fixtures:
  * linear / extended / map -> POST82.yaml (gap model, first order, ss = 0).
  * unscented               -> rbc_second_order.yaml (levels, order-2 solve; the
    gap model's augmented UKF state is degenerate, so POST82 can't exercise a
    real UKF path -- same reason the per-eval objective bench uses RBC here).

The sample length T is swept on a log-grid. Both pipelines pay the same O(T)
filter and O(n^3) solve per evaluation, so the native win is the per-eval
dispatch and the in-C optimizer loop, a roughly T-independent overhead: the
speedup is largest at small T (loop-dominated) and compresses as T grows and the
filter swamps everything, the same story as the per-eval objective bench.

Parity is gated: both pipelines must land on the same objective value (same
basin) at each grid point, else the point is flagged and the run exits non-zero,
so a silent divergence can't flatter the numbers. The two optimizers stop on
their own native defaults (native L-BFGS-B factr/pgtol vs scipy ftol/gtol), so
`x` and `fun` agree only to the looser of the two tolerances, not bit-for-bit.

Usage:
    uv run python scripts/bench_point_estimation.py
    uv run python scripts/bench_point_estimation.py --cases mle-linear map-linear
    uv run python scripts/bench_point_estimation.py --t-max 8000 --steps 5
    uv run python scripts/bench_point_estimation.py --method Nelder-Mead

Developer benchmark, not shipped package code; the end-to-end correctness lives
in tests/estimation/test_estimation_oracle.py.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "models"
POST82 = FIXTURES / "POST82.yaml"
RBC = FIXTURES / "rbc_second_order.yaml"

from scipy.optimize import minimize  # noqa: E402

from SymbolicDSGE import DSGESolver, ModelParser  # noqa: E402
from SymbolicDSGE.estimation import Estimator  # noqa: E402
from SymbolicDSGE.kalman.config import KalmanConfig  # noqa: E402

# (routine, fixture) -> (estimated params, theta0, bounds). Identity-transform
# scalar calibration parameters, so theta == param value (the vendored legacy
# objective's scope).
_PARAMS = {
    ("mle", "post82"): (["psi_pi", "rho_r"], [2.0, 0.8], [(1.0, 5.0), (0.0, 0.99)]),
    ("map", "post82"): (["psi_pi"], [2.0], [(1.0, 5.0)]),
    ("mle", "rbc"): (["rho"], [0.95], [(0.0, 0.999)]),
}

# label -> (fixture, filter mode, routine)
_SPECS = {
    "mle-linear": ("post82", "linear", "mle"),
    "mle-extended": ("post82", "extended", "mle"),
    "mle-unscented": ("rbc", "unscented", "mle"),
    "map-linear": ("post82", "linear", "map"),
}


# --------------------------------------------------------------------------- #
# Fixtures: compiled model + sim closure + solve/filter settings, built once.  #
# --------------------------------------------------------------------------- #
@dataclass
class Fixture:
    name: str
    solver: object
    compiled: object
    base_calib: dict[str, float]  # name -> value, full calibration
    ss_seed: np.ndarray
    obs: list[str]
    solve_kwargs: dict  # extra kwargs to solver.solve (e.g. order=2)
    jitter: float | None
    symmetrize: bool | None
    P0: np.ndarray | None
    _sim: Callable[[int], pd.DataFrame]
    y_cache: dict[int, pd.DataFrame] = field(default_factory=dict)

    def y(self, T: int) -> pd.DataFrame:
        if T not in self.y_cache:
            self.y_cache[T] = self._sim(T)
        return self.y_cache[T]

    def filter_kwargs(self) -> dict:
        kw: dict[str, Any] = {}
        if self.jitter is not None:
            kw["jitter"] = self.jitter
        if self.symmetrize is not None:
            kw["symmetrize"] = self.symmetrize
        return kw


def _post82_fixture() -> Fixture:
    from sympy import Symbol

    model, kalman = ModelParser(POST82).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    steady = np.zeros((len(compiled.var_names),), dtype=np.float64)
    solved = solver.solve(compiled=compiled, ss_seed=steady)

    params = model.calibration.parameters
    std_map = model.calibration.shock_std
    sig = {s: float(params[std_map[Symbol(s)]]) for s in ("e_g", "e_z", "e_r")}
    base_calib = {sym.name: float(val) for sym, val in params.items()}

    def _sim(T: int) -> pd.DataFrame:
        rng = np.random.default_rng(20260724)
        sim = solved.sim(
            T=T,
            shocks={
                "g": rng.normal(0.0, sig["e_g"], size=T),
                "z": rng.normal(0.0, sig["e_z"], size=T),
                "r": rng.normal(0.0, sig["e_r"], size=T),
            },
            x0=steady,
            observables=True,
        )
        return pd.DataFrame(
            {
                "OutGap": sim["OutGap"][1:],
                "Infl": sim["Infl"][1:],
                "Rate": sim["Rate"][1:],
            }
        )

    return Fixture(
        name="post82",
        solver=solver,
        compiled=compiled,
        base_calib=base_calib,
        ss_seed=steady,
        obs=["OutGap", "Infl", "Rate"],
        solve_kwargs={},
        jitter=None,
        symmetrize=None,
        P0=None,
        _sim=_sim,
    )


def _rbc_fixture() -> Fixture:
    model, _ = ModelParser(RBC).get_all()
    R = np.array([[1e-4]], dtype=np.float64)
    P0 = np.eye(3, dtype=np.float64) * 0.1  # z, k, c
    kalman = KalmanConfig(R=R, P0=P0)
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    seed = np.asarray(
        solver.solve(compiled=compiled, order=2).policy.steady_state, dtype=np.float64
    )
    solved = solver.solve(compiled=compiled, ss_seed=seed, order=2)
    base_calib = {
        sym.name: float(val) for sym, val in model.calibration.parameters.items()
    }

    def _sim(T: int) -> pd.DataFrame:
        rng = np.random.default_rng(20260724)
        sim = solved.sim(
            T=T,
            shocks={"z": rng.normal(0.0, 0.01, size=T)},
            x0=seed,
            observables=True,
        )
        return pd.DataFrame({"c_obs": sim["c_obs"][1:]})

    return Fixture(
        name="rbc",
        solver=solver,
        compiled=compiled,
        base_calib=base_calib,
        ss_seed=seed,
        obs=["c_obs"],
        solve_kwargs={"order": 2},
        jitter=1e-8,
        symmetrize=True,
        P0=P0,
        _sim=_sim,
    )


# --------------------------------------------------------------------------- #
# Legacy pipeline: scipy.optimize.minimize over a Python solve+filter objective.
# Self-contained on public API so it survives the Estimator native cleanup.    #
# --------------------------------------------------------------------------- #
def _legacy_objective(
    fx: Fixture,
    y: pd.DataFrame,
    mode: str,
    names: list[str],
    priors: dict[str, Any] | None,
) -> Callable[[np.ndarray], float]:
    """One negative-(log-likelihood|posterior) evaluation: scatter theta into the
    calibration, re-solve, run the filter. Mirrors the pre-native per-draw work."""
    fkw = fx.filter_kwargs()

    def obj(theta: np.ndarray) -> float:
        params = dict(fx.base_calib)
        for name, value in zip(names, np.asarray(theta, dtype=np.float64)):
            params[name] = float(value)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solved = fx.solver.solve(
                    compiled=fx.compiled,
                    parameters=params,
                    ss_seed=fx.ss_seed,
                    **fx.solve_kwargs,
                )
                ll = float(
                    solved.kalman(
                        y=y, filter_mode=mode, observables=fx.obs, **fkw
                    ).loglik
                )
        except BaseException:
            return float(np.inf)
        if priors is not None:
            for name, prior in priors.items():
                ll += float(prior.logpdf(np.float64(params[name])))
        return np.inf if not np.isfinite(ll) else -ll

    return obj


def _legacy_run(
    fx: Fixture,
    y: pd.DataFrame,
    mode: str,
    names: list[str],
    init: np.ndarray,
    bounds: list[tuple[float, float]],
    method: str,
    priors: dict[str, Any] | None,
) -> Any:
    obj = _legacy_objective(fx, y, mode, names, priors)
    # scipy's own FD gradient computes f_eval - f0 and warns (inf-inf -> nan)
    # whenever the objective returns +inf at a probed point; benign, silence it.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return minimize(obj, x0=init, method=method, bounds=bounds)


# --------------------------------------------------------------------------- #
# Native pipeline: Estimator.mle / .map.                                       #
# --------------------------------------------------------------------------- #
def _make_estimator(
    fx: Fixture,
    mode: str,
    y: pd.DataFrame,
    names: list[str],
    priors: dict[str, Any] | None,
) -> Estimator:
    return Estimator(
        solver=fx.solver,
        compiled=fx.compiled,
        y=y,
        observables=fx.obs,
        filter_mode=mode,
        estimated_params=names,
        priors=priors,
        ss_seed=fx.ss_seed,
        P0=fx.P0,
        jitter=fx.jitter,
        symmetrize=fx.symmetrize,
    )


@contextlib.contextmanager
def _quiet():
    """Swallow the per-call BK warning-count line and UserWarnings so the native
    runs don't scribble over the table."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------- #
# Case assembly                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    label: str
    T: int
    old: Callable[[], Any]
    new: Callable[[], Any]
    old_res: Any
    new_res: Any


def _build_case(fx: Fixture, label: str, T: int, method: str) -> Case:
    _, mode, routine = _SPECS[label]
    names_raw, init_raw, bounds = _PARAMS[(routine, fx.name)]
    names = list(names_raw)
    init = np.asarray(init_raw, dtype=np.float64)
    y = fx.y(T)

    if routine == "map":
        prior = Estimator.make_prior(
            distribution="normal",
            parameters={"mean": 2.0, "std": 0.5},
            transform="identity",
        )
        priors: dict[str, Any] | None = {names[0]: prior}
    else:
        priors = None

    est = _make_estimator(fx, mode, y, names, priors)

    def old() -> Any:
        return _legacy_run(fx, y, mode, names, init, list(bounds), method, priors)

    def new() -> Any:
        with _quiet():
            if routine == "map":
                return est.map(theta0=init, bounds=list(bounds), method=method)
            return est.mle(theta0=init, bounds=list(bounds), method=method)

    return Case(label, T, old, new, old(), new())


# --------------------------------------------------------------------------- #
# Grid + timing                                                               #
# --------------------------------------------------------------------------- #
def _log_grid(t_min: int, t_max: int, steps: int) -> list[int]:
    raw = np.logspace(np.log10(t_min), np.log10(t_max), steps)
    return sorted({int(round(v)) for v in raw})


def _time(
    fn: Callable[[], Any], max_reps: int, warmup: int, budget: float
) -> list[float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    start = time.perf_counter()
    for _ in range(max_reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
        if time.perf_counter() - start > budget:
            break
    return samples


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:11.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(_SPECS),
        choices=list(_SPECS),
        help="{mle,map}-{linear,extended,unscented} pairs to bench.",
    )
    parser.add_argument("--method", default="L-BFGS-B", help="Optimizer method.")
    parser.add_argument("--t-min", type=int, default=48, help="Smallest T.")
    parser.add_argument("--t-max", type=int, default=4000, help="Largest T.")
    parser.add_argument("--steps", type=int, default=4, help="Log-grid points.")
    parser.add_argument("--max-reps", type=int, default=25, help="Reps cap per point.")
    parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed calls per point."
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=4.0,
        help="Wall-clock seconds cap for each timed series (bounds large T).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for the old/new objective-value parity gate.",
    )
    args = parser.parse_args()

    grid = _log_grid(args.t_min, args.t_max, args.steps)
    fixtures: dict[str, Fixture] = {}
    needed = {_SPECS[label][0] for label in args.cases}
    if "post82" in needed:
        fixtures["post82"] = _post82_fixture()
    if "rbc" in needed:
        fixtures["rbc"] = _rbc_fixture()

    print(
        f"method={args.method}  T-grid={grid}\n"
        f"max_reps={args.max_reps} warmup={args.warmup} budget={args.budget}s "
        f"(wall time in ms/optimization; speedup = old/new)\n"
    )
    header = (
        f"{'case':<15} {'T':>7} {'old med':>11} {'new med':>11} {'speedup':>8} "
        f"{'old min':>11} {'nfev o/n':>13} {'dfun':>10} {'fun':>13}"
    )
    exit_code = 0
    for label in args.cases:
        fx = fixtures[_SPECS[label][0]]
        print(header)
        print("-" * len(header))
        for T in grid:
            case = _build_case(fx, label, T, args.method)
            dfun = abs(float(case.old_res.fun) - float(case.new_res.fun))
            if dfun > args.atol:
                print(
                    f"{label:<15} {T:>7}  PARITY FAIL old={case.old_res.fun!r} "
                    f"new={case.new_res.fun!r} (d={dfun:.3e})",
                    file=sys.stderr,
                )
                exit_code = 1
                continue
            old_t = _time(case.old, args.max_reps, args.warmup, args.budget)
            new_t = _time(case.new, args.max_reps, args.warmup, args.budget)
            old_med, new_med = statistics.median(old_t), statistics.median(new_t)
            speedup = old_med / new_med if new_med > 0 else float("nan")
            nfev = f"{int(case.old_res.nfev)}/{int(case.new_res.nfev)}"
            print(
                f"{label:<15} {T:>7} {_ms(old_med)} {_ms(new_med)} {speedup:7.2f}x "
                f"{_ms(min(old_t))} {nfev:>13} {dfun:10.2e} "
                f"{float(case.new_res.fun):13.4f}"
            )
        print()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
