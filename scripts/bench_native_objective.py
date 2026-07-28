"""Benchmark the native estimation objective against the Python solve+filter path.

Each case prepares an ``Estimator`` with one calibrated parameter in theta,
then constructs a persistent native ``NativeLogpost`` context. The benchmark
therefore measures the repeated objective evaluation used by native estimation:
solve -> filter -> loglik, with cfunc addresses, packed inputs, and filter
scratch retained across evaluations. The Python baseline is the corresponding
``Estimator.loglik(theta)`` call.

The sample length T is swept on a log-grid (default 100 .. 1e6, 10 whole-number
steps). The filter is O(T) while the solve is O(n^3) and T-independent, so the
grid separates the fixed solve overhead from the per-observation filter cost; the
T=1e6 end approximates the asymptotic-posterior regime where an estimator is run
on a very long (or pseudo-asymptotic) sample. Reps are budget-limited so the
large-T points stay bounded in wall time.

Fixtures:
  * linear / extended  -> tests/fixtures/models/POST82.yaml (gap model, ss = 0)
  * unscented          -> tests/fixtures/models/rbc_second_order.yaml (levels,
    order-2 solve; the gap model is degenerate for the UKF's augmented state)

Usage:
    uv run python scripts/bench_native_objective.py
    uv run python scripts/bench_native_objective.py --modes linear unscented
    uv run python scripts/bench_native_objective.py --t-max 100000 --steps 6
    uv run python scripts/bench_native_objective.py --budget 5 --max-reps 500

This is a developer benchmark, not shipped package code and not a correctness
test (that lives in tests/estimation/test_native_objective.py); it does assert
native/Python parity at each grid point so a silent divergence can't flatter the
numbers.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "models"
POST82 = FIXTURES / "POST82.yaml"
RBC = FIXTURES / "rbc_second_order.yaml"

from SymbolicDSGE import DSGESolver, ModelParser  # noqa: E402
from SymbolicDSGE.estimation import Estimator, backend  # noqa: E402
from SymbolicDSGE.kalman.config import KalmanConfig  # noqa: E402
from SymbolicDSGE.kalman.interface import FilterMode, _resolve_P0  # noqa: E402
from SymbolicDSGE._ckernels.estimation._estimation import NativeLogpost  # noqa: E402

_cc = np.ascontiguousarray
_POST82_MODES = ("linear", "extended")


@dataclass
class Case:
    """One (mode, T) pair's native / Python objective closures + parity values."""

    mode: str
    T: int
    native: Callable[[], float]  # full solve+filter+loglik in C
    python: Callable[[], float]  # equivalent Python solve+filter+loglik
    ll_native: float
    ll_python: float


# --------------------------------------------------------------------------- #
# Fixture contexts: compiled model + T-independent inputs, built once.        #
# --------------------------------------------------------------------------- #
@dataclass
class Post82Ctx:
    solver: object
    compiled: object
    kalman: object
    steady: np.ndarray
    obs: list[str]
    residual_addr: int
    calib: np.ndarray
    Q: np.ndarray
    solved: object
    sig: dict
    y_cache: dict[int, pd.DataFrame] = field(default_factory=dict)


def _post82_context() -> Post82Ctx:
    from sympy import Symbol

    model, kalman = ModelParser(POST82).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    steady = np.zeros((len(compiled.var_names),), dtype=np.float64)
    solved = solver.solve(compiled=compiled, ss_seed=steady)

    params = model.calibration.parameters
    std_map = model.calibration.shock_std
    sig = {s: float(params[std_map[Symbol(s)]]) for s in ("e_g", "e_z", "e_r")}

    base = backend.extract_base_params(compiled)
    return Post82Ctx(
        solver=solver,
        compiled=compiled,
        kalman=kalman,
        steady=steady,
        obs=["Infl", "Rate"],
        residual_addr=compiled.construct_objective_cfunc().address,
        calib=_cc(backend.build_calib_param_vector(compiled, base), dtype=np.float64),
        Q=_cc(backend.build_Q(compiled, base), dtype=np.float64),
        solved=solved,
        sig=sig,
    )


def _post82_y(ctx: Post82Ctx, T: int) -> pd.DataFrame:
    """Simulated observables of length T (cached; linear and extended share it)."""
    if T not in ctx.y_cache:
        rng = np.random.default_rng(20260303)
        sim = ctx.solved.sim(
            T=T,
            shocks={
                "g": rng.normal(0.0, ctx.sig["e_g"], size=T),
                "z": rng.normal(0.0, ctx.sig["e_z"], size=T),
                "r": rng.normal(0.0, ctx.sig["e_r"], size=T),
            },
            x0=ctx.steady,
            observables=True,
        )
        ctx.y_cache[T] = pd.DataFrame(
            {
                "OutGap": sim["OutGap"],
                "Infl": sim["Infl"],
                "Rate": sim["Rate"],
            }
        )
    return ctx.y_cache[T]


def _post82_case(ctx: Post82Ctx, mode: str, T: int) -> Case:
    y = _post82_y(ctx, T)
    estimator = Estimator(
        solver=ctx.solver,
        compiled=ctx.compiled,
        y=y,
        observables=ctx.obs,
        filter_mode=mode,
        estimated_params=["beta"],
        ss_seed=ctx.steady,
    )
    theta = estimator.resolve_theta0(None)
    native_ctx, native_mode = estimator._build_native_context()
    native_objective = NativeLogpost(native_ctx, native_mode)

    def native(_objective=native_objective, _theta=theta) -> float:
        return _objective.loglik(_theta)

    def python(_estimator=estimator, _theta=theta) -> float:
        return float(_estimator.loglik(_theta))

    return Case(mode, T, native, python, native(), python())


@dataclass
class RbcCtx:
    solver: object
    compiled: object
    obs: list[str]
    seed: np.ndarray
    calib: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    residual_addr: int
    bc_addr: int
    meas_addr: int
    P0_ukf: np.ndarray
    solved: object
    jitter: float = 1e-8
    sym: int = 1


def _rbc_context() -> RbcCtx:
    model, _ = ModelParser(RBC).get_all()
    R = np.array([[1e-4]], dtype=np.float64)
    kalman = KalmanConfig(R=R, P0=np.eye(3, dtype=np.float64) * 0.1)  # z, k, c
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    n_state = compiled.n_state
    obs = ["c_obs"]

    seed = np.asarray(
        solver.solve(compiled=compiled, order=2).policy.steady_state, dtype=np.float64
    )
    solved = solver.solve(compiled=compiled, ss_seed=seed, order=2)
    base = backend.extract_base_params(compiled)
    P0_base = np.eye(n_state, dtype=np.float64) * 0.1
    return RbcCtx(
        solver=solver,
        compiled=compiled,
        obs=obs,
        seed=seed,
        calib=_cc(backend.build_calib_param_vector(compiled, base), dtype=np.float64),
        Q=_cc(backend.build_Q(compiled, base), dtype=np.float64),
        R=_cc(R, dtype=np.float64),
        residual_addr=compiled.construct_objective_cfunc().address,
        bc_addr=compiled.construct_objective_cfunc_bicomplex().address,
        meas_addr=compiled.construct_measurement_cfunc(obs).address,
        P0_ukf=_cc(
            _resolve_P0(FilterMode.UNSCENTED, n_state, P0_base), dtype=np.float64
        ),
        solved=solved,
    )


def _rbc_case(ctx: RbcCtx, T: int) -> Case:
    rng = np.random.default_rng(20260303)
    sim = ctx.solved.sim(
        T=T,
        shocks={"z": rng.normal(0.0, 0.01, size=T)},
        x0=np.asarray(ctx.solved.policy.steady_state, dtype=np.float64),
        observables=True,
    )
    y = pd.DataFrame({"c_obs": sim["c_obs"]})
    estimator = Estimator(
        solver=ctx.solver,
        compiled=ctx.compiled,
        y=y,
        observables=ctx.obs,
        filter_mode="unscented",
        estimated_params=["beta"],
        ss_seed=ctx.seed,
        jitter=ctx.jitter,
        symmetrize=bool(ctx.sym),
        R=ctx.R,
        P0=np.eye(ctx.compiled.n_state, dtype=np.float64) * 0.1,
    )
    theta = estimator.resolve_theta0(None)
    native_ctx, native_mode = estimator._build_native_context()
    native_objective = NativeLogpost(native_ctx, native_mode)

    def native(_objective=native_objective, _theta=theta) -> float:
        return _objective.loglik(_theta)

    def python(_estimator=estimator, _theta=theta) -> float:
        return float(_estimator.loglik(_theta))

    return Case("unscented", T, native, python, native(), python())


# --------------------------------------------------------------------------- #
# Grid + timing                                                               #
# --------------------------------------------------------------------------- #
def _log_grid(t_min: int, t_max: int, steps: int) -> list[int]:
    """Whole-number log-spaced T values, deduplicated and sorted."""
    raw = np.logspace(np.log10(t_min), np.log10(t_max), steps)
    return sorted({int(round(v)) for v in raw})


def _time(
    fn: Callable[[], float], max_reps: int, warmup: int, budget: float
) -> list[float]:
    """Time fn: `warmup` untimed calls, then up to `max_reps`, stopping once the
    cumulative timed wall clock exceeds `budget` (always at least one sample)."""
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


def _us(seconds: float) -> str:
    return f"{seconds * 1e6:12.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["linear", "extended", "unscented"],
        choices=["linear", "extended", "unscented"],
        help="Filter modes to benchmark (default: all three).",
    )
    parser.add_argument("--t-min", type=int, default=100, help="Smallest T.")
    parser.add_argument("--t-max", type=int, default=1_000_000, help="Largest T.")
    parser.add_argument("--steps", type=int, default=10, help="Log-grid points.")
    parser.add_argument("--max-reps", type=int, default=200, help="Reps cap per point.")
    parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed calls per point."
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=2.0,
        help="Wall-clock seconds cap for each timed series (bounds large T).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for the native/Python parity gate.",
    )
    args = parser.parse_args()

    grid = _log_grid(args.t_min, args.t_max, args.steps)
    ctx_post82 = _post82_context() if set(args.modes) & set(_POST82_MODES) else None
    ctx_rbc = _rbc_context() if "unscented" in args.modes else None

    def build(mode: str, T: int) -> Case:
        if mode in _POST82_MODES:
            assert ctx_post82 is not None
            return _post82_case(ctx_post82, mode, T)
        assert ctx_rbc is not None
        return _rbc_case(ctx_rbc, T)

    print(
        f"T-grid={grid}\nmax_reps={args.max_reps} warmup={args.warmup} "
        f"budget={args.budget}s  (times in microseconds/eval)\n"
    )
    header = (
        f"{'mode':<10} {'T':>9} {'native med':>13} {'python med':>13} "
        f"{'speedup':>8} {'native min':>13} {'reps n/p':>10}  {'loglik':>14}"
    )
    exit_code = 0
    for mode in args.modes:
        print(header)
        print("-" * len(header))
        for T in grid:
            case = build(mode, T)
            if not np.isclose(case.ll_native, case.ll_python, rtol=args.rtol, atol=0.0):
                print(
                    f"{mode:<10} {T:>9}  PARITY FAIL native={case.ll_native!r} "
                    f"python={case.ll_python!r}",
                    file=sys.stderr,
                )
                exit_code = 1
                continue
            nat = _time(case.native, args.max_reps, args.warmup, args.budget)
            pyt = _time(case.python, args.max_reps, args.warmup, args.budget)
            nat_med, pyt_med = statistics.median(nat), statistics.median(pyt)
            speedup = pyt_med / nat_med if nat_med > 0 else float("nan")
            print(
                f"{mode:<10} {T:>9} {_us(nat_med)} {_us(pyt_med)} "
                f"{speedup:7.2f}x {_us(min(nat))} {len(nat):>4}/{len(pyt):<4}  "
                f"{case.ll_native:14.4f}"
            )
        print()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
