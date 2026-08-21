"""Benchmark end-to-end MCMC: the native path vs the oracle-driven pre-native one.

  * native  -> ``Estimator.mcmc``: the native ``run_mcmc`` mainloop (proposal,
    accept/reject, in-loop adaptation, and the per-draw objective all in C behind
    one Python handoff), plus the post-loop theta -> params conversion.
  * pre-native -> ``native_algorithm_mimic`` driving ``est._safe_logpost``: a
    Python random-walk-Metropolis loop calling the Python objective (solve +
    filter) once per draw, plus the same theta -> params conversion. This is the
    timing profile ``Estimator.mcmc`` had before it went native; the objective
    eval dominates, so the numpy-era proposal / covariance details it differs from
    are negligible for wall time.

Both paths run the SAME algorithm (Cholesky proposal on numpy ``standard_normal``
draws + running covariance) on the SAME seed, and the native and Python objectives
are bit-identical, so a fixed-seed chain is a parity guard (the accept rates must
match) as well as a timing comparison -- a silent divergence can't flatter the
numbers. The two paths still differ by the Cholesky ULP (``sdsge_chol`` vs LAPACK)
once adaptation is on, which does not flip accepts over these short chains.

Two model families, so the per-draw objective spans both regimes:
  * post82 -> tests/fixtures/models/POST82.yaml (gap model, first-order, linear
    filter);
  * rbc2   -> tests/fixtures/models/rbc_second_order.yaml (levels, second-order
    solve, unscented filter), which drives the nonlinear per-draw objective.

Short chains (``--n-draws`` kept after ``--burn-in``) are repeated ``--reps``
times; reported figures are post-warmup medians. The per-step figure is the
whole-chain median divided by the total objective evals (burn-in + kept), directly
comparable to scripts/bench_native_objective.py's per-eval numbers.

Usage:
    uv run python scripts/bench_native_mcmc.py
    uv run python scripts/bench_native_mcmc.py --n-draws 500 --burn-in 500 --reps 100
    uv run python scripts/bench_native_mcmc.py --models rbc2

This is a developer benchmark, not shipped package code and not a correctness test
(that lives in tests/estimation/test_native_mcmc_parity.py).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "models"
sys.path.insert(0, str(ROOT / "tests"))

from _oracles.estimation import native_algorithm_mimic  # noqa: E402
from SymbolicDSGE import DSGESolver, ModelParser  # noqa: E402
from SymbolicDSGE.estimation import Estimator  # noqa: E402
from SymbolicDSGE.kalman.config import KalmanConfig  # noqa: E402

_DEVNULL = open(os.devnull, "w")

# Normal-prior (mean, std) per estimable parameter used here; the mean doubles as
# theta0 (each equals the fixture's calibrated value).
_PRIOR = {
    "psi_pi": (2.0, 0.5),
    "rho_r": (0.8, 0.1),
    "rho": (0.95, 0.02),
}

# (label, model, filter_mode, estimated params, adapt, proposal_scale).
_CONFIGS = [
    ("post82 lin d2 adapt", "post82", "linear", ["psi_pi", "rho_r"], True, 0.1),
    ("post82 lin d2 fixed", "post82", "linear", ["psi_pi", "rho_r"], False, 0.1),
    ("post82 lin d1 adapt", "post82", "linear", ["psi_pi"], True, 0.1),
    ("rbc2 unsc d1 adapt", "rbc2", "unscented", ["rho"], True, 0.02),
    ("rbc2 unsc d1 fixed", "rbc2", "unscented", ["rho"], False, 0.02),
]

_MODELS = {c[1] for c in _CONFIGS}


# --------------------------------------------------------------------------- #
# Fixture bundles: compiled model + simulated panel + ss seed, built once.     #
# --------------------------------------------------------------------------- #
def _post82_bundle() -> dict:
    from sympy import Symbol

    model, kalman = ModelParser(FIXTURES / "POST82.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    steady = np.zeros(len(compiled.var_names), dtype=np.float64)
    solved = solver.solve(compiled=compiled, ss_seed=steady)
    calib = compiled.config.calibration
    sig = {
        s: float(calib.parameters[calib.shock_std[Symbol(s)]])
        for s in ("e_g", "e_z", "e_r")
    }
    rng = np.random.default_rng(20260724)
    sim = solved.sim(
        T=48,
        shocks={
            "g": rng.normal(0.0, sig["e_g"], 48),
            "z": rng.normal(0.0, sig["e_z"], 48),
            "r": rng.normal(0.0, sig["e_r"], 48),
        },
        x0=steady,
        observables=True,
    )
    y = pd.DataFrame(
        {"OutGap": sim["OutGap"][1:], "Infl": sim["Infl"][1:], "Rate": sim["Rate"][1:]}
    )
    return dict(
        solver=solver, compiled=compiled, y=y, obs=["OutGap", "Infl", "Rate"], ss=steady
    )


def _rbc2_bundle() -> dict:
    # The gap model is degenerate for the UKF; the levels RBC exercises the
    # order-2 solve + unscented filter. R / P0 are supplied here (the fixture
    # carries no measurement config), matching scripts/bench_native_objective.py.
    model, _ = ModelParser(FIXTURES / "rbc_second_order.yaml").get_all()
    kalman = KalmanConfig(
        R=np.array([[1e-4]], dtype=np.float64), P0=np.eye(3, dtype=np.float64) * 0.1
    )
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    seed = np.asarray(
        solver.solve(compiled=compiled, order=2).policy.steady_state, dtype=np.float64
    )
    solved = solver.solve(compiled=compiled, ss_seed=seed, order=2)
    rng = np.random.default_rng(20260303)
    sim = solved.sim(
        T=48, shocks={"z": rng.normal(0.0, 0.01, 48)}, x0=seed, observables=True
    )
    y = pd.DataFrame({"c_obs": sim["c_obs"][1:]})
    return dict(solver=solver, compiled=compiled, y=y, obs=["c_obs"], ss=seed)


_BUNDLE_FN = {"post82": _post82_bundle, "rbc2": _rbc2_bundle}
_bundles: dict[str, dict] = {}


def _bundle(key: str) -> dict:
    if key not in _bundles:
        _bundles[key] = _BUNDLE_FN[key]()
    return _bundles[key]


def _make_estimator(bundle: dict, mode: str, names: list[str]) -> Estimator:
    priors = {
        n: Estimator.make_prior(
            distribution="normal",
            parameters={"mean": _PRIOR[n][0], "std": _PRIOR[n][1]},
            transform="identity",
        )
        for n in names
    }
    return Estimator(
        solver=bundle["solver"],
        compiled=bundle["compiled"],
        y=bundle["y"],
        observables=bundle["obs"],
        filter_mode=mode,
        estimated_params=names,
        priors=priors,
        ss_seed=bundle["ss"],
    )


@dataclass
class Case:
    """One config row's native / pre-native chain closures + parity accepts."""

    label: str
    steps: int
    native: Callable[[], object]
    prenat: Callable[[], object]
    acc_native: float
    acc_prenat: float


def _build_case(
    label: str,
    model: str,
    mode: str,
    names: list[str],
    adapt: bool,
    proposal_scale: float,
    *,
    n_draws: int,
    burn_in: int,
    seed: int,
) -> Case:
    est = _make_estimator(_bundle(model), mode, names)
    theta0 = np.ascontiguousarray(
        est.resolve_theta0(np.array([_PRIOR[n][0] for n in names], dtype=np.float64)),
        dtype=np.float64,
    )
    kw = dict(
        n_draws=n_draws,
        burn_in=burn_in,
        thin=1,
        adapt=adapt,
        adapt_start=100,
        proposal_scale=proposal_scale,
        adapt_epsilon=1e-8,
    )
    param_names = list(est.param_names)

    def native() -> object:
        with contextlib.redirect_stdout(_DEVNULL):
            return est.mcmc(theta0=theta0, random_state=seed, **kw)

    def prenat() -> object:
        ref = native_algorithm_mimic(
            est._safe_logpost, theta0, np.random.default_rng(seed), **kw
        )
        # theta -> params, mirroring Estimator.mcmc's post-loop (the pre-native
        # path produced named-parameter samples too).
        out = np.empty_like(ref.kept)
        for i in range(ref.kept.shape[0]):
            p = est.theta_to_params(ref.kept[i])
            for j, nm in enumerate(param_names):
                out[i, j] = float(p[nm])
        return out

    nat_res = native()
    ref = native_algorithm_mimic(
        est._safe_logpost, theta0, np.random.default_rng(seed), **kw
    )
    return Case(
        label,
        burn_in + n_draws,
        native,
        prenat,
        float(nat_res.accept_rate),
        float(ref.accept_rate),
    )


def _time(
    fn: Callable[[], object], reps: int, warmup: int, budget: float
) -> list[float]:
    """Time fn: `warmup` untimed calls, then up to `reps`, stopping once the
    cumulative timed wall clock exceeds `budget` (always at least one sample)."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    start = time.perf_counter()
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
        if time.perf_counter() - start > budget:
            break
    return samples


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:11.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=250, help="Kept draws.")
    parser.add_argument("--burn-in", type=int, default=250, help="Burn-in draws.")
    parser.add_argument("--reps", type=int, default=200, help="Reps cap per path.")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed chains.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=sorted(_MODELS),
        choices=sorted(_MODELS),
        help="Model families to benchmark (default: all).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=30.0,
        help="Wall-clock seconds cap per timed series (bounds the slow path).",
    )
    parser.add_argument(
        "--accept-tol",
        type=float,
        default=0.02,
        help="Max |native - pre-native| accept-rate gap before flagging divergence.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Fixed chain seed.")
    args = parser.parse_args()

    configs = [c for c in _CONFIGS if c[1] in set(args.models)]

    print(
        f"n_draws={args.n_draws} burn_in={args.burn_in} "
        f"(steps={args.burn_in + args.n_draws})  reps={args.reps} "
        f"warmup={args.warmup} budget={args.budget}s seed={args.seed}\n"
        "native = Estimator.mcmc; pre-native = oracle mimic on the Python "
        "objective  (whole-chain times in ms)\n"
    )
    header = (
        f"{'config':<22} {'steps':>6} {'nat med':>11} {'pnat med':>11} "
        f"{'speedup':>8} {'nat us/step':>12} {'reps n/p':>10}  {'accept n/p':>13}"
    )
    print(header)
    print("-" * len(header))

    exit_code = 0
    for label, model, mode, names, adapt, pscale in configs:
        case = _build_case(
            label,
            model,
            mode,
            names,
            adapt,
            pscale,
            n_draws=args.n_draws,
            burn_in=args.burn_in,
            seed=args.seed,
        )
        flag = ""
        if abs(case.acc_native - case.acc_prenat) > args.accept_tol:
            flag = "  <- ACCEPT DIVERGENCE"
            exit_code = 1

        nat = _time(case.native, args.reps, args.warmup, args.budget)
        pnat = _time(case.prenat, args.reps, args.warmup, args.budget)
        nat_med, pnat_med = statistics.median(nat), statistics.median(pnat)
        speedup = pnat_med / nat_med if nat_med > 0 else float("nan")
        us_step = nat_med * 1e6 / case.steps
        print(
            f"{label:<22} {case.steps:>6} {_ms(nat_med)} {_ms(pnat_med)} "
            f"{speedup:7.2f}x {us_step:12.2f} {len(nat):>4}/{len(pnat):<4}  "
            f"{case.acc_native:5.3f}/{case.acc_prenat:<5.3f}{flag}"
        )

    print()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
