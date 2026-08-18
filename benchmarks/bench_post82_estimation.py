"""Benchmark POST82 maximum likelihood and MAP estimation against Dynare.

The timed native entries are ``Estimator.mle`` and ``Estimator.map``. Dynare
preprocesses and executes one initial estimation outside the timer, then the
timed repetitions call ``dynare_estimation``. Parse, compilation, input-panel
generation, model preprocessing, and warmup are excluded.

Raw outputs are discarded unless ``--output-dir`` is supplied. Relative output
paths are resolved from this benchmark's directory.
"""

from __future__ import annotations

import argparse
import contextlib
from contextlib import nullcontext
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURES = ROOT / "tests" / "fixtures" / "models"
MODEL = FIXTURES / "POST82.yaml"
DYNARE_MODEL = FIXTURES / "post82_kf.mod"
OBSERVABLES = ["OutGap", "Infl", "Rate"]
SHOCK_NAMES = ["e_g", "e_z", "e_r"]

sys.path.insert(0, str(ROOT))
from SymbolicDSGE import DSGESolver, ModelParser  # noqa: E402
from SymbolicDSGE.estimation import Estimator, make_prior  # noqa: E402


@dataclass(frozen=True)
class Routine:
    names: tuple[str, ...]
    theta0: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]


ROUTINES = {
    "mle": Routine(("psi_pi", "rho_r"), (2.0, 0.8), ((1.0, 5.0), (0.0, 0.99))),
    "map": Routine(("psi_pi",), (2.0,), ((1.0, 5.0),)),
}


def _quoted_matlab(value: Path | str) -> str:
    return str(value).replace("\\", "/").replace("'", "''")


def _matlab_matrix(values: np.ndarray) -> str:
    return "[" + "; ".join(" ".join(f"{x:.17g}" for x in row) for row in values) + "]"


def _prepare(periods: int) -> tuple[object, object, np.ndarray]:
    model, kalman = ModelParser(MODEL).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    solved = solver.solve(compiled=compiled, order=1)
    base = np.array(
        [
            [0.25, -0.10, 0.40],
            [-0.70, 0.55, -0.15],
            [0.10, 0.20, 0.00],
            [0.60, -0.35, 0.25],
            [-0.20, 0.05, -0.50],
            [0.00, 0.30, 0.10],
            [0.35, -0.45, 0.20],
            [-0.45, 0.15, -0.30],
            [0.15, 0.60, 0.05],
            [-0.10, -0.25, 0.35],
            [0.05, 0.40, -0.20],
            [0.20, -0.05, 0.15],
        ],
        dtype=np.float64,
    )
    shocks = np.resize(base, (periods, len(SHOCK_NAMES)))
    sim = solved.sim(
        T=periods,
        shocks={name: shocks[:, i] for i, name in enumerate(SHOCK_NAMES)},
        observables=True,
    )
    y = np.column_stack([np.asarray(sim.observables[name]) for name in OBSERVABLES])
    return solver, compiled, y


def _make_estimator(
    solver: object,
    compiled: object,
    y: np.ndarray,
    routine: str,
    zero_r: bool,
) -> Estimator:
    spec = ROUTINES[routine]
    priors = None
    if routine == "map":
        priors = {
            "psi_pi": make_prior(
                distribution="normal",
                parameters={"mean": 2.0, "std": 0.5},
                transform="identity",
            )
        }
    return Estimator(
        solver=solver,
        compiled=compiled,
        y=y,
        observables=OBSERVABLES,
        filter_mode="linear",
        estimated_params=list(spec.names),
        priors=priors,
        ss_seed=np.zeros(len(compiled.var_names), dtype=np.float64),
        R=(
            np.zeros((len(OBSERVABLES), len(OBSERVABLES)), dtype=np.float64)
            if zero_r
            else None
        ),
    )


def _time(fn, warmup: int, reps: int) -> tuple[np.ndarray, object]:
    for _ in range(warmup):
        fn()
    times = np.empty(reps, dtype=np.float64)
    result = None
    for i in range(reps):
        started = time.perf_counter()
        result = fn()
        times[i] = time.perf_counter() - started
    assert result is not None
    return times, result


def _native(
    solver: object,
    compiled: object,
    y: np.ndarray,
    routine: str,
    warmup: int,
    reps: int,
    zero_r: bool,
) -> dict:
    estimator = _make_estimator(solver, compiled, y, routine, zero_r)
    spec = ROUTINES[routine]
    theta0 = np.asarray(spec.theta0, dtype=np.float64)
    if routine == "mle":
        fn = lambda: estimator.mle(theta0=theta0, bounds=spec.bounds)
    else:
        fn = lambda: estimator.map(theta0=theta0, bounds=spec.bounds)
    with (
        open(os.devnull, "w", encoding="utf-8") as sink,
        contextlib.redirect_stdout(sink),
    ):
        times, result = _time(fn, warmup, reps)
    theta = np.asarray(result.x, dtype=np.float64)
    loglik = float(estimator.loglik(theta))
    logprior = float(estimator.logprior(theta))
    probes = (
        np.array([[2.0, 2.5], [0.8, 0.75]], dtype=np.float64)
        if routine == "mle"
        else np.array([[2.0, 2.5]], dtype=np.float64)
    )
    target = np.array(
        [
            (
                estimator.loglik(probes[:, i])
                if routine == "mle"
                else estimator.logpost(probes[:, i])
            )
            for i in range(probes.shape[1])
        ],
        dtype=np.float64,
    )
    objective = estimator.loglik if routine == "mle" else estimator.logpost
    objective_times, _ = _time(
        lambda: np.array(
            [objective(probes[:, i]) for i in range(probes.shape[1])],
            dtype=np.float64,
        ),
        warmup,
        reps,
    )
    return {
        "times": times,
        "objective_times": objective_times / probes.shape[1],
        "theta": theta,
        "loglik": loglik,
        "logprior": logprior,
        "terminal_target": loglik + logprior,
        "nfev": int(result.nfev),
        "nit": -1 if result.nit is None else int(result.nit),
        "success": bool(result.success),
        "probes": probes,
        "probe_target": target,
    }


def _write_data(y: np.ndarray, path: Path) -> None:
    lines = ["% Generated by bench_post82_estimation.py. Do not edit."]
    for column, name in enumerate(OBSERVABLES):
        lines.extend((f"{name} = [", *(f"{v:.17g}" for v in y[:, column]), "];"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model(routine: str, path: Path, periods: int, zero_r: bool) -> None:
    source = DYNARE_MODEL.read_text(encoding="utf-8")
    source = source.replace(
        "calib_smoother(datafile = post82_kf_data, filtered_vars, filter_step_ahead = [1]);",
        "",
    )
    if zero_r:
        source = source.replace("sig_me  = 1.00;", "sig_me  = 0.00;")
    if routine == "mle":
        estimated = "psi_pi, 2.0, 1.0, 5.0;\nrho_r, 0.8, 0.0, 0.99;"
    else:
        estimated = "psi_pi, 2.0, 1.0, 5.0, normal_pdf, 2.0, 0.5;"
    source += (
        "\nestimated_params;\n" + estimated + "\nend;\n"
        "estimation(datafile = post82_estimation_data, nobs = "
        + str(periods)
        + ", mode_compute = 4, mh_replic = 0, cova_compute = 0);\n"
    )
    path.write_text(source, encoding="utf-8")


def _run_dynare(
    runtime: str,
    routine: str,
    y: np.ndarray,
    probes: np.ndarray,
    output_dir: Path,
    warmup: int,
    reps: int,
    dynare_path: str,
    matlab_bin: str,
    octave_bin: str,
    zero_r: bool,
) -> Path:
    root = Path(dynare_path).resolve().parent
    platform = "matlab" if runtime == "matlab" else "octave"
    if not (root / "mex" / platform).is_dir():
        raise RuntimeError(f"Dynare at {root} has no {platform} MEX directory.")
    result_path = output_dir / f"dynare_{routine}_{runtime}.mat"
    with tempfile.TemporaryDirectory(
        prefix=f"dynare-{routine}-{runtime}-", dir=output_dir
    ) as tmp:
        workdir = Path(tmp)
        model_name = f"post82_estimation_{routine}"
        _write_model(routine, workdir / f"{model_name}.mod", y.shape[0], zero_r)
        _write_data(y, workdir / "post82_estimation_data.m")
        expression = (
            f"addpath('{_quoted_matlab(dynare_path)}'); addpath('{_quoted_matlab(SCRIPT_DIR)}'); "
            f"bench_post82_estimation_dynare('{_quoted_matlab(workdir)}', '{model_name}', "
            f"{_matlab_matrix(probes)}, {warmup}, {reps}, '{_quoted_matlab(result_path)}');"
        )
        command = (
            [matlab_bin, "-batch", expression]
            if runtime == "matlab"
            else [octave_bin, "--quiet", "--eval", expression]
        )
        completed = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Dynare {runtime} failed for {routine}:\n{completed.stderr.strip()}"
            )
    return result_path


def _print_table(routine: str, native: dict, dynare: dict[str, dict]) -> None:
    native_median = float(np.median(native["times"]))
    print(f"\n{routine.upper()}: implementation-specific optimizer")
    header = f"{'runtime':<26} {'median ms':>12} {'speedup':>10} {'nfev':>8} {'max |delta theta|':>20} {'|delta terminal obj|':>20}"
    print(header)
    print("-" * len(header))
    print(
        f"{'SymbolicDSGE':<26} {native_median * 1e3:12.2f} {'1.00x':>10} {native['nfev']:8d} {'0.000e+00':>20} {'0.000e+00':>20}"
    )
    for runtime, result in dynare.items():
        median = float(np.median(result["times"]))
        delta = float(np.max(np.abs(native["theta"] - result["theta"])))
        objective_delta = abs(native["terminal_target"] - result["terminal_target"])
        print(
            f"{'Dynare-' + runtime:<26} {median * 1e3:12.2f} {median / native_median:9.2f}x {'n/a':>8} {delta:20.3e} {objective_delta:20.3e}"
        )
    print(
        f"native log likelihood: {native['loglik']:.12g}  native log prior: {native['logprior']:.12g}"
    )

    native_objective_median = float(np.median(native["objective_times"]))
    objective_name = "log likelihood" if routine == "mle" else "log posterior"
    print(f"\n{routine.upper()} fixed-theta {objective_name}: shared probes")
    header = f"{'runtime':<26} {'median ms / eval':>18} {'speedup':>10} {'max |delta objective|':>24}"
    print(header)
    print("-" * len(header))
    print(
        f"{'SymbolicDSGE':<26} {native_objective_median * 1e3:18.3f} {'1.00x':>10} {'0.000e+00':>24}"
    )
    for runtime, result in dynare.items():
        median = float(np.median(result["objective_times"]))
        objective_delta = float(
            np.max(np.abs(native["probe_target"] - result["probe_target"]))
        )
        print(
            f"{'Dynare-' + runtime:<26} {median * 1e3:18.3f} {median / native_objective_median:9.2f}x {objective_delta:24.3e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--routines", nargs="+", choices=sorted(ROUTINES), default=["mle", "map"]
    )
    parser.add_argument(
        "--runtimes",
        nargs="+",
        choices=["native", "matlab", "octave"],
        default=["native"],
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dynare-matlab-path", default="")
    parser.add_argument("--matlab-bin", default="matlab")
    parser.add_argument("--octave-bin", default="octave-cli")
    parser.add_argument(
        "--zero-r",
        action="store_true",
        help="Override the calibrated measurement covariance with R = 0 on both sides.",
    )
    args = parser.parse_args()
    if args.periods < 1 or args.warmup < 0 or args.reps < 1:
        parser.error(
            "periods and reps must be positive, and warmup must be nonnegative"
        )
    if set(args.runtimes) - {"native"} and not args.dynare_matlab_path:
        parser.error("--dynare-matlab-path is required for Dynare runtimes")
    context = (
        nullcontext(
            args.output_dir.resolve()
            if args.output_dir and args.output_dir.is_absolute()
            else (SCRIPT_DIR / args.output_dir).resolve()
        )
        if args.output_dir
        else tempfile.TemporaryDirectory(prefix="symbolicdsge-post82-estimation-")
    )
    with context as output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        solver, compiled, y = _prepare(args.periods)
        print(
            f"POST82 estimation: periods={args.periods} warmup={args.warmup} reps={args.reps}"
        )
        print(
            "Setup, preprocessing, input generation, and warmup are outside the timer."
        )
        for routine in args.routines:
            native = _native(
                solver, compiled, y, routine, args.warmup, args.reps, args.zero_r
            )
            np.savez(output_dir / f"native_{routine}.npz", **native)
            dynare = {}
            for runtime in (item for item in args.runtimes if item != "native"):
                raw = loadmat(
                    _run_dynare(
                        runtime,
                        routine,
                        y,
                        native["probes"],
                        output_dir,
                        args.warmup,
                        args.reps,
                        args.dynare_matlab_path,
                        args.matlab_bin,
                        args.octave_bin,
                        args.zero_r,
                    ),
                    squeeze_me=True,
                )
                dynare[runtime] = {
                    "times": np.asarray(raw["times"], dtype=np.float64).reshape(-1),
                    "objective_times": np.asarray(
                        raw["objective_times"], dtype=np.float64
                    ).reshape(-1),
                    "theta": np.asarray(raw["theta"], dtype=np.float64).reshape(-1),
                    "terminal_target": float(
                        np.asarray(raw["terminal_target"]).squeeze()
                    ),
                    "probe_target": np.asarray(
                        raw["probe_target"], dtype=np.float64
                    ).reshape(-1),
                }
            _print_table(routine, native, dynare)
        if args.output_dir:
            (output_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "periods": args.periods,
                        "warmup": args.warmup,
                        "reps": args.reps,
                        "zero_r": args.zero_r,
                    },
                    indent=2,
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
