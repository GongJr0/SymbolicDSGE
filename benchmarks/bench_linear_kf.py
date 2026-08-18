"""Run first-order models through linear Kalman filters.

The timed workloads are SymbolicDSGE's public likelihood-only and retained
history filter entries, Dynare's direct ``kalman_filter`` likelihood entry,
and Dynare's generated ``calib_smoother`` entry. Parsing, compilation or
preprocessing, solution, data generation, and warmup are outside the timer.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
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

sys.path.insert(0, str(ROOT))
from SymbolicDSGE import DSGESolver, ModelParser, SolvedModel
from SymbolicDSGE.estimation import backend
from SymbolicDSGE.kalman.filter import FilterResult, KalmanFilter


@dataclass(frozen=True)
class CaseSpec:
    label: str
    yaml_name: str
    mod_name: str
    native_observables: tuple[str, ...]
    dynare_observables: tuple[str, ...]
    shock_names: tuple[str, ...]
    seed: int
    dynare_data_file: str = "fixture_kf_data.m"
    shock_mode: str = "q_scaled"
    native_to_dynare_state: tuple[tuple[str, str], ...] = ()


CASES = {
    "ls2004": CaseSpec(
        label="Lubik-Schorfheide 2004",
        yaml_name="POST82.yaml",
        mod_name="post82_kf.mod",
        native_observables=("OutGap", "Infl", "Rate"),
        dynare_observables=("OutGap", "Infl", "Rate"),
        shock_names=("e_g", "e_z", "e_r"),
        seed=1982,
        dynare_data_file="post82_kf_data.m",
        shock_mode="post82",
    ),
    "sw2007": CaseSpec(
        label="SW2007",
        yaml_name="sw2007.yaml",
        mod_name="sw2007.mod",
        native_observables=("dy", "dc", "dinve", "labobs", "pinfobs", "dw", "robs"),
        dynare_observables=("dy", "dc", "dinve", "labobs", "pinfobs", "dw", "robs"),
        shock_names=("ea", "eb", "eg", "eqs", "em", "epinf", "ew"),
        seed=2007,
        dynare_data_file="sw2007_kf_data.m",
        shock_mode="normal_quarter",
        native_to_dynare_state=(
            ("rep_dy", "dy"),
            ("rep_dc", "dc"),
            ("rep_dinve", "dinve"),
            ("rep_dw", "dw"),
        ),
    ),
    "g2015": CaseSpec(
        label="Gali 2015",
        yaml_name="gali_2015.yaml",
        mod_name="gali_2015.mod",
        native_observables=("obs_pi_ann", "obs_i_ann"),
        dynare_observables=("pi_ann", "i_ann"),
        shock_names=("eps_a", "eps_nu", "eps_z"),
        seed=2015,
    ),
    "gm2005": CaseSpec(
        label="Gali-Monacelli 2005",
        yaml_name="gali_monacelli_2005.yaml",
        mod_name="gali_monacelli_2005.mod",
        native_observables=("obs_pi", "obs_r"),
        dynare_observables=("pi", "r"),
        shock_names=("eps_star", "eps_a"),
        seed=2005,
    ),
    "i2004": CaseSpec(
        label="Ireland 2004",
        yaml_name="ireland_2004.yaml",
        mod_name="ireland_2004.mod",
        native_observables=("obs_gobs", "obs_piobs"),
        dynare_observables=("gobs", "piobs"),
        shock_names=("eps_a", "eps_e", "eps_z", "eps_r"),
        seed=2004,
    ),
}


@dataclass
class NativeCase:
    solved: SolvedModel
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    d: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    y: np.ndarray
    state_names: tuple[str, ...]


def _quoted_matlab(value: Path | str) -> str:
    return str(value).replace("\\", "/").replace("'", "''")


def _max_abs(left: np.ndarray | float, right: np.ndarray | float) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _draw_shocks(spec: CaseSpec, periods: int, Q: np.ndarray) -> np.ndarray:
    if spec.shock_mode == "post82":
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
        return np.resize(base, (periods, len(spec.shock_names)))
    draws = np.random.default_rng(spec.seed).normal(
        size=(periods, len(spec.shock_names))
    )
    if spec.shock_mode == "normal_quarter":
        return 0.25 * draws
    return draws @ np.linalg.cholesky(Q).T


def _prepare(spec: CaseSpec, periods: int) -> NativeCase:
    model, kalman = ModelParser(FIXTURES / spec.yaml_name).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    solved = solver.solve(compiled=compiled, order=1)

    base_params = backend.extract_base_params(compiled)
    Q = np.asarray(backend.build_Q(compiled, base_params), dtype=np.float64)
    shocks = _draw_shocks(spec, periods, Q)
    sim = solved.sim(
        T=periods,
        shocks={name: shocks[:, index] for index, name in enumerate(spec.shock_names)},
        observables=True,
    )
    y = np.column_stack(
        [
            np.asarray(sim.observables[name], dtype=np.float64)
            for name in spec.native_observables
        ]
    )
    C, d = compiled.build_affine_measurement_matrices(
        base_params,
        list(spec.native_observables),
        np.asarray(solved.policy.steady_state),
    )
    return NativeCase(
        solved=solved,
        A=np.asarray(solved.policy.A, dtype=np.float64),
        B=np.asarray(solved.policy.B, dtype=np.float64),
        C=np.asarray(C, dtype=np.float64),
        d=np.asarray(d, dtype=np.float64),
        Q=Q,
        R=np.asarray(
            backend.build_R(
                compiled, kalman, list(spec.native_observables), base_params
            ),
            dtype=np.float64,
        ),
        y=y,
        state_names=tuple(compiled.var_names),
    )


def _write_dynare_data(spec: CaseSpec, y_values: np.ndarray, path: Path) -> None:
    lines = ["% Generated by bench_linear_kf.py. Do not edit."]
    for column, name in enumerate(spec.dynare_observables):
        values = "\n".join(f"{value:.17g}" for value in y_values[:, column])
        lines.extend((f"{name} = [", values, "];"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _time(fn, warmup: int, reps: int) -> tuple[np.ndarray, FilterResult]:
    for _ in range(warmup):
        fn()
    times = np.empty(reps, dtype=np.float64)
    result = None
    for index in range(reps):
        started = time.perf_counter()
        result = fn()
        times[index] = time.perf_counter() - started
    assert result is not None
    return times, result


def _time_native(
    spec: CaseSpec, case: NativeCase, warmup: int, reps: int, use_joseph: bool
) -> dict:
    def loglik_only() -> FilterResult:
        return KalmanFilter.run(
            case.A,
            case.B,
            case.C,
            case.d,
            case.Q,
            case.R,
            case.y,
            P0=None,
            _store_history=False,
            symmetrize=False,
            joseph_cov=use_joseph,
        )

    def filter_history() -> FilterResult:
        return case.solved.kalman(
            case.y,
            filter_mode="linear",
            observables=list(spec.native_observables),
            P0=None,
            symmetrize=False,
            joseph_cov=False,
        )

    loglik_times, loglik_result = _time(loglik_only, warmup, reps)
    history_times, history_result = _time(filter_history, warmup, reps)
    return {
        "loglik_times": loglik_times,
        "history_times": history_times,
        "loglik": float(loglik_result.loglik),
        "x_pred": np.asarray(history_result.x_pred, dtype=np.float64),
        "x_filt": np.asarray(history_result.x_filt, dtype=np.float64),
    }


def _native_model_info(spec: CaseSpec, case: NativeCase) -> dict[str, int]:
    compiled = case.solved.compiled
    return {
        "declared_variables": compiled.n_var,
        "filter_state_dimension": case.A.shape[0],
        "predetermined_variables": compiled.n_state,
        "observables": len(spec.native_observables),
        "shocks": compiled.n_exog,
        "parameters": compiled.n_par,
    }


def _save_native(result: dict, output_dir: Path, model_info: dict[str, int]) -> None:
    np.savez(output_dir / "native.npz", **result)
    metadata = {
        "runtime": "SymbolicDSGE native",
        "loglik_only_entry_point": "KalmanFilter.run(_store_history=False)",
        "filter_history_entry_point": "FirstOrderSolvedModel.kalman",
        "repetitions": int(result["loglik_times"].size),
        "model_dimensions": model_info,
    }
    (output_dir / "native.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _matlab_cellstr(values: tuple[str, ...]) -> str:
    return "{" + ", ".join(f"'{value}'" for value in values) + "}"


def _run_dynare(
    runtime: str,
    spec: CaseSpec,
    output_dir: Path,
    y_values: np.ndarray,
    warmup: int,
    reps: int,
    dynare_matlab_path: str,
    matlab_bin: str,
    octave_bin: str,
) -> Path:
    dynare_root = Path(dynare_matlab_path).resolve().parent
    mex_platform = "matlab" if runtime == "matlab" else "octave"
    if not (dynare_root / "mex" / mex_platform).is_dir():
        raise RuntimeError(
            f"Dynare at {dynare_root} has no {mex_platform} MEX directory."
        )
    runner = SCRIPT_DIR / "bench_linear_kf_dynare.m"
    result_path = output_dir / f"dynare_{runtime}.mat"
    with tempfile.TemporaryDirectory(
        prefix=f"dynare-{runtime}-", dir=output_dir
    ) as temporary_dir:
        workdir = Path(temporary_dir)
        shutil.copy2(FIXTURES / spec.mod_name, workdir / spec.mod_name)
        data_file = spec.dynare_data_file
        _write_dynare_data(spec, y_values, workdir / data_file)
        expression = (
            f"addpath('{_quoted_matlab(dynare_matlab_path)}'); "
            f"addpath('{_quoted_matlab(runner.parent)}'); "
            f"bench_linear_kf_dynare('{_quoted_matlab(workdir)}', "
            f"'{spec.mod_name}', {_matlab_cellstr(spec.dynare_observables)}, "
            f"'{data_file}', {warmup}, {reps}, '{_quoted_matlab(result_path)}');"
        )
        command = (
            [matlab_bin, "-batch", expression]
            if runtime == "matlab"
            else [octave_bin, "--quiet", "--eval", expression]
        )
        subprocess.run(command, check=True, cwd=SCRIPT_DIR, stdout=subprocess.DEVNULL)
    return result_path


def _report_dynare(
    spec: CaseSpec, result_path: Path, case: NativeCase, native: dict
) -> dict:
    raw = loadmat(result_path, squeeze_me=True)
    dynare_names = np.atleast_1d(raw["state_names"]).astype(str).reshape(-1)
    index = {name: position for position, name in enumerate(dynare_names)}
    state_map = dict(spec.native_to_dynare_state)
    state_indices = np.asarray(
        [index[state_map.get(name, name)] for name in case.state_names],
        dtype=np.intp,
    )
    updated = np.asarray(raw["updated"], dtype=np.float64)[:, state_indices]
    filtered = np.asarray(raw["filtered"], dtype=np.float64)[:, state_indices]
    likelihood_times = np.asarray(raw["likelihood_times"], dtype=np.float64).reshape(-1)
    smoother_times = np.asarray(raw["smoother_times"], dtype=np.float64).reshape(-1)
    return {
        "likelihood_median_seconds": float(np.median(likelihood_times)),
        "smoother_median_seconds": float(np.median(smoother_times)),
        "max_abs_loglik": _max_abs(native["loglik"], float(np.asarray(raw["loglik"]))),
        "max_abs_updated": _max_abs(native["x_filt"], updated),
        "max_abs_filtered": _max_abs(native["x_pred"][1:], filtered[:-1]),
        "model_info": {
            "declared_variables": int(np.asarray(raw["declared_endo_nbr"])),
            "filter_state_dimension": int(np.asarray(raw["filter_state_nbr"])),
            "predetermined_variables": int(np.asarray(raw["predetermined_nbr"])),
            "observables": int(np.asarray(raw["observable_nbr"])),
            "shocks": int(np.asarray(raw["shock_nbr"])),
            "parameters": int(np.asarray(raw["parameter_nbr"])),
        },
    }


def _us(seconds: float) -> str:
    return f"{seconds * 1e6:12.2f}"


def _print_tables(
    args: argparse.Namespace,
    spec: CaseSpec,
    case: NativeCase,
    native: dict,
    reports: dict[str, dict],
) -> None:
    loglik_median = float(np.median(native["loglik_times"]))
    history_median = float(np.median(native["history_times"]))
    print(
        f"{spec.label} linear KF: periods={args.periods} warmup={args.warmup} reps={args.reps}\n"
        "Setup is outside the timer: parse, compile or preprocess, solve, input generation, and warmup. Times are microseconds per call.\n"
    )
    native_info = _native_model_info(spec, case)
    dynare_info = next(iter(reports.values()))["model_info"] if reports else None
    print("Model dimensions:")
    header = f"{'quantity':<29} {'SymbolicDSGE':>14} {'Dynare':>14}"
    print(header)
    print("-" * len(header))
    rows = (
        ("Declared model variables", "declared_variables"),
        ("Filter state dimension", "filter_state_dimension"),
        ("Predetermined variables", "predetermined_variables"),
        ("Selected observables", "observables"),
        ("Shocks", "shocks"),
        ("Declared parameters", "parameters"),
    )
    for label, key in rows:
        dynare_value = str(dynare_info[key]) if dynare_info is not None else "not run"
        print(f"{label:<29} {native_info[key]:14d} {dynare_value:>14}")
    if dynare_info is not None:
        print(
            "Dynare's filter state dimension is its preprocessed endogenous count, "
            "including reported observables and any auxiliary variables."
        )
    print()

    header = (
        f"{'runtime':<39} {'median':>12} {'speedup':>10} {'max |delta loglik|':>20}"
    )
    print("Loglikelihood Only:")
    print(header)
    print("-" * len(header))
    print(
        f"{'SymbolicDSGE-loglikonly':<39} {_us(loglik_median)} {'1.00x':>10} {'0.000e+00':>20}"
    )
    for runtime, report in reports.items():
        median = report["likelihood_median_seconds"]
        print(
            f"{'Dynare-' + runtime + '-loglikonly':<39} {_us(median)} {median / loglik_median:9.2f}x {report['max_abs_loglik']:20.3e}"
        )

    print("\nRetained Results:")
    print(
        "SymbolicDSGE retains FilterResult histories. Dynare's closest public alternative is calib_smoother, which also computes smoothed paths; this table is informative but not an equivalent workload."
    )
    header = f"{'runtime':<39} {'median':>12} {'speedup':>10} {'max |delta updated|':>20} {'max |delta filtered|':>21}"
    print(header)
    print("-" * len(header))
    print(
        f"{'SymbolicDSGE-filterhistory':<39} {_us(history_median)} {'1.00x':>10} {'0.000e+00':>20} {'0.000e+00':>21}"
    )
    for runtime, report in reports.items():
        median = report["smoother_median_seconds"]
        print(
            f"{'Dynare-' + runtime + '-smoother':<39} {_us(median)} {median / history_median:9.2f}x {report['max_abs_updated']:20.3e} {report['max_abs_filtered']:21.3e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASES), default=tuple(CASES)
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
    parser.add_argument("--octave-bin", default="octave")
    parser.add_argument("--use-joseph", action="store_true")
    args = parser.parse_args()
    if args.periods < 1 or args.warmup < 0 or args.reps < 1:
        parser.error(
            "periods and reps must be positive, and warmup must be nonnegative"
        )
    if set(args.runtimes) - {"native"} and not args.dynare_matlab_path:
        parser.error("--dynare-matlab-path is required for Dynare runtimes")

    output_context = (
        nullcontext(
            (
                args.output_dir
                if args.output_dir.is_absolute()
                else SCRIPT_DIR / args.output_dir
            ).resolve()
        )
        if args.output_dir is not None
        else tempfile.TemporaryDirectory(prefix="symbolicdsge-linear-kf-")
    )
    with output_context as output_path:
        root_output = Path(output_path)
        root_output.mkdir(parents=True, exist_ok=True)
        for case_name in args.cases:
            spec = CASES[case_name]
            output_dir = root_output / case_name
            output_dir.mkdir(exist_ok=True)
            case = _prepare(spec, args.periods)
            native = _time_native(spec, case, args.warmup, args.reps, args.use_joseph)
            _save_native(native, output_dir, _native_model_info(spec, case))
            reports = {
                runtime: _report_dynare(
                    spec,
                    _run_dynare(
                        runtime,
                        spec,
                        output_dir,
                        case.y,
                        args.warmup,
                        args.reps,
                        args.dynare_matlab_path,
                        args.matlab_bin,
                        args.octave_bin,
                    ),
                    case,
                    native,
                )
                for runtime in args.runtimes
                if runtime != "native"
            }
            _print_tables(args, spec, case, native, reports)
            if case_name != args.cases[-1]:
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
