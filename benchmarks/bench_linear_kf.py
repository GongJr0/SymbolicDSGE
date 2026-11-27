"""Run first-order models through linear Kalman filters and smoothers.

Three workloads are timed on each side.

Likelihood only, recursion throughput: SymbolicDSGE's ``run_linear`` with
history storage off against Dynare's ``kalman_filter``. Both take the same
model inputs and a stationary ``P0`` computed outside the timer.

Smoother, call to result: SymbolicDSGE's public ``LinearKalman.smooth`` against
Dynare's ``evaluate_smoother``, which is what ``calib_smoother`` runs. Each is
timed from the call to the result in the library's own form, so each side's
own input resolution, ``P0`` solve, and result packaging are inside the timer.
Dynare's entry also rebuilds its dataset and solves the model again.

Smoother, recursion throughput: SymbolicDSGE's ``smooth_linear`` against
Dynare's ``missing_DiffuseKalmanSmootherH1_Z``. Both take precomputed inputs
and ``P0``, and each stores its histories its own way.

Parsing, compilation or preprocessing, solution, data generation, and warmup
are outside every timer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from _cases import (
    CASES,
    Case,
    Prepared,
    add_common_args,
    check_case_table,
    check_common_args,
    dynare_observable_names,
    output_root,
    prepare,
    quoted_matlab,
    run_dynare,
    time_calls,
    write_dynare_data,
)
from SymbolicDSGE import LinearKalman
from SymbolicDSGE.filters.filter import (
    FilterResult,
    SmootherResult,
    run_linear,
    smooth_linear,
)
from SymbolicDSGE.filters.resolvers import LinearRunArgs, _resolve_linear_args

#: Native variable names that Dynare reports under another name.
STATE_MAP: dict[str, tuple[tuple[str, str], ...]] = {
    "sw2007": (
        ("rep_dy", "dy"),
        ("rep_dc", "dc"),
        ("rep_dinve", "dinve"),
        ("rep_dw", "dw"),
    ),
}
check_case_table("STATE_MAP", STATE_MAP, total=False)

#: Smoother fields compared against Dynare, in table order.
_SMOOTHER_FIELDS = ("updated", "smoothed", "P_smoothed", "eps_smoothed")


@dataclass
class KalmanCase:
    prepared: Prepared
    run_args: LinearRunArgs

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(self.prepared.solved.compiled.var_names)

    @property
    def shock_names(self) -> tuple[str, ...]:
        return tuple(self.prepared.solved.compiled.shock_names)


def _prepare_kalman(case: Case, periods: int) -> KalmanCase:
    prepared = prepare(case, periods)
    run_args = _resolve_linear_args(
        prepared.solved,
        prepared.y,
        prepared.observable_names,
        symmetrize=False,
    )
    return KalmanCase(prepared=prepared, run_args=run_args)


def _max_abs(left: np.ndarray | float, right: np.ndarray | float) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _smoother_fields(result: SmootherResult[FilterResult]) -> dict[str, np.ndarray]:
    return {
        "updated": result.filter_result.x_filt,
        "smoothed": result.x_smooth,
        "P_smoothed": result.P_smooth,
        "eps_smoothed": result.eps_smooth,
    }


def _time_native(kalman: KalmanCase, warmup: int, reps: int, use_joseph: bool) -> dict:
    prepared = kalman.prepared
    loglik_args = {**kalman.run_args, "joseph_cov": use_joseph}
    smoother = LinearKalman(symmetrize=False, joseph_cov=False)

    def loglik_only() -> FilterResult:
        return run_linear(**loglik_args, _store_history=False)

    def smoother_call() -> SmootherResult[FilterResult]:
        return smoother.smooth(
            prepared.solved, prepared.y, list(prepared.observable_names)
        )

    def smoother_throughput() -> SmootherResult[FilterResult]:
        return smooth_linear(**kalman.run_args)

    loglik_times, loglik_result = time_calls(loglik_only, warmup, reps)
    call_times, call_result = time_calls(smoother_call, warmup, reps)
    throughput_times, throughput_result = time_calls(smoother_throughput, warmup, reps)
    return {
        "loglik_times": loglik_times,
        "call_times": call_times,
        "throughput_times": throughput_times,
        "loglik": float(loglik_result.loglik),
        **{f"call_{k}": v for k, v in _smoother_fields(call_result).items()},
        **{
            f"throughput_{k}": v for k, v in _smoother_fields(throughput_result).items()
        },
    }


def _native_model_info(kalman: KalmanCase) -> dict[str, int]:
    compiled = kalman.prepared.solved.compiled
    return {
        "declared_variables": compiled.n_var,
        "filter_state_dimension": kalman.run_args["A"].shape[0],
        "predetermined_variables": compiled.n_state,
        "observables": len(kalman.prepared.observable_names),
        "shocks": compiled.n_exog,
        "parameters": compiled.n_par,
    }


def _save_native(result: dict, output_dir: Path, model_info: dict[str, int]) -> None:
    np.savez(output_dir / "native.npz", **result)
    metadata = {
        "runtime": "SymbolicDSGE native",
        "loglik_only_entry_point": "run_linear(_store_history=False), P0 precomputed",
        "smoother_call_entry_point": "LinearKalman.smooth",
        "smoother_throughput_entry_point": "smooth_linear, inputs and P0 precomputed",
        "repetitions": int(result["loglik_times"].size),
        "model_dimensions": model_info,
    }
    (output_dir / "native.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _matlab_cellstr(values: tuple[str, ...]) -> str:
    return "{" + ", ".join(f"'{value}'" for value in values) + "}"


def _run_dynare(
    runtime: str,
    case_name: str,
    kalman: KalmanCase,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    case = CASES[case_name]
    prepared = kalman.prepared
    result_path = output_dir / f"dynare_{runtime}.mat"
    with tempfile.TemporaryDirectory(
        prefix=f"dynare-{runtime}-", dir=output_dir
    ) as temporary_dir:
        workdir = Path(temporary_dir)
        model_file = f"{case.model}.mod"
        data_file = f"{case_name}_data.m"
        shutil.copy2(case.mod_path, workdir / model_file)
        observable_names = dynare_observable_names(case, prepared.observable_names)
        write_dynare_data(observable_names, prepared.y, workdir / data_file)
        run_dynare(
            runtime,
            f"bench_linear_kf_dynare('{quoted_matlab(workdir)}', '{model_file}', "
            f"{_matlab_cellstr(observable_names)}, '{data_file}', {args.warmup}, "
            f"{args.reps}, '{quoted_matlab(result_path)}');",
            args,
            case_name,
        )
    return result_path


#: The .mat keys each Dynare smoother path saves: updated, smoothed, shocks, and
#: smoothed state uncertainty.
_DYNARE_KEYS = {
    "call": ("updated", "smoothed", "smoothed_shocks", "state_uncertainty"),
    "throughput": (
        "direct_updated",
        "direct_smoothed",
        "direct_shocks",
        "direct_state_uncertainty",
    ),
}


def _dynare_fields(
    raw: dict, row: str, states: np.ndarray, shocks: np.ndarray, periods: int
) -> dict[str, np.ndarray]:
    """One Dynare smoother path's outputs, cut to the native states and shocks."""
    updated, smoothed, shocks_key, uncertainty = _DYNARE_KEYS[row]
    n_endo = int(np.asarray(raw["filter_state_nbr"]))
    P = np.moveaxis(
        np.asarray(raw[uncertainty], dtype=np.float64).reshape(n_endo, n_endo, periods),
        2,
        0,
    )
    return {
        "updated": np.asarray(raw[updated], dtype=np.float64)[:, states],
        "smoothed": np.asarray(raw[smoothed], dtype=np.float64)[:, states],
        "P_smoothed": P[:, states][:, :, states],
        "eps_smoothed": np.asarray(raw[shocks_key], dtype=np.float64).reshape(
            periods, -1
        )[:, shocks],
    }


def _report_dynare(
    case_name: str, result_path: Path, kalman: KalmanCase, native: dict
) -> dict:
    raw = loadmat(result_path, squeeze_me=True)
    periods = kalman.prepared.y.shape[0]
    dynare_states = np.atleast_1d(raw["state_names"]).astype(str).reshape(-1)
    dynare_shocks = np.atleast_1d(raw["exo_names"]).astype(str).reshape(-1)
    state_index = {name: position for position, name in enumerate(dynare_states)}
    shock_index = {name: position for position, name in enumerate(dynare_shocks)}
    state_map = dict(STATE_MAP.get(case_name, ()))
    states = np.asarray(
        [state_index[state_map.get(name, name)] for name in kalman.state_names],
        dtype=np.intp,
    )
    shocks = np.asarray(
        [shock_index[name] for name in kalman.shock_names], dtype=np.intp
    )
    deltas = {}
    for row in _DYNARE_KEYS:
        dynare = _dynare_fields(raw, row, states, shocks, periods)
        deltas[row] = {
            field: _max_abs(native[f"{row}_{field}"], dynare[field])
            for field in _SMOOTHER_FIELDS
        }
    return {
        "likelihood_median_seconds": float(
            np.median(np.asarray(raw["likelihood_times"], dtype=np.float64))
        ),
        "call_median_seconds": float(
            np.median(np.asarray(raw["call_times"], dtype=np.float64))
        ),
        "throughput_median_seconds": float(
            np.median(np.asarray(raw["throughput_times"], dtype=np.float64))
        ),
        "max_abs_loglik": _max_abs(native["loglik"], float(np.asarray(raw["loglik"]))),
        "deltas": deltas,
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


def _print_smoother_table(
    title: str, row: str, native: dict, reports: dict[str, dict]
) -> None:
    native_median = float(np.median(native[f"{row}_times"]))
    print(f"\n{title}:")
    header = f"{'runtime':<34} {'median':>12} {'speedup':>10}" + "".join(
        f" {'max |delta ' + field + '|':>24}" for field in _SMOOTHER_FIELDS
    )
    print(header)
    print("-" * len(header))
    zeros = "".join(f" {'0.000e+00':>24}" for _ in _SMOOTHER_FIELDS)
    print(f"{'SymbolicDSGE':<34} {_us(native_median)} {'1.00x':>10}{zeros}")
    for runtime, report in reports.items():
        median = report[f"{row}_median_seconds"]
        deltas = "".join(
            f" {report['deltas'][row][field]:24.3e}" for field in _SMOOTHER_FIELDS
        )
        print(
            f"{'Dynare-' + runtime:<34} {_us(median)} {median / native_median:9.2f}x{deltas}"
        )


def _print_tables(
    args: argparse.Namespace,
    case: Case,
    kalman: KalmanCase,
    native: dict,
    reports: dict[str, dict],
) -> None:
    loglik_median = float(np.median(native["loglik_times"]))
    print(
        f"{case.label} linear KF: periods={args.periods} warmup={args.warmup} reps={args.reps}\n"
        "Setup is outside the timer: parse, compile or preprocess, solve, input generation, and warmup. Times are microseconds per call.\n"
    )
    native_info = _native_model_info(kalman)
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
        f"{'runtime':<34} {'median':>12} {'speedup':>10} {'max |delta loglik|':>20}"
    )
    print("Loglikelihood only, recursion throughput:")
    print(header)
    print("-" * len(header))
    print(f"{'SymbolicDSGE':<34} {_us(loglik_median)} {'1.00x':>10} {'0.000e+00':>20}")
    for runtime, report in reports.items():
        median = report["likelihood_median_seconds"]
        print(
            f"{'Dynare-' + runtime:<34} {_us(median)} {median / loglik_median:9.2f}x {report['max_abs_loglik']:20.3e}"
        )

    _print_smoother_table("Smoother, call to result", "call", native, reports)
    _print_smoother_table(
        "Smoother, recursion throughput", "throughput", native, reports
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, warmup=3, reps=30)
    parser.add_argument("--use-joseph", action="store_true")
    args = parser.parse_args()
    check_common_args(parser, args)

    with output_root(args, "symbolicdsge-linear-kf-") as output_path:
        root_output = Path(output_path)
        root_output.mkdir(parents=True, exist_ok=True)
        for case_name in args.cases:
            case = CASES[case_name]
            output_dir = root_output / case_name
            output_dir.mkdir(exist_ok=True)
            kalman = _prepare_kalman(case, args.periods)
            native = _time_native(kalman, args.warmup, args.reps, args.use_joseph)
            _save_native(native, output_dir, _native_model_info(kalman))
            reports = {
                runtime: _report_dynare(
                    case_name,
                    _run_dynare(runtime, case_name, kalman, output_dir, args),
                    kalman,
                    native,
                )
                for runtime in args.runtimes
                if runtime != "native"
            }
            _print_tables(args, case, kalman, native, reports)
            if case_name != args.cases[-1]:
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
