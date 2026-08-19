"""Benchmark POST82 random-walk MCMC against Dynare.

The timed entries are the public ``Estimator.mcmc`` method and Dynare's
``dynare_estimation`` sampling phase. Dynare's posterior-mode and Hessian
setup, native parse/compile/solve, data generation, and optional warmup chains
are outside the timer. Both samplers use adaptive random-walk MH, but their
RNGs and adaptation schedules differ, so this reports marginal posterior
summaries rather than path parity.

Raw outputs are discarded unless ``--output-dir`` is supplied. Relative output
paths are resolved from this benchmark's directory.
"""

from __future__ import annotations

import argparse
import contextlib
from contextlib import nullcontext
import json
import os
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

from SymbolicDSGE import DSGESolver, ModelParser, Shock
from SymbolicDSGE.estimation import Estimator, make_prior


@dataclass(frozen=True)
class Prior:
    name: str
    distribution: str
    parameters: tuple[tuple[str, float], ...]
    transform: str
    dynare_density: str


@dataclass(frozen=True)
class CaseSpec:
    label: str
    yaml_name: str
    mod_name: str
    data_file: str
    theta0: tuple[float, ...]
    priors: tuple[Prior, ...]
    estimated_params: tuple[str, ...] = ()
    seed: int = 0
    native_to_dynare_observable: tuple[tuple[str, str], ...] = ()
    dynare_remove: tuple[str, ...] = ()


CASES = {
    "ls2004": CaseSpec(
        label="Lubik-Schorfheide 2004",
        yaml_name="POST82.yaml",
        mod_name="post82_kf.mod",
        data_file="post82_mcmc_data",
        theta0=(2.0,),
        priors=(
            Prior(
                name="psi_pi",
                distribution="normal",
                parameters=(("mean", 2.0), ("std", 0.5)),
                transform="identity",
                dynare_density="normal_pdf",
            ),
        ),
        dynare_remove=(
            "calib_smoother(datafile = post82_kf_data, filtered_vars, filter_step_ahead = [1]);",
        ),
    ),
    "sw2007": CaseSpec(
        label="SW2007",
        yaml_name="sw2007.yaml",
        mod_name="sw2007.mod",
        data_file="sw2007_mcmc_data",
        theta0=(1.488,),
        priors=(
            Prior(
                name="crpi",
                distribution="normal",
                parameters=(("mean", 1.5), ("std", 0.25)),
                transform="identity",
                dynare_density="normal_pdf",
            ),
        ),
        seed=2007,
        dynare_remove=(
            "calib_smoother(datafile = sw2007_kf_data, filtered_vars, filter_step_ahead = [1]);",
        ),
    ),
    "g2015": CaseSpec(
        label="Gali 2015",
        yaml_name="gali_2015.yaml",
        mod_name="gali_2015.mod",
        data_file="gali_2015_mcmc_data",
        theta0=(1.5,),
        priors=(
            Prior(
                name="phi_pi",
                distribution="normal",
                parameters=(("mean", 1.5), ("std", 0.25)),
                transform="identity",
                dynare_density="normal_pdf",
            ),
        ),
        seed=2015,
        native_to_dynare_observable=(
            ("obs_pi_ann", "pi_ann"),
            ("obs_i_ann", "i_ann"),
        ),
    ),
    "gm2005": CaseSpec(
        label="Gali-Monacelli 2005",
        yaml_name="gali_monacelli_2005.yaml",
        mod_name="gali_monacelli_2005.mod",
        data_file="gali_monacelli_2005_mcmc_data",
        theta0=(3.0,),
        priors=(
            Prior(
                name="phi",
                distribution="normal",
                parameters=(("mean", 3.0), ("std", 0.5)),
                transform="identity",
                dynare_density="normal_pdf",
            ),
        ),
        seed=2005,
        native_to_dynare_observable=(("obs_pi", "pi"), ("obs_r", "r")),
    ),
    "i2004": CaseSpec(
        label="Ireland 2004",
        yaml_name="ireland_2004.yaml",
        mod_name="ireland_2004.mod",
        data_file="ireland_2004_mcmc_data",
        theta0=(0.9048,),
        priors=(
            Prior(
                name="rho_a",
                distribution="normal",
                parameters=(("mean", 0.9), ("std", 0.05)),
                transform="identity",
                dynare_density="normal_pdf",
            ),
        ),
        seed=2004,
        native_to_dynare_observable=(
            ("obs_gobs", "gobs"),
            ("obs_piobs", "piobs"),
        ),
    ),
}


def _estimated_params(case: CaseSpec) -> tuple[str, ...]:
    if case.estimated_params:
        return case.estimated_params
    return tuple(prior.name for prior in case.priors)


def _dynare_observables(
    case: CaseSpec, native_observables: tuple[str, ...]
) -> tuple[str, ...]:
    names = dict(case.native_to_dynare_observable)
    return tuple(names.get(name, name) for name in native_observables)


def _quoted_matlab(value: Path | str) -> str:
    return str(value).replace("\\", "/").replace("'", "''")


def _prepare(case: CaseSpec, periods: int):
    model, kalman = ModelParser(FIXTURES / case.yaml_name).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    solved = solver.solve(compiled=compiled, order=1)
    sim = solved.sim(
        T=periods,
        shocks={
            ",".join(compiled.shock_names): Shock("norm", multivar=True, seed=case.seed)
        },
        observables=True,
    )
    if sim.y is None:
        raise RuntimeError("Simulation did not return observables.")
    return compiled, solved, sim.y, tuple(sim.observable_names)


def _make_estimator(case: CaseSpec, compiled, solved, y: np.ndarray, observables):
    priors = {
        prior.name: make_prior(
            distribution=prior.distribution,
            parameters=dict(prior.parameters),
            transform=prior.transform,
        )
        for prior in case.priors
    }
    return Estimator(
        solver=solved,
        compiled=compiled,
        y=y,
        observables=list(observables),
        filter_mode="linear",
        estimated_params=list(case.estimated_params) or None,
        priors=priors,
        ss_seed=np.zeros(len(compiled.var_names), dtype=np.float64),
        joseph_cov=False,
        symmetrize=False,
    )


def _run_native(
    case: CaseSpec,
    compiled,
    solved,
    y: np.ndarray,
    observables: tuple[str, ...],
    draws: int,
    burn_in: int,
    warmup: int,
    reps: int,
    seed: int,
    proposal_scale: float,
    adapt: bool,
    adapt_start: int,
) -> dict[str, np.ndarray | float]:
    estimator = _make_estimator(case, compiled, solved, y, observables)
    theta0 = np.asarray(case.theta0, dtype=np.float64)

    def run(chain_seed: int):
        return estimator.mcmc(
            n_draws=draws,
            burn_in=burn_in,
            theta0=theta0,
            random_state=chain_seed,
            proposal_scale=proposal_scale,
            adapt=adapt,
            adapt_start=adapt_start,
        )

    with (
        open(os.devnull, "w", encoding="utf-8") as sink,
        contextlib.redirect_stdout(sink),
    ):
        for i in range(warmup):
            run(seed + i)
        times = np.empty(reps, dtype=np.float64)
        result = None
        for i in range(reps):
            started = time.perf_counter()
            result = run(seed + warmup + i)
            times[i] = time.perf_counter() - started
    assert result is not None
    return {
        "times": times,
        "samples": result.samples,
        "accept_rate": float(result.accept_rate),
    }


def _write_data(path: Path, observables: tuple[str, ...], y: np.ndarray) -> None:
    lines: list[str] = []
    for column, name in enumerate(observables):
        lines.extend((f"{name} = [", *(f"{v:.17g}" for v in y[:, column]), "];"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dynare_estimated_params(case: CaseSpec) -> str:
    priors = {prior.name: prior for prior in case.priors}
    lines = []
    for name, initial in zip(_estimated_params(case), case.theta0, strict=True):
        prior = priors[name]
        line = f"{name}, {initial:.17g}, , , {prior.dynare_density}"
        line += ", " + ", ".join(f"{value:.17g}" for _, value in prior.parameters)
        lines.append(line + ";")
    return "\n".join(lines)


def _write_model(case: CaseSpec, path: Path, periods: int) -> None:
    source = (FIXTURES / case.mod_name).read_text(encoding="utf-8")
    for statement in case.dynare_remove:
        source = source.replace(statement, "")
    source += (
        "\nestimated_params;\n"
        + _dynare_estimated_params(case)
        + "\nend;\n"
        + f"estimation(datafile = {case.data_file}, nobs = {periods}, "
        + "mode_compute = 4, cova_compute = 1, mh_replic = 0);\n"
    )
    path.write_text(source, encoding="utf-8")


def _run_dynare(
    runtime: str,
    case_name: str,
    case: CaseSpec,
    y: np.ndarray,
    observables: tuple[str, ...],
    output_dir: Path,
    draws: int,
    burn_in: int,
    warmup: int,
    reps: int,
    seed: int,
    adapt: bool,
    adapt_start: int,
    dynare_path: str,
    matlab_bin: str,
    octave_bin: str,
) -> Path:
    root = Path(dynare_path).resolve().parent
    platform = "matlab" if runtime == "matlab" else "octave"
    if not (root / "mex" / platform).is_dir():
        raise RuntimeError(f"Dynare at {root} has no {platform} MEX directory.")
    result_path = output_dir / f"dynare_{runtime}.mat"
    with tempfile.TemporaryDirectory(
        prefix=f"dynare-mcmc-{runtime}-", dir=output_dir
    ) as tmp:
        workdir = Path(tmp)
        model_name = f"{case_name}_mcmc"
        _write_model(case, workdir / f"{model_name}.mod", y.shape[0])
        _write_data(
            workdir / f"{case.data_file}.m", _dynare_observables(case, observables), y
        )
        expression = (
            f"addpath('{_quoted_matlab(dynare_path)}'); "
            f"addpath('{_quoted_matlab(SCRIPT_DIR)}', '-begin'); "
            f"bench_mcmc_dynare('{_quoted_matlab(workdir)}', '{model_name}', "
            f"{draws}, {burn_in}, {warmup}, {reps}, {seed}, {int(adapt)}, {adapt_start}, "
            f"'{_quoted_matlab(result_path)}');"
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
                f"Dynare {runtime} MCMC failed:\n{completed.stderr.strip()}"
            )
    return result_path


def _summary(samples: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean(samples, axis=0),
        "std": np.std(samples, axis=0, ddof=1),
        "q05": np.quantile(samples, 0.05, axis=0),
        "q50": np.quantile(samples, 0.50, axis=0),
        "q95": np.quantile(samples, 0.95, axis=0),
    }


def _print_report(
    case: CaseSpec,
    estimated_params: tuple[str, ...],
    draws: int,
    burn_in: int,
    native: dict,
    dynare: dict,
):
    print(
        f"{case.label} MCMC: retained draws={draws} burn-in={burn_in} "
        f"reps={len(native['times'])}"
    )
    print(
        "Mode and Hessian setup are outside Dynare's timer. Both chains use "
        "adaptive random-walk MH from the shared start draw, but their update "
        "schedules and RNGs differ, so posterior summaries are descriptive."
    )
    native_median = float(np.median(native["times"]))
    header = f"{'runtime':<18} {'median s':>12} {'draws / s':>12} {'acceptance':>12}"
    print("\n" + header)
    print("-" * len(header))
    print(
        f"{'SymbolicDSGE':<18} {native_median:12.3f} {draws / native_median:12.1f} {native['accept_rate']:12.3f}"
    )
    for runtime, result in dynare.items():
        median = float(np.median(result["times"]))
        print(
            f"{'Dynare-' + runtime:<18} {median:12.3f} {draws / median:12.1f} {result['accept_rate']:12.3f}"
        )

    native_summary = _summary(native["samples"])
    for runtime, result in dynare.items():
        result["summary"] = _summary(result["samples"])
    print("\nPosterior summaries: max absolute difference from SymbolicDSGE")
    header = f"{'parameter':<16} {'runtime':<18} {'mean':>12} {'std':>12} {'q05':>12} {'q50':>12} {'q95':>12}"
    print(header)
    print("-" * len(header))
    for index, name in enumerate(estimated_params):
        for runtime, result in dynare.items():
            summary = result["summary"]
            deltas = [
                abs(float(native_summary[key][index] - summary[key][index]))
                for key in ("mean", "std", "q05", "q50", "q95")
            ]
            print(
                f"{name:<16} {'Dynare-' + runtime:<18} "
                + " ".join(f"{delta:12.3e}" for delta in deltas)
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=120)
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native-proposal-scale", type=float, default=0.1)
    parser.add_argument("--no-native-adapt", action="store_true")
    parser.add_argument("--adapt-start", type=int, default=100)
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
    args = parser.parse_args()
    if (
        args.periods < 1
        or args.draws < 1
        or args.burn_in < 0
        or args.warmup < 0
        or args.reps < 1
    ):
        parser.error(
            "periods, draws, and reps must be positive, and burn-in and warmup nonnegative"
        )
    if args.native_proposal_scale <= 0:
        parser.error("--native-proposal-scale must be positive")
    if args.adapt_start < 0:
        parser.error("--adapt-start must be nonnegative")
    if set(args.runtimes) - {"native"} and not args.dynare_matlab_path:
        parser.error("--dynare-matlab-path is required for Dynare runtimes")

    context = (
        nullcontext(
            args.output_dir.resolve()
            if args.output_dir and args.output_dir.is_absolute()
            else (SCRIPT_DIR / args.output_dir).resolve()
        )
        if args.output_dir
        else tempfile.TemporaryDirectory(prefix="symbolicdsge-mcmc-")
    )
    with context as root:
        root_output = Path(root)
        root_output.mkdir(parents=True, exist_ok=True)
        for case_name in args.cases:
            case = CASES[case_name]
            estimated_params = _estimated_params(case)
            if len(case.theta0) != len(estimated_params):
                parser.error(
                    f"{case_name} has {len(estimated_params)} estimated parameters but "
                    f"{len(case.theta0)} initial values"
                )
            output_dir = root_output / case_name
            output_dir.mkdir(exist_ok=True)
            compiled, solved, y, observables = _prepare(case, args.periods)
            native = _run_native(
                case,
                compiled,
                solved,
                y,
                observables,
                args.draws,
                args.burn_in,
                args.warmup,
                args.reps,
                args.seed,
                args.native_proposal_scale,
                not args.no_native_adapt,
                args.adapt_start,
            )
            np.savez(output_dir / "native.npz", **native)
            dynare: dict[str, dict] = {}
            for runtime in (item for item in args.runtimes if item != "native"):
                raw = loadmat(
                    _run_dynare(
                        runtime,
                        case_name,
                        case,
                        y,
                        observables,
                        output_dir,
                        args.draws,
                        args.burn_in,
                        args.warmup,
                        args.reps,
                        args.seed,
                        not args.no_native_adapt,
                        args.adapt_start,
                        args.dynare_matlab_path,
                        args.matlab_bin,
                        args.octave_bin,
                    ),
                    squeeze_me=False,
                )
                samples = np.asarray(raw["samples"], dtype=np.float64).reshape(
                    -1, len(estimated_params)
                )
                dynare[runtime] = {
                    "times": np.asarray(raw["times"], dtype=np.float64).reshape(-1),
                    "samples": samples[-args.draws :],
                    "accept_rate": float(np.asarray(raw["accept_rate"]).squeeze()),
                }
            _print_report(
                case,
                estimated_params,
                args.draws,
                args.burn_in,
                native,
                dynare,
            )
        if args.output_dir:
            (root_output / "metadata.json").write_text(
                json.dumps(
                    {
                        "cases": args.cases,
                        "periods": args.periods,
                        "draws": args.draws,
                        "burn_in": args.burn_in,
                        "warmup": args.warmup,
                        "reps": args.reps,
                        "seed": args.seed,
                        "native_proposal_scale": args.native_proposal_scale,
                        "native_adapt": not args.no_native_adapt,
                        "adapt_start": args.adapt_start,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
