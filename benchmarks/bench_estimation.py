"""Benchmark maximum likelihood and MAP estimation against Dynare.

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
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from SymbolicDSGE.estimation import Estimator, make_prior


@dataclass(frozen=True)
class Routine:
    names: tuple[str, ...]
    theta0: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    probes: tuple[tuple[float, ...], ...]
    priors: tuple["Prior", ...] = ()


@dataclass(frozen=True)
class Prior:
    name: str
    distribution: str
    parameters: tuple[tuple[str, float], ...]
    transform: str
    dynare_density: str


@dataclass(frozen=True)
class OptimizerProfile:
    dynare_mode_compute: int
    native_method: Literal["L-BFGS-B", "Nelder-Mead"]
    native_options: tuple[tuple[str, int | float], ...]
    dynare_optim_options: tuple[tuple[str, int | float], ...]
    maxfun_factor: int | None = None

    def native_kwargs(self, n_params: int) -> dict[str, int | float]:
        options = dict(self.native_options)
        if self.maxfun_factor is not None:
            options["maxfun"] = self.maxfun_factor * n_params
        return options

    def dynare_optim(self) -> str:
        options = list(self.dynare_optim_options)
        if self.maxfun_factor is not None:
            options.append(("MaxFunEvalFactor", self.maxfun_factor))
        return ", ".join(
            (
                f"'{name}', {value:.17g}"
                if isinstance(value, float)
                else f"'{name}', {value}"
            )
            for name, value in options
        )


OPTIMIZER_PROFILES = {
    # csminwel uses a full BFGS inverse-Hessian update. L-BFGS-B has no
    # equivalent initial-Hessian control, but these align the exposed limits,
    # function-progress threshold, and forward finite-difference step.
    4: OptimizerProfile(
        dynare_mode_compute=4,
        native_method="L-BFGS-B",
        native_options=(
            ("maxiter", 1000),
            ("maxfun", 0),
            ("factr", 1e-7 / np.finfo(np.float64).eps),
            ("pgtol", 0.0),
            ("fd_step", 1e-6),
            ("cov", False),
        ),
        dynare_optim_options=(
            ("MaxIter", 1000),
            ("TolFun", 1e-7),
            ("NumgradAlgorithm", 2),
            ("NumgradEpsilon", 1e-6),
        ),
    ),
    # fminsearch uses the host MATLAB or Octave Nelder-Mead implementation.
    7: OptimizerProfile(
        dynare_mode_compute=7,
        native_method="Nelder-Mead",
        native_options=(
            ("maxiter", 6000),
            ("maxfun", 1_000_000),
            ("xatol", 1e-6),
            ("fatol", 1e-8),
            ("cov", False),
        ),
        dynare_optim_options=(
            ("MaxIter", 6000),
            ("MaxFunEvals", 1_000_000),
            ("TolFun", 1e-8),
            ("TolX", 1e-6),
        ),
    ),
    # Dynare's shared Nelder-Mead implementation. Its default evaluation cap
    # is 500 times the parameter count.
    8: OptimizerProfile(
        dynare_mode_compute=8,
        native_method="Nelder-Mead",
        native_options=(
            ("maxiter", 10_000),
            ("xatol", 1e-4),
            ("fatol", 1e-4),
            ("cov", False),
        ),
        dynare_optim_options=(
            ("MaxIter", 10_000),
            ("TolFun", 1e-4),
            ("TolX", 1e-4),
            ("InitialSimplexSize", 0.05),
        ),
        maxfun_factor=500,
    ),
}


ROUTINES: dict[str, dict[str, Routine]] = {
    "ls2004": {
        "mle": Routine(
            ("psi_pi", "rho_r"),
            (2.0, 0.8),
            ((1.0, 5.0), (0.0, 0.99)),
            ((2.0, 0.8), (2.5, 0.75)),
        ),
        "map": Routine(
            ("psi_pi", "rho_r"),
            (2.0, 0.8),
            ((1.0, 5.0), (0.0, 0.99)),
            ((2.0, 0.8), (2.5, 0.75)),
            priors=(
                Prior(
                    name="psi_pi",
                    distribution="normal",
                    parameters=(("mean", 2.0), ("std", 0.5)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
                Prior(
                    name="rho_r",
                    distribution="normal",
                    parameters=(("mean", 0.8), ("std", 0.1)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
            ),
        ),
    },
    "sw2007": {
        "mle": Routine(
            ("crpi", "crr"),
            (1.488, 0.8762),
            ((1.0, 3.0), (0.0, 0.99)),
            ((1.488, 0.8762), (1.7, 0.8)),
        ),
        "map": Routine(
            ("crpi", "crr"),
            (1.488, 0.8762),
            ((1.0, 3.0), (0.0, 0.99)),
            ((1.488, 0.8762), (1.7, 0.8)),
            priors=(
                Prior(
                    name="crpi",
                    distribution="normal",
                    parameters=(("mean", 1.5), ("std", 0.25)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
                Prior(
                    name="crr",
                    distribution="normal",
                    parameters=(("mean", 0.8762), ("std", 0.1)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
            ),
        ),
    },
    "g2015": {
        "mle": Routine(
            ("phi_pi", "rho_a"),
            (1.5, 0.9),
            ((1.01, 3.0), (0.0, 0.99)),
            ((1.5, 0.9), (1.75, 0.8)),
        ),
        "map": Routine(
            ("phi_pi", "rho_a"),
            (1.5, 0.9),
            ((1.01, 3.0), (0.0, 0.99)),
            ((1.5, 0.9), (1.75, 0.8)),
            priors=(
                Prior(
                    name="phi_pi",
                    distribution="normal",
                    parameters=(("mean", 1.5), ("std", 0.25)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
                Prior(
                    name="rho_a",
                    distribution="normal",
                    parameters=(("mean", 0.9), ("std", 0.1)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
            ),
        ),
    },
    "gm2005": {
        "mle": Routine(
            ("phi", "rhoa"),
            (3.0, 0.9),
            ((1.0, 10.0), (0.0, 0.99)),
            ((3.0, 0.9), (2.5, 0.8)),
        ),
        "map": Routine(
            ("phi", "rhoa"),
            (3.0, 0.9),
            ((1.0, 10.0), (0.0, 0.99)),
            ((3.0, 0.9), (2.5, 0.8)),
            priors=(
                Prior(
                    name="phi",
                    distribution="normal",
                    parameters=(("mean", 3.0), ("std", 0.5)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
                Prior(
                    name="rhoa",
                    distribution="normal",
                    parameters=(("mean", 0.9), ("std", 0.1)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
            ),
        ),
    },
    "i2004": {
        "mle": Routine(
            ("rho_a", "rho_e"),
            (0.9048, 0.9907),
            ((0.0, 0.99), (0.0, 0.999)),
            ((0.9048, 0.9907), (0.8, 0.95)),
        ),
        "map": Routine(
            ("rho_a", "rho_e"),
            (0.9048, 0.9907),
            ((0.0, 0.99), (0.0, 0.999)),
            ((0.9048, 0.9907), (0.8, 0.95)),
            priors=(
                Prior(
                    name="rho_a",
                    distribution="normal",
                    parameters=(("mean", 0.9), ("std", 0.05)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
                Prior(
                    name="rho_e",
                    distribution="normal",
                    parameters=(("mean", 0.9907), ("std", 0.05)),
                    transform="identity",
                    dynare_density="normal_pdf",
                ),
            ),
        ),
    },
}
check_case_table("ROUTINES", ROUTINES, total=True)


def _matlab_matrix(values: np.ndarray) -> str:
    return "[" + "; ".join(" ".join(f"{x:.17g}" for x in row) for row in values) + "]"


def _make_estimator(prepared: Prepared, spec: Routine) -> Estimator:
    priors = {
        prior.name: make_prior(
            distribution=prior.distribution,
            parameters=dict(prior.parameters),
            transform=prior.transform,
        )
        for prior in spec.priors
    }
    compiled = prepared.solved.compiled
    return Estimator(
        config=prepared.config,
        compiled=compiled,
        estimated_params=priors or list(spec.names),
        y=prepared.y,
        observables=list(prepared.observable_names),
        filter=LinearKalman(joseph_cov=False, symmetrize=False),
        ss_seed=np.zeros(len(compiled.var_names), dtype=np.float64),
    )


def _native(
    prepared: Prepared,
    spec: Routine,
    routine: str,
    warmup: int,
    reps: int,
    optimizer: OptimizerProfile,
) -> dict:
    estimator = _make_estimator(prepared, spec)
    theta0 = dict(zip(spec.names, spec.theta0))
    bounds = dict(zip(spec.names, spec.bounds))
    optimizer_kwargs = optimizer.native_kwargs(len(spec.names))
    if routine == "mle":
        fn = lambda: estimator.mle(
            theta0=theta0,
            bounds=bounds,
            method=optimizer.native_method,
            **optimizer_kwargs,
        )
    else:
        fn = lambda: estimator.map(
            theta0=theta0,
            bounds=bounds,
            method=optimizer.native_method,
            **optimizer_kwargs,
        )
    with (
        open(os.devnull, "w", encoding="utf-8") as sink,
        contextlib.redirect_stdout(sink),
    ):
        times, result = time_calls(fn, warmup, reps)
    theta = np.array([result.theta[name] for name in spec.names], dtype=np.float64)
    loglik = float(estimator.loglik(result.theta))
    logprior = float(estimator.logprior(result.theta))
    probes = np.asarray(spec.probes, dtype=np.float64).T
    probe_params = [dict(zip(spec.names, probe)) for probe in probes.T]
    objective = estimator.loglik if routine == "mle" else estimator.logpost
    target = np.array([objective(params) for params in probe_params], dtype=np.float64)
    objective_times, _ = time_calls(
        lambda: np.array(
            [objective(params) for params in probe_params], dtype=np.float64
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
        "nit": int(result.nit),
        "success": bool(result.success),
        "probes": probes,
        "probe_target": target,
    }


def _dynare_estimated_params(spec: Routine) -> str:
    priors = {prior.name: prior for prior in spec.priors}
    lines = []
    for name, initial, bounds in zip(spec.names, spec.theta0, spec.bounds, strict=True):
        line = f"{name}, {initial:.17g}, {bounds[0]:.17g}, {bounds[1]:.17g}"
        if prior := priors.get(name):
            line += ", " + prior.dynare_density
            line += ", " + ", ".join(f"{value:.17g}" for _, value in prior.parameters)
        lines.append(line + ";")
    return "\n".join(lines)


def _write_model(
    case: Case,
    spec: Routine,
    path: Path,
    data_file: str,
    periods: int,
    optimizer: OptimizerProfile,
) -> None:
    source = case.mod_path.read_text(encoding="utf-8")
    source += (
        "\nestimated_params;\n" + _dynare_estimated_params(spec) + "\nend;\n"
        f"estimation(datafile = {data_file}, nobs = {periods}, "
        f"mode_compute = {optimizer.dynare_mode_compute}, "
        f"optim = ({optimizer.dynare_optim()}), mh_replic = 0, cova_compute = 0);\n"
    )
    path.write_text(source, encoding="utf-8")


def _run_dynare(
    runtime: str,
    case_name: str,
    spec: Routine,
    routine: str,
    prepared: Prepared,
    probes: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    optimizer: OptimizerProfile,
) -> Path:
    case = CASES[case_name]
    result_path = output_dir / f"dynare_{routine}_{runtime}.mat"
    with tempfile.TemporaryDirectory(
        prefix=f"dynare-{routine}-{runtime}-", dir=output_dir
    ) as tmp:
        workdir = Path(tmp)
        model_name = f"{case_name}_estimation_{routine}"
        data_file = f"{case_name}_data"
        _write_model(
            case,
            spec,
            workdir / f"{model_name}.mod",
            data_file,
            prepared.y.shape[0],
            optimizer,
        )
        write_dynare_data(
            dynare_observable_names(case, prepared.observable_names),
            prepared.y,
            workdir / f"{data_file}.m",
        )
        run_dynare(
            runtime,
            f"bench_estimation_dynare('{quoted_matlab(workdir)}', '{model_name}', "
            f"{_matlab_matrix(probes)}, {int(routine == 'map')}, {args.warmup}, "
            f"{args.reps}, '{quoted_matlab(result_path)}');",
            args,
            f"{case_name} {routine}",
        )
    return result_path


def _print_table(routine: str, native: dict, dynare: dict[str, dict]) -> None:
    native_median = float(np.median(native["times"]))
    print(f"\n{routine.upper()}: implementation-specific optimizer")
    header = f"{'runtime':<26} {'median ms':>12} {'speedup':>10} {'nfev':>8} {'nit':>8} {'max |delta theta|':>20} {'|delta terminal obj|':>20}"
    print(header)
    print("-" * len(header))
    native_nit = "n/a" if native["nit"] < 0 else str(native["nit"])
    print(
        f"{'SymbolicDSGE':<26} {native_median * 1e3:12.2f} {'1.00x':>10} {native['nfev']:8d} {native_nit:>8} {'0.000e+00':>20} {'0.000e+00':>20}"
    )
    for runtime, result in dynare.items():
        median = float(np.median(result["times"]))
        nfev = int(np.median(result["nfev"]))
        nit_value = float(np.median(result["nit"]))
        nit = "n/a" if not np.isfinite(nit_value) else str(int(nit_value))
        delta = float(np.max(np.abs(native["theta"] - result["theta"])))
        objective_delta = abs(native["terminal_target"] - result["terminal_target"])
        print(
            f"{'Dynare-' + runtime:<26} {median * 1e3:12.2f} {median / native_median:9.2f}x {nfev:8d} {nit:>8} {delta:20.3e} {objective_delta:20.3e}"
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
    add_common_args(parser, warmup=1, reps=5)
    parser.add_argument(
        "--routines",
        nargs="+",
        choices=sorted({name for table in ROUTINES.values() for name in table}),
        default=["mle", "map"],
    )
    parser.add_argument(
        "--dynare-mode-compute",
        type=int,
        choices=tuple(OPTIMIZER_PROFILES),
        default=4,
        help=(
            "Paired optimizer profile: 4 is L-BFGS-B/csminwel, "
            "7 and 8 are Nelder-Mead (default: 4)."
        ),
    )
    args = parser.parse_args()
    check_common_args(parser, args)
    optimizer = OPTIMIZER_PROFILES[args.dynare_mode_compute]
    with output_root(args, "symbolicdsge-estimation-") as output_path:
        root_output = Path(output_path)
        root_output.mkdir(parents=True, exist_ok=True)
        for case_index, case_name in enumerate(args.cases):
            case = CASES[case_name]
            routines = ROUTINES[case_name]
            unknown_routines = set(args.routines) - set(routines)
            if unknown_routines:
                parser.error(
                    f"{case_name} does not define: {', '.join(sorted(unknown_routines))}"
                )
            output_dir = root_output / case_name
            output_dir.mkdir(exist_ok=True)
            prepared = prepare(case, args.periods)
            print(
                f"{case.label} estimation: periods={args.periods} warmup={args.warmup} reps={args.reps}"
            )
            print(
                "Setup, preprocessing, input generation, and warmup are outside the timer."
            )
            print(
                f"Optimizer profile: SymbolicDSGE {optimizer.native_method}, "
                f"Dynare mode_compute={optimizer.dynare_mode_compute}."
            )
            for routine in args.routines:
                native = _native(
                    prepared,
                    routines[routine],
                    routine,
                    args.warmup,
                    args.reps,
                    optimizer,
                )
                np.savez(output_dir / f"native_{routine}.npz", **native)
                dynare = {}
                for runtime in (item for item in args.runtimes if item != "native"):
                    raw = loadmat(
                        _run_dynare(
                            runtime,
                            case_name,
                            routines[routine],
                            routine,
                            prepared,
                            native["probes"],
                            output_dir,
                            args,
                            optimizer,
                        ),
                        squeeze_me=True,
                    )
                    dynare[runtime] = {
                        "times": np.asarray(raw["times"], dtype=np.float64).reshape(-1),
                        "nfev": np.asarray(raw["nfev"], dtype=np.int64).reshape(-1),
                        "nit": np.asarray(raw["nit"], dtype=np.float64).reshape(-1),
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
            if case_index != len(args.cases) - 1:
                print()
        if args.output_dir:
            (root_output / "metadata.json").write_text(
                json.dumps(
                    {
                        "cases": args.cases,
                        "periods": args.periods,
                        "warmup": args.warmup,
                        "reps": args.reps,
                        "optimizer": {
                            "dynare_mode_compute": optimizer.dynare_mode_compute,
                            "native_method": optimizer.native_method,
                            "native_options": dict(optimizer.native_options),
                            "native_maxfun_factor": optimizer.maxfun_factor,
                            "dynare_optim": optimizer.dynare_optim(),
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
