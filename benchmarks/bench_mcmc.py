"""Benchmark random-walk MCMC against Dynare.

The timed entries are the public ``Estimator.mcmc`` method and Dynare's
``dynare_estimation`` sampling phase. Dynare's posterior-mode setup, native
parse/compile/solve, the shared mode and proposal covariance, data generation,
and optional warmup chains are outside the timer. Both samplers run fixed
proposal random-walk MH with a proposal standard deviation of
``2.38 / sqrt(d) * --proposal-scale`` on the same covariance. Only their RNGs
differ, so this reports marginal posterior summaries rather than path parity.

The MAP is found once per case, before either timer starts, and both runtimes
begin every chain from it. The native side takes it as ``theta0`` under
``compute_map=False``; Dynare reads it from the mode file, whose ``xparam1`` the
setup pass's own estimate is overwritten with.

That same MAP call returns the covariance at the mode, and both runtimes sample
with it, so neither timed call builds a Hessian and ``draws / s`` covers
sampling alone. The native side takes it as ``proposal_cov``; Dynare reads its
inverse off the mode file as ``hh``, which ``dynare_estimation_1`` inverts back
into the jumping covariance. The setup pass runs under ``cova_compute = 0``, so
Dynare never builds one either.

Every case estimates its model's full structural parameter set: each calibrated
parameter that is not a shock standard deviation, a shock correlation, or a
measurement-error entry of R. The priors come from the calibration itself, one
normal per parameter centered on the calibrated value with a standard deviation
of ``--prior-scale`` times its magnitude.

The .mod files reach the same set from the other side. A coefficient the .mod
computes as a model-local is a derived calibration entry in the yaml, so both
runtimes rebuild it from the same base parameters and a draw moves the same
coefficients on either side. The temporary copy of the .mod only appends the
estimation block, and nothing is written back to the fixture.

Raw outputs are discarded unless ``--output-dir`` is supplied. Relative output
paths are resolved from this benchmark's directory.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from _cases import (
    CASES,
    Prepared,
    add_common_args,
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
from SymbolicDSGE._ckernels.estimation import ess_rhat, mcse_quantile
from SymbolicDSGE.estimation import Estimator, make_prior

PRIOR_DISTRIBUTION = "normal"
PRIOR_TRANSFORM = "identity"
DYNARE_PRIOR_DENSITY = "normal_pdf"
QUANTILES = (0.05, 0.50, 0.95)

# The shared proposal covariance comes from a finite-difference Hessian at the
# mode, and this scales its step. The shipped default is too wide for sw2007,
# whose crhoa sits 0.0023 below the unit root: the upward probe lands at
# 1.00003, and the Blanchard-Kahn failure there leaves the covariance NaN
# throughout. Narrowing it keeps the probe inside the stable region.
HESSIAN_STEP_SCALE = 0.5


@dataclass(frozen=True)
class Targets:
    """The estimated parameters, their calibrated values, and prior widths."""

    names: tuple[str, ...]
    values: tuple[float, ...]
    stds: tuple[float, ...]


def _estimation_targets(prepared: Prepared, prior_scale: float) -> Targets:
    """Every structural parameter of the model, with a prior built from its
    calibrated value.

    Shock and measurement-noise covariance parameters are left out: they carry
    a role-authoritative constraining transform on this side and reach Dynare
    through stderr and corr entries rather than by name, so estimating them
    would compare two different parameterizations. A parameter calibrated at
    zero takes ``prior_scale`` itself as its prior standard deviation, since
    scaling would leave it with no width at all.
    """
    calibration = prepared.config.calibration
    excluded = calibration.shocks.param_names
    if calibration.measurement_noise is not None:
        excluded |= calibration.measurement_noise.param_names
    names = tuple(
        name for name in prepared.solved.compiled.calib_params if name not in excluded
    )
    values = tuple(float(calibration.parameters[name]) for name in names)
    return Targets(
        names=names,
        values=values,
        stds=tuple(
            prior_scale * abs(value) if value != 0.0 else prior_scale
            for value in values
        ),
    )


def _make_estimator(targets: Targets, prepared: Prepared) -> Estimator:
    priors = {
        name: make_prior(
            distribution=PRIOR_DISTRIBUTION,
            parameters={"mean": mean, "std": std},
            transform=PRIOR_TRANSFORM,
        )
        for name, mean, std in zip(
            targets.names, targets.values, targets.stds, strict=True
        )
    }
    compiled = prepared.solved.compiled
    return Estimator(
        config=prepared.config,
        compiled=compiled,
        estimated_params=priors,
        y=prepared.y,
        observables=list(prepared.observable_names),
        filter=LinearKalman(joseph_cov=False, symmetrize=False),
        ss_seed=np.zeros(len(compiled.var_names), dtype=np.float64),
    )


def _posterior_mode(targets: Targets, prepared: Prepared, hessian_step_scale: float):
    """The MAP both runtimes start from and the covariance both propose with,
    found once and outside every timer.

    Letting each side find its own would compare two chains that began in
    different places, and letting the native side find it per rep would time
    the same optimization once per rep against a Dynare that reads its own from
    a file.
    """
    estimator = _make_estimator(targets, prepared)
    with (
        open(os.devnull, "w", encoding="utf-8") as sink,
        contextlib.redirect_stdout(sink),
    ):
        result = estimator.map(
            theta0=dict(zip(targets.names, targets.values, strict=True)),
            cov_fd_step_scale=hessian_step_scale,
        )
    if not result.success:
        raise RuntimeError(f"MAP did not converge before sampling: {result.message}")
    if result.vcov is None or not np.all(np.isfinite(result.vcov)):
        raise RuntimeError(
            "The MAP found no usable proposal covariance at the mode "
            f"(cov_status={result.cov_status})."
        )
    return result


def _proposal_covariance(targets: Targets, mode) -> np.ndarray:
    """``mode.vcov`` reordered to follow ``targets.names``.

    It describes ``mode.x``, so its rows and columns follow the estimator's own
    parameter order, while both runtimes take the proposal by name.
    """
    order = [list(mode.theta).index(name) for name in targets.names]
    return np.asarray(mode.vcov, dtype=np.float64)[np.ix_(order, order)]


def _run_native(
    targets: Targets, prepared: Prepared, mode, args: argparse.Namespace
) -> dict[str, np.ndarray | float]:
    estimator = _make_estimator(targets, prepared)
    proposal_cov = _proposal_covariance(targets, mode)
    seeds = itertools.count(args.seed)

    def run():
        return estimator.mcmc(
            n_draws=args.draws,
            burn_in=args.burn_in,
            theta0=mode.theta,
            compute_map=False,
            random_state=next(seeds),
            proposal_scale=args.proposal_scale,
            adapt=False,
            proposal_cov=proposal_cov,
        )

    with (
        open(os.devnull, "w", encoding="utf-8") as sink,
        contextlib.redirect_stdout(sink),
    ):
        times, result = time_calls(run, args.warmup, args.reps)
    order = [result.param_names.index(name) for name in targets.names]
    return {
        "times": times,
        "samples": result.samples[0][:, order],
        "accept_rate": float(result.accept_rate[0]),
    }


def _write_shared_mode(path: Path, targets: Targets, mode) -> None:
    """The shared MAP and proposal covariance, written for the MATLAB side to
    match up by name.

    Dynare's own parameter order is whatever its mode file records, so the names
    travel with the values rather than the two sides agreeing on an ordering.
    """
    savemat(
        str(path),
        {
            "xparam1": np.asarray(
                [float(mode.theta[name]) for name in targets.names], dtype=np.float64
            ).reshape(-1, 1),
            "proposal_cov": _proposal_covariance(targets, mode),
            "parameter_names": np.array(
                [np.str_(name) for name in targets.names], dtype=object
            ).reshape(-1, 1),
        },
    )


def _dynare_estimated_params(targets: Targets) -> str:
    return "\n".join(
        f"{name}, {value:.17g}, , , {DYNARE_PRIOR_DENSITY}, {value:.17g}, {std:.17g};"
        for name, value, std in zip(
            targets.names, targets.values, targets.stds, strict=True
        )
    )


def _write_model(
    case_name: str,
    path: Path,
    data_file: str,
    periods: int,
    targets: Targets,
    proposal_scale: float,
) -> None:
    jscale = 2.38 / np.sqrt(len(targets.names)) * proposal_scale
    source = CASES[case_name].mod_path.read_text(encoding="utf-8")
    source += (
        "\nestimated_params;\n"
        + _dynare_estimated_params(targets)
        + "\nend;\n"
        + f"estimation(datafile = {data_file}, nobs = {periods}, "
        + "mode_compute = 4, cova_compute = 0, mh_replic = 0, "
        + f"mh_jscale = {jscale:.17g});\n"
    )
    path.write_text(source, encoding="utf-8")


def _run_dynare(
    runtime: str,
    case_name: str,
    targets: Targets,
    prepared: Prepared,
    mode,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    result_path = output_dir / f"dynare_{runtime}.mat"
    with tempfile.TemporaryDirectory(
        prefix=f"dynare-mcmc-{runtime}-", dir=output_dir
    ) as tmp:
        workdir = Path(tmp)
        model_name = f"{case_name}_mcmc"
        data_file = f"{case_name}_data"
        _write_model(
            case_name,
            workdir / f"{model_name}.mod",
            data_file,
            prepared.y.shape[0],
            targets,
            args.proposal_scale,
        )
        write_dynare_data(
            dynare_observable_names(CASES[case_name], prepared.observable_names),
            prepared.y,
            workdir / f"{data_file}.m",
        )
        shared_mode_path = workdir / "shared_mode.mat"
        _write_shared_mode(shared_mode_path, targets, mode)
        run_dynare(
            runtime,
            f"bench_mcmc_dynare('{quoted_matlab(workdir)}', '{model_name}', "
            f"{args.draws}, {args.burn_in}, {args.warmup}, {args.reps}, "
            f"{args.seed}, '{quoted_matlab(shared_mode_path)}', "
            f"'{quoted_matlab(result_path)}');",
            args,
            f"{case_name} MCMC",
        )
    return result_path


def _scores(samples: np.ndarray) -> dict[str, np.ndarray]:
    """ESS and the MCSEs of the mean and of each of ``QUANTILES`` for one chain,
    masked and clipped the way ``MCMCResult`` reports them.
    """
    chain = samples[None]
    ess, _, status = ess_rhat(chain)
    n_draws = samples.shape[0]
    ess = np.clip(np.where(status == 0, ess, np.nan), 0.0, n_draws * np.log10(n_draws))
    scores = {
        "ess": ess,
        "mcse": np.std(samples, axis=0, ddof=1) / np.sqrt(ess),
    }
    for q in QUANTILES:
        scores[f"q{round(q * 100):02d}_mcse"] = mcse_quantile(chain, q)
    return scores


def _summary(samples: np.ndarray) -> dict[str, np.ndarray]:
    quantile_values = np.quantile(samples, QUANTILES, axis=0)
    return {
        "std": np.std(samples, axis=0, ddof=1),
        "mean": np.mean(samples, axis=0),
        **{
            f"q{round(q * 100):02d}": values
            for q, values in zip(QUANTILES, quantile_values, strict=True)
        },
        **_scores(samples),
    }


def _normalized_delta(
    native_values: np.ndarray,
    runtime_values: np.ndarray,
    native_mcse: np.ndarray,
    runtime_mcse: np.ndarray,
) -> np.ndarray:
    combined_mcse = np.hypot(native_mcse, runtime_mcse)
    normalized = np.full(combined_mcse.shape, np.nan, dtype=np.float64)
    valid = combined_mcse > 0.0
    normalized[valid] = (
        np.abs(native_values[valid] - runtime_values[valid]) / combined_mcse[valid]
    )
    return normalized


def _print_report(
    label: str,
    estimated_params: tuple[str, ...],
    draws: int,
    burn_in: int,
    native: dict,
    dynare: dict,
):
    print(
        f"{label} MCMC: {len(estimated_params)} estimated parameters, "
        f"retained draws={draws} burn-in={burn_in} reps={len(native['times'])}"
    )
    print(
        "Both runtimes start from one MAP and propose from the covariance found "
        "at it, both outside every timer, so neither timed call builds a "
        "Hessian. Only their RNGs differ."
    )
    native_summary = _summary(native["samples"])
    summaries = {
        runtime: _summary(result["samples"]) for runtime, result in dynare.items()
    }
    header = (
        f"{'runtime':<18} {'median s':>12} {'draws / s':>12} "
        f"{'acceptance':>12} {'min ESS':>12} {'min ESS / s':>14}"
    )
    print("\n" + header)
    print("-" * len(header))
    rows = [("SymbolicDSGE", native, native_summary)] + [
        ("Dynare-" + runtime, result, summaries[runtime])
        for runtime, result in dynare.items()
    ]
    for name, result, summary in rows:
        median = float(np.median(result["times"]))
        min_ess = float(np.min(summary["ess"]))
        print(
            f"{name:<18} {median:12.3f} {draws / median:12.1f} "
            f"{result['accept_rate']:12.3f} {min_ess:12.1f} {min_ess / median:14.1f}"
        )

    print(
        "\nPosterior comparisons: std is absolute; mean and quantiles are "
        "|delta| / combined MCSE"
    )
    header = (
        f"{'parameter':<16} {'runtime':<18} {'mean / MCSE':>14} {'std delta':>12} "
        f"{'q05 / MCSE':>14} {'q50 / MCSE':>14} {'q95 / MCSE':>14} "
        f"{'ESS native':>12} {'ESS runtime':>12}"
    )
    print(header)
    print("-" * len(header))
    for index, name in enumerate(estimated_params):
        for runtime, summary in summaries.items():
            normalized = {
                key: _normalized_delta(
                    native_summary[key],
                    summary[key],
                    native_summary["mcse" if key == "mean" else f"{key}_mcse"],
                    summary["mcse" if key == "mean" else f"{key}_mcse"],
                )
                for key in ("mean", "q05", "q50", "q95")
            }
            std_delta = abs(float(native_summary["std"][index] - summary["std"][index]))
            print(
                f"{name:<16} {'Dynare-' + runtime:<18} "
                + f"{normalized['mean'][index]:14.3f} {std_delta:12.3e} "
                + " ".join(
                    f"{normalized[key][index]:14.3f}" for key in ("q05", "q50", "q95")
                )
                + f" {native_summary['ess'][index]:12.1f} {summary['ess'][index]:12.1f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, warmup=0, reps=1)
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proposal-scale", type=float, default=1)
    parser.add_argument("--prior-scale", type=float, default=0.05)
    parser.add_argument("--hessian-step-scale", type=float, default=HESSIAN_STEP_SCALE)
    args = parser.parse_args()
    check_common_args(parser, args)
    if args.draws < 1 or args.burn_in < 0:
        parser.error("draws must be positive, and burn-in nonnegative")
    if args.proposal_scale <= 0:
        parser.error("--proposal-scale must be positive")
    if args.prior_scale <= 0:
        parser.error("--prior-scale must be positive")
    if args.hessian_step_scale <= 0:
        parser.error("--hessian-step-scale must be positive")

    with output_root(args, "symbolicdsge-mcmc-") as output_path:
        root_output = Path(output_path)
        root_output.mkdir(parents=True, exist_ok=True)
        for case_name in args.cases:
            output_dir = root_output / case_name
            output_dir.mkdir(exist_ok=True)
            prepared = prepare(CASES[case_name], args.periods)
            targets = _estimation_targets(prepared, args.prior_scale)
            mode = _posterior_mode(targets, prepared, args.hessian_step_scale)
            native = _run_native(targets, prepared, mode, args)
            np.savez(output_dir / "native.npz", **native)
            dynare: dict[str, dict] = {}
            for runtime in (item for item in args.runtimes if item != "native"):
                raw = loadmat(
                    _run_dynare(
                        runtime, case_name, targets, prepared, mode, output_dir, args
                    ),
                    squeeze_me=False,
                )
                samples = np.asarray(raw["samples"], dtype=np.float64).reshape(
                    -1, len(targets.names)
                )
                dynare[runtime] = {
                    "times": np.asarray(raw["times"], dtype=np.float64).reshape(-1),
                    "samples": samples[-args.draws :],
                    "accept_rate": float(np.asarray(raw["accept_rate"]).squeeze()),
                }
            _print_report(
                CASES[case_name].label,
                targets.names,
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
                        "proposal_scale": args.proposal_scale,
                        "prior_scale": args.prior_scale,
                        "hessian_step_scale": args.hessian_step_scale,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
