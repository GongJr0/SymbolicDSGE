"""Benchmark random-walk MCMC against Dynare.

The timed entries are the public ``Estimator.mcmc`` method and Dynare's
``dynare_estimation`` sampling phase. Dynare's posterior-mode and Hessian
setup, native parse/compile/solve, data generation, and optional warmup chains
are outside the timer. Both samplers use adaptive random-walk MH, but their
RNGs and adaptation schedules differ, so this reports marginal posterior
summaries rather than path parity.

The two timers do not cover the same work. ``Estimator.mcmc`` finds its own MAP
and builds the finite-difference Hessian its proposal starts from inside the
timed call, where Dynare arrives with both already computed. The Hessian alone
costs ``d * (d + 1)`` likelihood evaluations, negligible at one estimated
parameter and comparable to the whole sampling run at sw2007's 43, so read
``draws / s`` as the cost of a chain plus its setup on this side and a chain
alone on Dynare's.

Every case estimates its model's full structural parameter set: each calibrated
parameter that is not a shock standard deviation, a shock correlation, or a
measurement-error entry of R. The priors come from the calibration itself, one
normal per parameter centered on the calibrated value with a standard deviation
of ``--prior-scale`` times its magnitude.

The .mod files reach the same set from the other side. Where a yaml carries a
free parameter that the .mod computes as a model-local, the temporary copy of
the .mod drops that local and declares the name as a parameter, so a draw moves
the same coefficients in both runtimes. Nothing is written back to the fixture,
and the copy runs no statement the unpatched file would not.

Raw outputs are discarded unless ``--output-dir`` is supplied. Relative output
paths are resolved from this benchmark's directory.
"""

from __future__ import annotations

import argparse
import contextlib
from contextlib import nullcontext
import json
import os
import re
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
MODEL_STATEMENT = "model(linear);"
sys.path.insert(0, str(ROOT))

from SymbolicDSGE import DSGESolver, ModelParser, Shock
from SymbolicDSGE.estimation import Estimator, make_prior

PRIOR_DISTRIBUTION = "normal"
PRIOR_TRANSFORM = "identity"
DYNARE_PRIOR_DENSITY = "normal_pdf"

# Both samplers draw their initial proposal from a finite-difference Hessian at
# the mode, and they already agree on how wide that step is: Dynare's
# hessian.m steps by eps^(1/6) * gstep(2) with a floor of sqrt(gstep(1)), and
# the native kernel steps by sqrt(cbrt(eps)) * scale with a floor of 0.1, which
# is the same 2.4607e-3 and the same 0.1 under the shipped defaults.
#
# The default is too wide for sw2007, whose crhoa sits 0.0023 below the unit
# root: the upward probe lands at 1.00003, and the Blanchard-Kahn failure there
# is fatal to both runtimes. Native aborts the chain outright; Dynare fills hh
# with non-finite entries, which inv() spreads into a jumping covariance that
# chol rejects. Narrowing it keeps the probe inside the stable region. One
# scale drives both sides so the two proposals stay comparable.
HESSIAN_STEP_SCALE = 0.5


@dataclass(frozen=True)
class CaseSpec:
    label: str
    yaml_name: str
    mod_name: str
    data_file: str
    seed: int = 0
    native_to_dynare_observable: tuple[tuple[str, str], ...] = ()
    dynare_remove: tuple[str, ...] = ()
    dynare_replace: tuple[tuple[str, str], ...] = ()
    # Model-locals to delete from the .mod copy. The ones the yaml calibrates
    # are redeclared as parameters; the rest only fed those, so they go too.
    dynare_locals: tuple[str, ...] = ()


@dataclass(frozen=True)
class Targets:
    """The estimated parameters, their calibrated values, and prior widths."""

    names: tuple[str, ...]
    values: tuple[float, ...]
    stds: tuple[float, ...]

    def value_of(self, name: str) -> float:
        return self.values[self.names.index(name)]


CASES = {
    "ls2004": CaseSpec(
        label="Lubik-Schorfheide 2004",
        yaml_name="POST82.yaml",
        mod_name="post82_kf.mod",
        data_file="post82_mcmc_data",
        dynare_remove=(
            "calib_smoother(datafile = post82_kf_data, filtered_vars, filter_step_ahead = [1]);",
        ),
    ),
    "sw2007": CaseSpec(
        label="SW2007",
        yaml_name="sw2007.yaml",
        mod_name="sw2007.mod",
        data_file="sw2007_mcmc_data",
        seed=2007,
        dynare_remove=(
            "calib_smoother(datafile = sw2007_kf_data, filtered_vars, filter_step_ahead = [1]);",
        ),
        # conster is a free parameter once its model-local is gone, so the
        # steady state of robs reads it rather than rebuilding it from the
        # discount, growth and inflation parameters.
        dynare_replace=(
            (
                "robs = (((1+constepinf/100)/((1/(1+constebeta/100))"
                "*(1+ctrend/100)^(-csigma)))-1)*100;",
                "robs = conster;",
            ),
        ),
        dynare_locals=(
            "cpie",
            "cgamma",
            "cbeta",
            "clandap",
            "cbetabar",
            "cr",
            "crk",
            "cw",
            "cikbar",
            "cik",
            "clk",
            "cky",
            "ciy",
            "ccy",
            "crkky",
            "cwhlc",
            "cwly",
            "conster",
        ),
    ),
    "g2015": CaseSpec(
        label="Gali 2015",
        yaml_name="gali_2015.yaml",
        mod_name="gali_2015.mod",
        data_file="gali_2015_mcmc_data",
        seed=2015,
        native_to_dynare_observable=(
            ("obs_pi_ann", "pi_ann"),
            ("obs_i_ann", "i_ann"),
        ),
        dynare_locals=("Omega", "psi_n_ya", "lambda", "kappa"),
    ),
    "gm2005": CaseSpec(
        label="Gali-Monacelli 2005",
        yaml_name="gali_monacelli_2005.yaml",
        mod_name="gali_monacelli_2005.mod",
        data_file="gali_monacelli_2005_mcmc_data",
        seed=2005,
        native_to_dynare_observable=(("obs_pi", "pi"), ("obs_r", "r")),
        # The surviving locals sit at sigma = eta = gamma = 1, where omega,
        # sigma_a, Theta, Gamma and Psi are constants: only kappa_a moves with
        # an estimated parameter, and the yaml calibrates it directly.
        dynare_locals=("lambda", "kappa_a"),
    ),
    "i2004": CaseSpec(
        label="Ireland 2004",
        yaml_name="ireland_2004.yaml",
        mod_name="ireland_2004.mod",
        data_file="ireland_2004_mcmc_data",
        seed=2004,
        native_to_dynare_observable=(
            ("obs_gobs", "gobs"),
            ("obs_piobs", "piobs"),
        ),
    ),
}


def _estimation_targets(compiled, prior_scale: float) -> Targets:
    """Every structural parameter of the model, with a prior built from its
    calibrated value.

    Shock standard deviations, shock correlations and the measurement-error
    entries of R are left out: they carry a role-authoritative constraining
    transform on this side and reach Dynare through stderr and corr entries
    rather than by name, so estimating them would compare two different
    parameterizations. A parameter calibrated at zero takes ``prior_scale``
    itself as its prior standard deviation, since scaling would leave it with
    no width at all.
    """
    calibration = compiled.config.calibration
    excluded = {
        symbol.name
        for mapping in (calibration.shock_std, calibration.shock_corr)
        for symbol in (mapping or {}).values()
        if symbol is not None
    }
    excluded |= set(getattr(compiled.kalman, "R_param_names", None) or ())
    names = tuple(
        str(param) for param in compiled.calib_params if str(param) not in excluded
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


def _make_estimator(targets: Targets, compiled, solved, y: np.ndarray, observables):
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
    return Estimator(
        solver=solved,
        compiled=compiled,
        y=y,
        observables=list(observables),
        filter_mode="linear",
        estimated_params=list(targets.names),
        priors=priors,
        ss_seed=np.zeros(len(compiled.var_names), dtype=np.float64),
        joseph_cov=False,
        symmetrize=False,
    )


def _run_native(
    targets: Targets,
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
    hessian_step_scale: float,
) -> dict[str, np.ndarray | float]:
    estimator = _make_estimator(targets, compiled, solved, y, observables)
    theta0 = dict(zip(targets.names, targets.values, strict=True))

    def run(chain_seed: int):
        return estimator.mcmc(
            n_draws=draws,
            burn_in=burn_in,
            theta0=theta0,
            random_state=chain_seed,
            proposal_scale=proposal_scale,
            adapt=adapt,
            adapt_start=adapt_start,
            hessian_fd_step_scale=hessian_step_scale,
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


def _dynare_estimated_params(targets: Targets) -> str:
    return "\n".join(
        f"{name}, {value:.17g}, , , {DYNARE_PRIOR_DENSITY}, {value:.17g}, {std:.17g};"
        for name, value, std in zip(
            targets.names, targets.values, targets.stds, strict=True
        )
    )


def _promote_dynare_locals(case: CaseSpec, source: str, targets: Targets) -> str:
    """Drop the listed model-locals, declaring the estimated ones as parameters.

    A local the yaml calibrates is a free parameter on the native side, so
    leaving Dynare to rebuild it from other estimated parameters would move a
    coefficient the native draw holds fixed. The locals that only fed those
    definitions go with them.
    """
    declared: list[str] = []
    for name in case.dynare_locals:
        pattern = re.compile(
            rf"^[ \t]*#[ \t]*{re.escape(name)}[ \t]*=[^;]*;.*\n", re.MULTILINE
        )
        source, count = pattern.subn("", source)
        if count != 1:
            raise RuntimeError(
                f"{case.mod_name} defines the model-local '{name}' {count} times."
            )
        if name in targets.names:
            declared.append(name)
    if not declared:
        return source

    block = (
        "parameters "
        + " ".join(declared)
        + ";\n"
        + "\n".join(f"{name} = {targets.value_of(name):.17g};" for name in declared)
        + "\n\n"
    )
    patched = source.replace(MODEL_STATEMENT, block + MODEL_STATEMENT, 1)
    if patched == source:
        raise RuntimeError(
            f"{case.mod_name} has no '{MODEL_STATEMENT}' to declare parameters before."
        )
    return patched


def _write_model(
    case: CaseSpec,
    path: Path,
    periods: int,
    targets: Targets,
    hessian_step_scale: float,
) -> None:
    source = (FIXTURES / case.mod_name).read_text(encoding="utf-8")
    for statement in case.dynare_remove:
        source = source.replace(statement, "")
    for statement, replacement in case.dynare_replace:
        if statement not in source:
            raise RuntimeError(f"{case.mod_name} has no statement '{statement}'.")
        source = source.replace(statement, replacement)
    source = _promote_dynare_locals(case, source, targets)
    source += (
        "\nestimated_params;\n"
        + _dynare_estimated_params(targets)
        + "\nend;\n"
        + f"options_.gstep(2) = {hessian_step_scale:.17g};\n"
        + f"estimation(datafile = {case.data_file}, nobs = {periods}, "
        + "mode_compute = 4, cova_compute = 1, mh_replic = 0);\n"
    )
    path.write_text(source, encoding="utf-8")


def _run_dynare(
    runtime: str,
    case_name: str,
    case: CaseSpec,
    targets: Targets,
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
    hessian_step_scale: float,
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
        _write_model(
            case,
            workdir / f"{model_name}.mod",
            y.shape[0],
            targets,
            hessian_step_scale,
        )
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
    std = np.std(samples, axis=0, ddof=1)
    ess = _effective_sample_size(samples)
    mcse = np.full(ess.shape, np.nan, dtype=np.float64)
    positive_ess = ess > 0.0
    mcse[positive_ess] = std[positive_ess] / np.sqrt(ess[positive_ess])
    quantiles = (0.05, 0.50, 0.95)
    quantile_values = np.quantile(samples, quantiles, axis=0)
    quantile_mcse = _quantile_mcse(samples, quantiles)
    return {
        "std": std,
        "mean": np.mean(samples, axis=0),
        "q05": quantile_values[0],
        "q50": quantile_values[1],
        "q95": quantile_values[2],
        "q05_mcse": quantile_mcse[0],
        "q50_mcse": quantile_mcse[1],
        "q95_mcse": quantile_mcse[2],
        "ess": ess,
        "mcse": mcse,
    }


def _effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """Geyer's initial-positive-sequence ESS for each single-chain column."""
    n_draws, n_params = samples.shape
    ess = np.zeros(n_params, dtype=np.float64)
    if n_draws < 3:
        return ess

    n_fft = 1 << (2 * n_draws - 1).bit_length()
    for index in range(n_params):
        centered = samples[:, index] - np.mean(samples[:, index])
        autocov = np.fft.irfft(np.abs(np.fft.rfft(centered, n=n_fft)) ** 2, n=n_fft)[
            :n_draws
        ]
        autocov /= np.arange(n_draws, 0, -1, dtype=np.float64)
        if autocov[0] <= np.finfo(np.float64).eps:
            continue
        rho = autocov / autocov[0]
        pair_sum = rho[1:-1:2] + rho[2::2]
        nonpositive = np.flatnonzero(pair_sum <= 0.0)
        if nonpositive.size:
            pair_sum = pair_sum[: nonpositive[0]]
        tau = 1.0 + 2.0 * float(np.sum(pair_sum))
        ess[index] = min(float(n_draws), float(n_draws) / tau)
    return ess


def _quantile_mcse(samples: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    """Estimate quantile MCSEs with square-root-sized contiguous batches."""
    n_draws, n_params = samples.shape
    n_batches = int(np.sqrt(n_draws))
    batch_size = n_draws // n_batches if n_batches else 0
    mcse = np.full((len(quantiles), n_params), np.nan, dtype=np.float64)
    if n_batches < 2 or batch_size < 2:
        return mcse

    batches = samples[: n_batches * batch_size].reshape(n_batches, batch_size, n_params)
    for index, quantile in enumerate(quantiles):
        batch_quantiles = np.quantile(batches, quantile, axis=1)
        mcse[index] = np.std(batch_quantiles, axis=0, ddof=1) / np.sqrt(n_batches)
    return mcse


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
    case: CaseSpec,
    estimated_params: tuple[str, ...],
    draws: int,
    burn_in: int,
    native: dict,
    dynare: dict,
):
    print(
        f"{case.label} MCMC: {len(estimated_params)} estimated parameters, "
        f"retained draws={draws} burn-in={burn_in} reps={len(native['times'])}"
    )
    print(
        "Mode and Hessian setup are outside Dynare's timer. Both chains use "
        "adaptive random-walk MH from the shared start draw, but their update "
        "schedules and RNGs differ, so posterior summaries are descriptive."
    )
    native_median = float(np.median(native["times"]))
    native_ess = _effective_sample_size(native["samples"])
    header = (
        f"{'runtime':<18} {'median s':>12} {'draws / s':>12} "
        f"{'acceptance':>12} {'min ESS':>12} {'min ESS / s':>14}"
    )
    print("\n" + header)
    print("-" * len(header))
    print(
        f"{'SymbolicDSGE':<18} {native_median:12.3f} {draws / native_median:12.1f} "
        f"{native['accept_rate']:12.3f} {np.min(native_ess):12.1f} "
        f"{np.min(native_ess) / native_median:14.1f}"
    )
    for runtime, result in dynare.items():
        median = float(np.median(result["times"]))
        ess = _effective_sample_size(result["samples"])
        print(
            f"{'Dynare-' + runtime:<18} {median:12.3f} {draws / median:12.1f} "
            f"{result['accept_rate']:12.3f} {np.min(ess):12.1f} "
            f"{np.min(ess) / median:14.1f}"
        )

    native_summary = _summary(native["samples"])
    for runtime, result in dynare.items():
        result["summary"] = _summary(result["samples"])
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
        for runtime, result in dynare.items():
            summary = result["summary"]
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
    parser.add_argument("--periods", type=int, default=120)
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native-proposal-scale", type=float, default=0.1)
    parser.add_argument("--prior-scale", type=float, default=0.05)
    parser.add_argument("--no-native-adapt", action="store_true")
    parser.add_argument("--adapt-start", type=int, default=100)
    parser.add_argument("--hessian-step-scale", type=float, default=HESSIAN_STEP_SCALE)
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
    if args.prior_scale <= 0:
        parser.error("--prior-scale must be positive")
    if args.adapt_start < 0:
        parser.error("--adapt-start must be nonnegative")
    if args.hessian_step_scale <= 0:
        parser.error("--hessian-step-scale must be positive")
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
            output_dir = root_output / case_name
            output_dir.mkdir(exist_ok=True)
            compiled, solved, y, observables = _prepare(case, args.periods)
            targets = _estimation_targets(compiled, args.prior_scale)
            native = _run_native(
                targets,
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
                args.hessian_step_scale,
            )
            np.savez(output_dir / "native.npz", **native)
            dynare: dict[str, dict] = {}
            for runtime in (item for item in args.runtimes if item != "native"):
                raw = loadmat(
                    _run_dynare(
                        runtime,
                        case_name,
                        case,
                        targets,
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
                        args.hessian_step_scale,
                        args.dynare_matlab_path,
                        args.matlab_bin,
                        args.octave_bin,
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
                case,
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
                        "native_proposal_scale": args.native_proposal_scale,
                        "prior_scale": args.prior_scale,
                        "native_adapt": not args.no_native_adapt,
                        "adapt_start": args.adapt_start,
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
