"""Benchmark complete native Monte Carlo pipelines.

Each sample measures one public ``MCPipeline.run`` call, including Python
lowering, arena allocation, native execution, retained-result construction,
and metadata assembly. Native timings are compared with the preserved Python
pipeline oracle. The three cases use the POST82 fixture:

* raw data -> filter -> Ljung-Box on standardized innovations;
* simulation -> filter -> OLS of the first innovation on ``x_pred[:, 2:]``;
* simulation -> per-column standardization -> Breusch-Godfrey.

Every per-replication step explicitly uses ``n_retain=-1``. The DGP changes
``psi_pi`` and ``rho_r`` from the reference calibration, avoiding the
degenerate reference-equals-DGP benchmark. Defaults run 200 Monte Carlo
replications per invocation and collect 200 timed invocations per case.

Usage:
    uv run python scripts/bench_native_monte_carlo.py
    uv run python scripts/bench_native_monte_carlo.py --runs 50 --n-rep 100
    uv run python scripts/bench_native_monte_carlo.py --cases raw_filter_lb
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from sympy import Symbol

ROOT = Path(__file__).resolve().parents[1]
POST82 = ROOT / "tests" / "fixtures" / "models" / "POST82.yaml"
sys.path.insert(0, str(ROOT))

NDF: TypeAlias = NDArray[np.float64]

from SymbolicDSGE import DSGESolver, ModelParser  # noqa: E402
from SymbolicDSGE.core.shock_generators import Shock  # noqa: E402
from SymbolicDSGE.core.solved_model import SolvedModel  # noqa: E402
from SymbolicDSGE.monte_carlo.core import MCPipeline  # noqa: E402
from SymbolicDSGE.monte_carlo.mc_constructs import (
    MCStep,
    MCPipelineResult,
)  # noqa: E402
from SymbolicDSGE.monte_carlo.step_factories import (  # noqa: E402
    breusch_godfrey_test_step,
    ljung_box_test_step,
    raw_model_data_step,
    reference_filter_step,
    regression_step,
    standardize_step,
    simulation_step,
)
from tests._oracles.monte_carlo.core import MCPipeline as LegacyMCPipeline  # noqa: E402
from tests._oracles.monte_carlo.mc_constructs import (  # noqa: E402
    MCStep as LegacyMCStep,
)
from tests._oracles.monte_carlo.operations.core import (  # noqa: E402
    raw_model_data_step as legacy_raw_model_data_step,
)
from tests._oracles.monte_carlo.operations.core import (  # noqa: E402
    reference_filter_step as legacy_reference_filter_step,
)
from tests._oracles.monte_carlo.operations.core import (  # noqa: E402
    simulation_step as legacy_simulation_step,
)
from tests._oracles.monte_carlo.operations.regressions import (  # noqa: E402
    regression_step as legacy_regression_step,
)
from tests._oracles.monte_carlo.operations.tests import (  # noqa: E402
    breusch_godfrey_test_step as legacy_breusch_godfrey_test_step,
)
from tests._oracles.monte_carlo.operations.tests import (  # noqa: E402
    ljung_box_test_step as legacy_ljung_box_test_step,
)
from tests._oracles.monte_carlo.operations.transforms import (  # noqa: E402
    standardize_step as legacy_standardize_step,
)


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    pipeline: MCPipeline
    oracle_pipeline: LegacyMCPipeline


def _pipeline(steps: Sequence[MCStep]) -> MCPipeline:
    """Build a benchmark pipeline with full output retention at every step."""
    return MCPipeline(tuple(replace(step, n_retain=-1) for step in steps))


def _legacy_pipeline(steps: Sequence[LegacyMCStep]) -> LegacyMCPipeline:
    return LegacyMCPipeline(tuple(replace(step, n_retain=-1) for step in steps))


def _solve_post82_models() -> tuple[SolvedModel, SolvedModel]:
    model, kalman = ModelParser(POST82).get_all()
    solver = DSGESolver(model, kalman)
    reference = solver.solve(solver.compile())

    dgp_model, dgp_kalman = ModelParser(POST82).get_all()
    dgp_model.calibration.parameters[Symbol("psi_pi")] = np.float64(2.45)
    dgp_model.calibration.parameters[Symbol("rho_r")] = np.float64(0.70)
    dgp_solver = DSGESolver(dgp_model, dgp_kalman)
    dgp = dgp_solver.solve(dgp_solver.compile())
    return reference, dgp


def _raw_observables(reference: SolvedModel, T: int) -> NDF:
    simulated = reference.sim(T=T, shocks=_shocks(reference), observables=True)
    return cast(
        NDF,
        np.column_stack(
            [simulated[name] for name in reference.compiled.observable_names]
        ).astype(np.float64, copy=False),
    )


def _shocks(reference: SolvedModel) -> dict[str, Shock]:
    return {
        name: Shock(dist="norm", seed=20_260_730 + index)
        for index, name in enumerate(reference.compiled.layout.exo_state_names)
    }


def _cases(
    reference: SolvedModel,
    dgp: SolvedModel,
    T: int,
) -> tuple[Case, ...]:
    raw_observables = _raw_observables(dgp, T)
    return (
        Case(
            "raw_filter_lb",
            "raw data -> filter -> LB(std_innov[:, 0])",
            _pipeline(
                (
                    raw_model_data_step(
                        "data",
                        observables=raw_observables,
                        observable_names=reference.compiled.observable_names,
                    ),
                    reference_filter_step("filter"),
                    ljung_box_test_step(
                        "lb",
                        source="filter",
                        field="std_innov",
                        column=0,
                        lags=10,
                    ),
                )
            ),
            _legacy_pipeline(
                (
                    legacy_raw_model_data_step(
                        "data",
                        observables=raw_observables,
                        observable_names=reference.compiled.observable_names,
                    ),
                    legacy_reference_filter_step("filter"),
                    legacy_ljung_box_test_step(
                        "lb",
                        source="filter",
                        field="std_innov",
                        column=0,
                        lags=10,
                    ),
                )
            ),
        ),
        Case(
            "sim_filter_ols",
            "sim(POST82) -> filter -> OLS(innov[:, 0] ~ x_pred[:, 2:])",
            _pipeline(
                (
                    simulation_step(
                        "sim",
                        target="dgp",
                        T=T,
                        shocks=_shocks(reference),
                        observables=True,
                    ),
                    reference_filter_step("filter"),
                    regression_step(
                        "ols",
                        y_source="filter",
                        y_field="innov",
                        y_column=0,
                        X_source="filter",
                        X_field="x_pred",
                        X_columns=slice(2, None),
                        variables=["r", "x", "Pi"],
                    ),
                )
            ),
            _legacy_pipeline(
                (
                    legacy_simulation_step(
                        "sim",
                        target="dgp",
                        T=T,
                        shocks=_shocks(reference),
                        observables=True,
                    ),
                    legacy_reference_filter_step("filter"),
                    legacy_regression_step(
                        "ols",
                        y_source="filter",
                        y_field="innov",
                        y_column=0,
                        X_source="filter",
                        X_field="x_pred",
                        X_columns=slice(2, None),
                        variables=["r", "x", "Pi"],
                    ),
                )
            ),
        ),
        Case(
            "sim_standardize_bg",
            "sim(POST82) -> standardize axis 0 -> BG(std_obs[:, 0], std_obs[:, 1:])",
            _pipeline(
                (
                    simulation_step(
                        "sim",
                        target="dgp",
                        T=T,
                        shocks=_shocks(reference),
                        observables=True,
                    ),
                    standardize_step(
                        "std_obs",
                        source="sim",
                        field="observables",
                    ),
                    breusch_godfrey_test_step(
                        "bg",
                        residuals_source="std_obs",
                        residuals_field="payload",
                        residual_col=0,
                        X_source="std_obs",
                        X_field="payload",
                        X_columns=slice(1, None),
                        lags=1,
                    ),
                )
            ),
            _legacy_pipeline(
                (
                    legacy_simulation_step(
                        "sim",
                        target="dgp",
                        T=T,
                        shocks=_shocks(reference),
                        observables=True,
                    ),
                    legacy_standardize_step(
                        "std_obs",
                        source="sim",
                        field="observables",
                    ),
                    legacy_breusch_godfrey_test_step(
                        "bg",
                        residuals_source="std_obs",
                        residuals_field="payload",
                        residual_col=0,
                        X_source="std_obs",
                        X_field="payload",
                        X_columns=slice(1, None),
                        lags=1,
                    ),
                )
            ),
        ),
    )


def _time(fn: Callable[[], MCPipelineResult], runs: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(runs):
        started_s = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started_s)
    return samples


def _validate(result: MCPipelineResult, n_rep: int, label: str) -> None:
    if not result.succeeded or result.n_successful != n_rep:
        raise RuntimeError(f"{label} pipeline failed: {result.failures!r}")
    retained_by_step = getattr(result.meta, "n_retained_by_step", None)
    if retained_by_step is not None and set(retained_by_step.values()) != {n_rep}:
        raise RuntimeError(
            "Benchmark expected n_retain=-1 to retain every replication, got "
            f"{retained_by_step!r}."
        )
    nonzero_statuses: dict[str, list[int]] = {}
    for name, statuses in result.test_status_traces.items():
        values = sorted({int(status) for status in statuses if status != 0})
        if values:
            nonzero_statuses[f"test.{name}"] = values
    for name, statuses in result.regression_status_traces.items():
        values = sorted({int(status) for status in statuses if status != 0})
        if values:
            nonzero_statuses[f"regression.{name}"] = values
    if nonzero_statuses:
        raise RuntimeError(f"{label} emitted nonzero statuses: {nonzero_statuses!r}")


def _result_traces(result: MCPipelineResult) -> dict[str, np.ndarray]:
    traces: dict[str, np.ndarray] = {}
    for name, summary in result.test_summaries.items():
        traces[f"test.{name}.statistic"] = np.asarray(summary.statistic_trace)
        traces[f"test.{name}.pval"] = np.asarray(summary.pval_trace)
    for name, summary in result.regression_summaries.items():
        traces[f"regression.{name}.coef"] = np.asarray(summary.coef_trace)
    return traces


def _moment_deltas(
    native: MCPipelineResult,
    oracle: MCPipelineResult,
) -> tuple[float, float]:
    native_traces = _result_traces(native)
    oracle_traces = _result_traces(oracle)
    if native_traces.keys() != oracle_traces.keys():
        raise RuntimeError("Native and oracle result traces do not match.")
    max_mean_delta = 0.0
    max_variance_delta = 0.0
    for name, native_trace in native_traces.items():
        oracle_trace = oracle_traces[name]
        if native_trace.shape != oracle_trace.shape:
            raise RuntimeError(f"Trace {name!r} has incompatible shapes.")
        native_values = np.asarray(native_trace, dtype=np.float64).reshape(-1)
        oracle_values = np.asarray(oracle_trace, dtype=np.float64).reshape(-1)
        max_mean_delta = max(
            max_mean_delta,
            abs(float(np.nanmean(native_values)) - float(np.nanmean(oracle_values))),
        )
        max_variance_delta = max(
            max_variance_delta,
            abs(
                float(np.nanvar(native_values, ddof=1))
                - float(np.nanvar(oracle_values, ddof=1))
            ),
        )
    return max_mean_delta, max_variance_delta


def _milliseconds(seconds: float) -> str:
    return f"{seconds * 1e3:12.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rep", type=int, default=200)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("raw_filter_lb", "sim_filter_ols", "sim_standardize_bg"),
        default=("raw_filter_lb", "sim_filter_ols", "sim_standardize_bg"),
    )
    args = parser.parse_args()
    if args.n_rep <= 0 or args.runs <= 0 or args.warmup < 0 or args.T <= 0:
        parser.error(
            "n-rep, runs, and T must be positive; warmup must be non-negative."
        )

    reference, dgp = _solve_post82_models()
    selected = {
        case.name: case
        for case in _cases(reference, dgp, args.T)
        if case.name in args.cases
    }
    print(
        f"POST82, T={args.T}, n_rep={args.n_rep}, n_retain=-1, "
        f"runs={args.runs}, warmup={args.warmup}, n_jobs={args.n_jobs}\n"
        "Reference calibration: psi_pi=2.19, rho_r=0.84. "
        "DGP calibration: psi_pi=2.45, rho_r=0.70.\n"
        "Times include lowering, allocation, native execution, and result collection.\n"
    )
    header = (
        f"{'case':<21} {'native med':>12} {'python med':>12} {'speedup':>9} "
        f"{'native min':>12} {'python min':>12} {'max |dmean|':>13} "
        f"{'max |dvar|':>13}"
    )
    print(header)
    print("-" * len(header))
    for case_name in args.cases:
        case = selected[case_name]

        def run_case(case: Case = case) -> MCPipelineResult:
            return case.pipeline.run(
                reference=reference,
                dgp=dgp,
                n_rep=args.n_rep,
                n_jobs=args.n_jobs,
                verbosity=0,
            )

        def run_oracle(case: Case = case) -> MCPipelineResult:
            return case.oracle_pipeline.run(
                reference=reference,
                dgp=dgp,
                n_rep=args.n_rep,
                retain_payloads=False,
                retain_test_results=False,
                retain_contexts=False,
                verbosity=0,
            )

        native_result = run_case()
        oracle_result = run_oracle()
        _validate(native_result, args.n_rep, "native")
        _validate(oracle_result, args.n_rep, "oracle")
        mean_delta, variance_delta = _moment_deltas(native_result, oracle_result)
        native_samples = _time(run_case, args.runs, args.warmup)
        oracle_samples = _time(run_oracle, args.runs, args.warmup)
        native_median_s = statistics.median(native_samples)
        oracle_median_s = statistics.median(oracle_samples)
        speedup = oracle_median_s / native_median_s
        print(
            f"{case.name:<21} {_milliseconds(native_median_s)} "
            f"{_milliseconds(oracle_median_s)} {speedup:8.2f}x "
            f"{_milliseconds(min(native_samples))} "
            f"{_milliseconds(min(oracle_samples))} {mean_delta:13.3e} "
            f"{variance_delta:13.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
