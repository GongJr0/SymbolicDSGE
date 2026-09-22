# type: ignore
"""Shared fixtures for the Monte Carlo suite.

The transform payloads come from a single pipeline run rather than per-test
runs. The transform arithmetic lives in C and the library exports no per-step
runner, so the pipeline is the interface: a producer per kind of input, then one
transform per case reading a producer directly. Nothing chains, which keeps each
expected value a function of the declared input rather than of an upstream
step's parameters, and lets a degenerate input be injected for the branches a
well-behaved sample never enters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.step_factories import (
    add_payload_step,
    diff_step,
    log_diff_step,
    log_step,
    raw_model_data_step,
    rolling_mean_step,
    rolling_std_step,
    rolling_var_step,
    standardize_step,
)


@pytest.fixture(scope="session")
def mc_run():
    """Run a Monte Carlo pipeline the way this suite wants it run.

    ``n_jobs=1`` is the one that matters: these pipelines are toys, a worker pool
    costs more than it saves on them, and the suite runs under ``-n auto`` where
    an all-cores pool per test oversubscribes the machine badly. Centralized here
    so no test has to remember it.

    Returns a callable, since a run is per-pipeline rather than per-session.
    """

    def run(pipeline, reference, *, n_rep, **kwargs):
        kwargs.setdefault("n_jobs", 1)
        kwargs.setdefault("verbosity", 0)
        kwargs.setdefault("check_memory_availability", False)
        return pipeline.run({"reference": reference}, n_rep=n_rep, **kwargs)

    return run


@pytest.fixture(scope="session")
def solved_test_model() -> SolvedModel:
    """A first-order solve of ``MODELS/test.yaml``, shared across the suite."""
    model, kalman = ModelParser("MODELS/test.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


@dataclass(frozen=True)
class TransformRun:
    """Retained transform payloads, beside the inputs and parameters used.

    Carries the step parameters so a consumer needs no module constants to state
    an expected shape.
    """

    out: dict[str, np.ndarray]
    observables: np.ndarray
    positive: np.ndarray
    zeros: np.ndarray
    window: int
    order: int

    @property
    def periods(self) -> int:
        return self.observables.shape[0]

    @property
    def columns(self) -> int:
        return self.observables.shape[1]


@pytest.fixture(scope="session")
def transform_run(solved_test_model, mc_run) -> TransformRun:
    """One pipeline covering every built-in TRANSFORM step, run once.

    Costs one solve and one lowering, with no simulation.
    """
    periods, columns, window, order = 20, 2, 3, 2

    rng = np.random.default_rng(0)
    observables = rng.normal(size=(periods, columns))
    # ``log`` and ``log_diff`` need a positive domain and a standardized sample
    # is centred on zero. Injecting the input keeps them off a NaN path without
    # picking an offset that depends on the data.
    positive = np.abs(rng.normal(size=(periods, columns))) + 1.0
    zeros = np.zeros((periods, columns))

    pipeline = MCPipeline(
        [
            raw_model_data_step(
                "dat", observables=observables, observable_names=("y", "x")
            ),
            add_payload_step("pos", positive),
            add_payload_step("zed", zeros),
            standardize_step("std", source="dat", field="observables"),
            standardize_step("std_zero_var", source="zed", field="payload"),
            log_step("log", source="pos", field="payload"),
            log_diff_step("log_diff", source="pos", field="payload"),
            diff_step("diff", source="dat", field="observables", order=order),
            rolling_mean_step(
                "rmean", source="dat", field="observables", window=window
            ),
            rolling_std_step("rstd", source="dat", field="observables", window=window),
            rolling_var_step("rvar", source="dat", field="observables", window=window),
        ]
    )

    result = mc_run(pipeline, solved_test_model, n_rep=1)
    assert result.failures == ()

    # One replication, so the leading axis is dropped here rather than in every
    # assertion. Frozen because the mapping is shared and an in-place operation
    # in one test would otherwise reach the others.
    out: dict[str, np.ndarray] = {}
    for name, payload in result.transform_outputs.items():
        arr = np.asarray(payload)[0]
        arr.flags.writeable = False
        out[name] = arr

    return TransformRun(
        out=out,
        observables=observables,
        positive=positive,
        zeros=zeros,
        window=window,
        order=order,
    )
