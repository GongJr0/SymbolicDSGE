from __future__ import annotations
from typing import cast

import numpy as np
import pytest

from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.mc_constructs import (
    MCDataGenResult,
    MCPipelineResult,
)
from SymbolicDSGE.monte_carlo.step_factories import (
    jarque_bera_test_step,
    raw_model_data_step,
    regression_step,
    standardize_step,
)
from SymbolicDSGE.monte_carlo.serialize import (
    json_safe,
    serialize_regression_results,
    serialize_run_meta,
    serialize_test_results,
    serialize_transform_results,
)

_REFERENCE = cast(SolvedModel, object())


def _run(n_rep: int = 4) -> MCPipelineResult:
    rng = np.random.default_rng(0)
    T = 40
    x = rng.normal(size=T)
    observables = np.column_stack([2.0 * x + rng.normal(size=T), x])
    pipeline = MCPipeline(
        raw_model_data_step(observables=observables, observable_names=("y", "x")),
        jarque_bera_test_step("jb", source="datagen", field="observables", column=0),
        regression_step(
            "ols",
            y_source="datagen",
            y_field="observables",
            X_source="datagen",
            X_field="observables",
            y_column=0,
            X_columns=[1],
            variables=["x"],
        ),
        standardize_step("std", source="datagen", field="observables"),
    )
    return pipeline.run(models={"reference": _REFERENCE}, n_rep=n_rep, verbosity=2)


def _empty_datagen() -> MCDataGenResult:
    """A datagen carrying nothing, for a result assembled without a run.

    A pipeline always has a datagen, so a hand-built result still has to supply
    one. The library never constructs this, which is why it is spelled out here
    rather than offered by the container.
    """
    empty = np.empty((0, 0, 0), dtype=np.float64)
    return MCDataGenResult(
        n_rep=0,
        n_retained=0,
        retained_reps=np.empty((0,), dtype=np.int64),
        var_names=(),
        X=empty,
        shock_names=(),
        eps=empty,
    )


def test_run_meta_records_what_the_run_cannot_recompute() -> None:
    result = _run()
    meta = serialize_run_meta(result)

    assert meta["n_rep"] == result.n_rep
    assert meta["n_successful"] == result.n_successful
    assert meta["step_counts"]["jb"] == result.meta.step_counts["jb"]
    assert meta["run_config"] == result.run_config
    # The rates, `succeeded` and the failed-step tallies are all properties over
    # the timings and failures beside them, so none of them are stored.
    for derived in ("it_s", "step_it_s", "succeeded", "failed_steps"):
        assert derived not in meta


def test_traces_are_the_result_arrays_not_copies() -> None:
    # A large run's traces are gigabytes; the serializer hands over the run's own
    # buffers, so nothing here may allocate.
    result = _run()
    summary = result.test_summaries["jb"]
    _, traces = serialize_test_results(result.test_summaries)["jb"]

    assert traces["statistic_trace"] is summary.statistic_trace
    assert traces["status_trace"] is summary._raw_status


def test_test_meta_carries_the_fields_from_spec_needs() -> None:
    result = _run(n_rep=5)
    meta, traces = serialize_test_results(result.test_summaries)["jb"]

    assert meta["test_name"] and meta["n_rep"] == 5
    assert set(traces) == {"statistic_trace", "status_trace", "retained_reps"}
    assert traces["retained_reps"].shape == (meta["n_retained"],)


def test_regression_omits_absent_standard_errors() -> None:
    result = _run()
    meta, traces = serialize_regression_results(result.regression_summaries)["ols"]

    assert traces["coef_trace"].shape[1] == meta["k"]
    assert "se_trace" in traces  # ols carries them

    stripped = result.regression_summaries["ols"]
    object.__setattr__(stripped, "_se_trace", None)
    _, without = serialize_regression_results({"ols": stripped})["ols"]
    assert "se_trace" not in without  # absent, never a null column


def test_transform_shape_rides_the_meta_and_carries_only_its_payload() -> None:
    result = _run(n_rep=6)
    meta, traces = serialize_transform_results(result.transform_outputs)["std"]

    payload = result.transform_outputs["std"]
    assert meta["shape"] == list(payload.shape)
    assert traces["value"] is payload
    # A transform is a bare array with no container to hold an index map, so the
    # member carries the payload and nothing beside it.
    assert set(traces) == {"value"}


def test_json_safe_reaches_into_containers() -> None:
    value = {
        "arr": np.arange(3),
        "nested": [np.float64(0.5), (np.int64(2),)],
        "nan": np.float64("nan"),
    }

    assert json_safe(value) == {
        "arr": [0, 1, 2],
        "nested": [0.5, [2]],
        "nan": None,  # JSON has no NaN; a null is what the readers expect
    }


@pytest.mark.parametrize("n_rep", [1, 3])
def test_retained_indices_span_the_run(n_rep: int) -> None:
    result = _run(n_rep=n_rep)
    _, traces = serialize_test_results(result.test_summaries)["jb"]

    assert traces["retained_reps"].tolist() == list(range(n_rep))
