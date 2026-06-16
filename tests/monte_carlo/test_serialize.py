from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest

from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import (
    EdgeSpec,
    MCPipeline,
    MCPipelineResult,
    NodeSpec,
    PipelineSpec,
)
from SymbolicDSGE.monte_carlo.operations.core import raw_data_step
from SymbolicDSGE.monte_carlo.operations.regressions import regression_step
from SymbolicDSGE.monte_carlo.operations.tests import jarque_bera_test_step
from SymbolicDSGE.monte_carlo.serialize import (
    pipeline_result_wire,
    result_document,
    result_traces,
    serialize_pipeline_result,
)


def _run_demo_pipeline(n_rep: int = 3) -> MCPipelineResult:
    rng = np.random.default_rng(0)
    T = 60
    x = rng.normal(size=(n_rep, T))
    y = 2.0 * x + rng.normal(size=(n_rep, T))
    observables = np.stack([y, x], axis=-1)  # (n_rep, T, 2): col 0 = y, col 1 = x
    pipeline = MCPipeline(
        [
            raw_data_step(observables=observables, observable_names=("y", "x")),
            jarque_bera_test_step("jb", source="observables", column=0),
            regression_step(
                "ols",
                y_source="observables",
                X_source="observables",
                y_column=0,
                X_columns=[1],
                variables=["x"],
            ),
        ]
    )
    return pipeline.run(
        reference=cast(SolvedModel, object()),
        n_rep=n_rep,
        retain_contexts=True,
        verbosity=0,
    )


def test_result_document_drops_bulk_traces_and_is_json_safe() -> None:
    result = _run_demo_pipeline()
    document = result_document(result, run_id="r1")

    test_entry = document["test_summaries"]["jb"]
    for key in ("statistic_trace", "pval_trace", "status_trace"):
        assert key not in test_entry
    # Scalar summaries / metadata survive.
    assert test_entry["statistic_summary"]["n"] == 3
    assert "mean_statistic" in test_entry

    reg_entry = document["regression_summaries"]["ols"]
    for key in ("coef_trace", "r2_trace", "status_trace"):
        assert key not in reg_entry

    # No ndarrays / numpy scalars left behind.
    json.dumps(document)


def test_result_traces_keys_and_shapes() -> None:
    result = _run_demo_pipeline(n_rep=3)
    traces = result_traces(result)

    assert traces["test.jb.statistic"].shape == (3,)
    assert traces["test.jb.pval"].shape == (3,)
    assert traces["test.jb.status"].shape == (3,)
    assert traces["test.jb.status"].dtype == np.int64

    assert traces["regression.ols.coef"].ndim == 2  # n_rep x k
    assert traces["regression.ols.coef"].shape[0] == 3
    assert traces["regression.ols.r2"].shape == (3,)
    assert traces["regression.ols.status"].shape == (3,)


def test_wire_equals_document_plus_traces() -> None:
    result = _run_demo_pipeline()
    wire = serialize_pipeline_result(result, run_id="r1")
    recombined = pipeline_result_wire(
        result_document(result, run_id="r1"), result_traces(result)
    )
    assert recombined == wire


def test_wire_reconstructs_dropped_all_nan_trace_columns() -> None:
    # A test whose statistic/pval are NaN in every rep yields all-null float
    # trace columns, which the Parquet encoder drops. Hydration must not raise
    # on the missing keys; it reconstructs them as null-filled traces.
    result = _run_demo_pipeline(n_rep=3)
    document = result_document(result, run_id="r1")
    traces = result_traces(result)
    # Simulate the encoder dropping the all-null float columns for "jb".
    del traces["test.jb.statistic"]
    del traces["test.jb.pval"]

    wire = pipeline_result_wire(document, traces)

    entry = wire["test_summaries"]["jb"]
    assert entry["statistic_trace"] == [None, None, None]
    assert entry["pval_trace"] == [None, None, None]
    # status (integer-valued) survives and is unchanged.
    assert len(entry["status_trace"]) == 3


def test_pipeline_spec_round_trips() -> None:
    spec = PipelineSpec(
        nodes=[
            NodeSpec(id="n0", step_type="simulation", name="datagen", params={"T": 50}),
            NodeSpec(
                id="n1",
                step_type="jarque_bera",
                name="jb",
                params={"source": "observables"},
            ),
        ],
        edges=[EdgeSpec(source="n0", target="n1")],
    )
    as_dict = spec.to_dict()
    assert PipelineSpec.from_dict(as_dict).to_dict() == as_dict
    assert PipelineSpec.from_json(spec.to_json()).to_dict() == as_dict


def test_pipeline_spec_rejects_unknown_step() -> None:
    with pytest.raises(ValueError):
        PipelineSpec.from_dict(
            {"nodes": [{"id": "n", "step_type": "bogus", "name": "n"}], "edges": []}
        )
