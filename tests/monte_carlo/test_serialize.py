from __future__ import annotations

import json
from io import StringIO
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
from SymbolicDSGE.monte_carlo.mc_constructs import MCMeta
from SymbolicDSGE.monte_carlo.step_factories import (
    jarque_bera_test_step,
    raw_model_data_step,
    regression_step,
)
from SymbolicDSGE.monte_carlo.postproc import Artifact, Raw, Summary
from SymbolicDSGE.monte_carlo.serialize import (
    pipeline_result_wire,
    result_document,
    result_postproc_arrays,
    run_traces,
    serialize_pipeline_result,
)


def _table_result(postproc: dict) -> MCPipelineResult:
    return MCPipelineResult(
        n_rep=3,
        meta=MCMeta(
            n_rep=3,
            n_retained_by_step={},
        ),
        n_successful=3,
        test_summaries={},
        transform_outputs=None,
        postproc=postproc,
    )


def _postproc_result() -> MCPipelineResult:
    """A bare result carrying a scalar Summary plus 1-D and 2-D array artifacts."""
    return MCPipelineResult(
        n_rep=5,
        meta=MCMeta(
            n_rep=5,
            n_retained_by_step={},
        ),
        n_successful=5,
        test_summaries={},
        transform_outputs=None,
        postproc={
            "pcs": Artifact(raw=None, summary=Summary(value=0.6)),
            "kde": Artifact(
                raw=Raw(value=np.arange(8.0).reshape(4, 2)),
                summary=Summary(value={"mean": 1.0, "n": 4}),
            ),
            "moments": Artifact(raw=None, summary=Summary(value=np.array([1.0, 2.0]))),
        },
    )


def _run_demo_pipeline(n_rep: int = 3) -> MCPipelineResult:
    rng = np.random.default_rng(0)
    T = 60
    x = rng.normal(size=T)
    y = 2.0 * x + rng.normal(size=T)
    observables = np.column_stack([y, x])
    pipeline = MCPipeline(
        [
            raw_model_data_step(observables=observables, observable_names=("y", "x")),
            jarque_bera_test_step(
                "jb", source="datagen", field="observables", column=0
            ),
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
        ]
    )
    return pipeline.run(
        reference=cast(SolvedModel, object()),
        n_rep=n_rep,
        verbosity=2,
    )


def test_result_document_drops_bulk_traces_and_is_json_safe() -> None:
    result = _run_demo_pipeline()
    document = result_document(result, run_id="r1")

    assert set(document["step_worker_it_s"]) == {"datagen", "jb", "ols"}
    assert set(document["step_wall_it_s"]) == {"datagen", "jb", "ols"}

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


def test_run_traces_keys_and_shapes() -> None:
    result = _run_demo_pipeline(n_rep=3)
    traces = run_traces(result)

    assert traces["test.jb.statistic"].shape == (3,)
    assert traces["test.jb.status"].shape == (3,)
    assert traces["test.jb.status"].dtype == np.int64

    assert traces["regression.ols.coef"].ndim == 2  # n_rep x k
    assert traces["regression.ols.coef"].shape[0] == 3
    assert traces["regression.ols.ssr"].shape == (3,)
    assert traces["regression.ols.sst"].shape == (3,)
    assert traces["regression.ols.se"].shape == (
        3,
        traces["regression.ols.coef"].shape[1],
    )
    assert traces["regression.ols.status"].shape == (3,)

    # Derived columns are recomputed from the ones beside them, never stored.
    assert "test.jb.pval" not in traces
    assert "regression.ols.r2" not in traces


@pytest.mark.xfail(
    strict=True,
    reason="run_traces stores no pval/r2, so the remerge nulls what "
    "serialize_pipeline_result derives; dies with pipeline_result_wire.",
)
def test_wire_equals_document_plus_traces() -> None:
    result = _run_demo_pipeline()
    wire = serialize_pipeline_result(result, run_id="r1")
    recombined = pipeline_result_wire(
        result_document(result, run_id="r1"), run_traces(result)
    )
    assert recombined == wire


def test_wire_reconstructs_dropped_all_nan_trace_columns() -> None:
    # A test whose statistic/pval are NaN in every rep yields all-null float
    # trace columns, which the Parquet encoder drops. Hydration must not raise
    # on the missing keys; it reconstructs them as null-filled traces.
    result = _run_demo_pipeline(n_rep=3)
    document = result_document(result, run_id="r1")
    traces = run_traces(result)
    # Simulate the encoder dropping the all-null float columns for "jb".
    del traces["test.jb.statistic"]

    wire = pipeline_result_wire(document, traces)

    entry = wire["test_summaries"]["jb"]
    assert entry["statistic_trace"] == [None, None, None]

    # status (integer-valued) survives and is unchanged.
    assert len(entry["status_trace"]) == 3


def test_postproc_summary_inlines_and_raw_goes_to_parquet() -> None:
    result = _postproc_result()
    document = result_document(result, run_id="r1")
    arrays = result_postproc_arrays(result)

    # A summary-only step has no bulk slot and keeps its value inline.
    assert document["postproc"]["pcs"] == {"raw": None, "summary": {"value": 0.6}}
    # With both slots, the summary stays inline and the raw value is stripped.
    kde = document["postproc"]["kde"]
    assert kde["raw"] == {"shape": [4, 2]}
    assert kde["summary"] == {"value": {"mean": 1.0, "n": 4}}
    # An ndarray summary inlines too; only Raw is bulk.
    assert document["postproc"]["moments"]["summary"] == {"value": [1.0, 2.0]}

    assert set(arrays) == {"kde"}
    assert arrays["kde"].shape == (4, 2)
    json.dumps(document)


def test_postproc_wire_round_trips_scalar_and_arrays() -> None:
    result = _postproc_result()
    wire = serialize_pipeline_result(result, run_id="r1")
    recombined = pipeline_result_wire(
        result_document(result, run_id="r1"),
        run_traces(result),
        result_postproc_arrays(result),
    )
    assert recombined == wire
    assert recombined["postproc"]["pcs"]["summary"]["value"] == 0.6
    assert recombined["postproc"]["kde"]["raw"]["value"] == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
    ]


def test_postproc_wire_reconstructs_dropped_all_nan_array() -> None:
    # An all-NaN Raw becomes an all-null column the Parquet encoder drops;
    # hydration rebuilds it as a NaN array of the recorded shape (-> JSON null).
    result = MCPipelineResult(
        n_rep=3,
        meta=MCMeta(
            n_rep=3,
            n_retained_by_step={},
        ),
        n_successful=3,
        test_summaries={},
        transform_outputs=None,
        postproc={"empty": Artifact(raw=Raw(value=np.full(3, np.nan)), summary=None)},
    )
    document = result_document(result, run_id="r1")
    wire = pipeline_result_wire(document, {}, {})  # array dropped -> absent
    assert wire["postproc"]["empty"]["raw"]["value"] == [None, None, None]


def test_postproc_summary_mapping_inlines() -> None:
    # A mapping is JSON-native; it rides the summary slot unchanged.
    value = {"a": [1, 2], "b": np.float64(0.5)}
    result = _table_result({"m": Artifact(raw=None, summary=Summary(value=value))})

    wire = serialize_pipeline_result(result, run_id="r1")

    assert wire["postproc"]["m"]["summary"]["value"] == {"a": [1, 2], "b": 0.5}
    json.dumps(wire)


def test_postproc_summary_frame_inlines_as_table_schema() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {"stat": ["a", "b"], "value": [1 / 3, np.nan], "ok": [True, False]}
    )
    labeled = pd.DataFrame({"v": [10.0, 20.0]}, index=pd.Index(["x", "y"], name="lab"))
    result = _table_result(
        {
            "desc": Artifact(raw=None, summary=Summary(df)),
            "idx": Artifact(raw=None, summary=Summary(labeled)),
        }
    )

    document = result_document(result, run_id="r1")
    json.dumps(document)  # inline and JSON-safe, no side-channel

    # pandas' own schema carries columns, dtypes and the index, so it reads back.
    # A summary is a presentation surface: JSON caps floats at 15 digits, and the
    # traces are what carries the exact values.
    for name, original in (("desc", df), ("idx", labeled)):
        payload = document["postproc"][name]["summary"]["value"]
        back = pd.read_json(StringIO(json.dumps(payload)), orient="table")
        assert list(back.columns) == list(original.columns)
        assert back.index.equals(original.index)
        assert dict(back.dtypes) == dict(original.dtypes)
        pd.testing.assert_frame_equal(back, original, rtol=1e-15, atol=0)
    assert document["postproc"]["idx"]["summary"]["value"]["schema"]["primaryKey"] == [
        "lab"
    ]
