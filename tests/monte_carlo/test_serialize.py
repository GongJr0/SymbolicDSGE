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
from SymbolicDSGE.monte_carlo.mc_constructs import MCMeta
from SymbolicDSGE.monte_carlo.operations.core import raw_data_step
from SymbolicDSGE.monte_carlo.operations.regressions import regression_step
from SymbolicDSGE.monte_carlo.operations.tests import jarque_bera_test_step
from SymbolicDSGE.monte_carlo.postproc import Raw, Summary
from SymbolicDSGE.monte_carlo.serialize import (
    pipeline_result_wire,
    result_document,
    result_postproc_arrays,
    result_postproc_tables,
    result_traces,
    serialize_pipeline_result,
)


def _table_result(postproc: dict) -> MCPipelineResult:
    return MCPipelineResult(
        n_rep=3,
        meta=MCMeta(
            n_rep=3,
            payloads_retained=False,
            test_results_retained=False,
            contexts_retained=False,
        ),
        n_successful=3,
        test_summaries={},
        test_results=None,
        payloads=None,
        contexts=None,
        postproc=postproc,
    )


def _postproc_result() -> MCPipelineResult:
    """A bare result carrying a scalar Summary plus 1-D and 2-D array artifacts."""
    return MCPipelineResult(
        n_rep=5,
        meta=MCMeta(
            n_rep=5,
            payloads_retained=False,
            test_results_retained=False,
            contexts_retained=False,
        ),
        n_successful=5,
        test_summaries={},
        test_results=None,
        payloads=None,
        contexts=None,
        postproc={
            "pcs": Summary(value=0.6, title="PCS"),
            "selection": Raw(value=np.array([0.0, 1.0, 0.0, 1.0, 0.0])),
            "density": Raw(value=np.arange(8.0).reshape(4, 2)),
            "moments": Summary(value=np.array([1.0, 2.0]), render="array"),
        },
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


def test_postproc_scalar_inline_array_to_traces() -> None:
    result = _postproc_result()
    document = result_document(result, run_id="r1")
    arrays = result_postproc_arrays(result)

    # Scalars (and their metadata) stay inline in the document.
    pcs = document["postproc"]["pcs"]
    assert pcs == {
        "kind": "summary",
        "title": "PCS",
        "render": "auto",
        "artifact": "scalar",
        "value": 0.6,
    }
    # Array artifacts keep metadata but the bulk value is stripped to parquet.
    density = document["postproc"]["density"]
    assert density == {"kind": "raw", "artifact": "array", "shape": [4, 2]}
    assert "value" not in document["postproc"]["selection"]
    assert "value" not in document["postproc"]["moments"]

    # Only the ndarray artifacts (Raw + array-valued Summary) become payloads.
    assert set(arrays) == {"selection", "density", "moments"}
    assert arrays["density"].shape == (4, 2)
    json.dumps(document)


def test_postproc_wire_round_trips_scalar_and_arrays() -> None:
    result = _postproc_result()
    wire = serialize_pipeline_result(result, run_id="r1")
    recombined = pipeline_result_wire(
        result_document(result, run_id="r1"),
        result_traces(result),
        result_postproc_arrays(result),
    )
    assert recombined == wire
    assert recombined["postproc"]["pcs"]["value"] == 0.6
    assert recombined["postproc"]["density"]["value"] == [
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
            payloads_retained=False,
            test_results_retained=False,
            contexts_retained=False,
        ),
        n_successful=3,
        test_summaries={},
        test_results=None,
        payloads=None,
        contexts=None,
        postproc={"empty": Raw(value=np.full(3, np.nan))},
    )
    document = result_document(result, run_id="r1")
    wire = pipeline_result_wire(document, {}, {})  # array dropped -> absent
    assert wire["postproc"]["empty"]["value"] == [None, None, None]


def test_postproc_summary_rejects_unserializable_value() -> None:
    # Scalar / ndarray / DataFrame are supported; a bare dict is not.
    result = _table_result({"bad": Summary(value={"a": [1, 2]})})
    with pytest.raises(TypeError, match="scalar, ndarray, or DataFrame"):
        serialize_pipeline_result(result, run_id="r1")


def test_postproc_table_metadata_inline_data_to_parquet() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {"stat": ["mean", "std"], "value": [1.5, 2.0], "ok": [True, False]}
    )
    result = _table_result({"desc": Summary(df, render="table")})

    document = result_document(result, run_id="r1")
    entry = document["postproc"]["desc"]
    assert entry["artifact"] == "table"
    assert entry["columns"] == ["stat", "value", "ok"]
    assert entry["dtypes"] == {"stat": "string", "value": "float", "ok": "bool"}
    assert entry["index"] == {"kind": "range", "name": None}
    assert "data" not in entry  # bulk -> parquet side-channel
    json.dumps(document)

    tables = result_postproc_tables(result)
    assert tables["desc"]["stat"] == ["mean", "std"]
    assert tables["desc"]["ok"] == [True, False]


def test_postproc_table_wire_round_trips() -> None:
    import pandas as pd

    df = pd.DataFrame({"stat": ["a", "b", "c"], "value": [1.0, np.nan, 3.0]})
    labeled = pd.DataFrame(
        {"v": [10.0, 20.0]}, index=pd.Index(["x", "y"], name="label")
    )
    result = _table_result(
        {"desc": Summary(df, render="table"), "lab": Summary(labeled)}
    )

    wire = serialize_pipeline_result(result, run_id="r1")
    recombined = pipeline_result_wire(
        result_document(result, run_id="r1"),
        result_traces(result),
        result_postproc_arrays(result),
        result_postproc_tables(result),
    )
    assert recombined == wire
    # NaN cell -> JSON null, matching the trace convention.
    assert recombined["postproc"]["desc"]["data"]["value"] == [1.0, None, 3.0]
    # Labeled index rides the reserved __index__ column.
    assert recombined["postproc"]["lab"]["index"]["kind"] == "labeled"
    assert recombined["postproc"]["lab"]["data"]["__index__"] == ["x", "y"]


def test_postproc_table_wire_reconstructs_dropped_all_null_column() -> None:
    import pandas as pd

    # An all-null column is dropped by the Parquet encoder; hydration rebuilds it
    # as n nulls from the document's column metadata.
    df = pd.DataFrame({"a": [1.0, 2.0], "blank": [np.nan, np.nan]})
    result = _table_result({"t": Summary(df, render="table")})
    document = result_document(result, run_id="r1")

    wire = pipeline_result_wire(document, {}, {}, {"t": {"a": [1.0, 2.0]}})
    assert wire["postproc"]["t"]["data"]["blank"] == [None, None]
    assert wire["postproc"]["t"]["data"]["a"] == [1.0, 2.0]


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
