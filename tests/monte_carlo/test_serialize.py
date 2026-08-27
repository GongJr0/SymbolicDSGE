from __future__ import annotations

import json
from io import StringIO
from typing import cast

import numpy as np
import pytest

from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.mc_constructs import MCMeta, MCPipelineResult
from SymbolicDSGE.monte_carlo.postproc import Artifact, Raw, Summary
from SymbolicDSGE.monte_carlo.step_factories import (
    jarque_bera_test_step,
    raw_model_data_step,
    regression_step,
    standardize_step,
)
from SymbolicDSGE.monte_carlo.serialize import (
    json_safe,
    serialize_pipeline_result,
    serialize_postproc_results,
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
            standardize_step("std", source="datagen", field="observables"),
        ]
    )
    return pipeline.run(reference=_REFERENCE, n_rep=n_rep, verbosity=2)


def _postproc_result(postproc: dict[str, Artifact]) -> MCPipelineResult:
    return MCPipelineResult(
        meta=MCMeta(n_rep=3, n_retained_by_step={}),
        n_rep=3,
        n_successful=3,
        postproc=postproc,
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


def test_transform_shape_rides_the_meta_and_indices_the_traces() -> None:
    result = _run(n_rep=6)
    meta, traces = serialize_transform_results(result.transform_outputs, result.n_rep)[
        "std"
    ]

    payload = result.transform_outputs["std"]
    assert meta["shape"] == list(payload.shape)
    assert traces["value"] is payload
    # The arenas that held them are gone, so the indices come back from the
    # retained row count and n_rep.
    assert traces["retained_reps"].tolist() == list(range(6))


def test_postproc_slots_are_independent() -> None:
    curve = np.arange(8.0).reshape(4, 2)
    steps = serialize_postproc_results(
        {
            "both": Artifact(raw=Raw(value=curve), summary=Summary(value=0.6)),
            "summary_only": Artifact(raw=None, summary=Summary(value={"n": 4})),
            "raw_only": Artifact(raw=Raw(value=curve), summary=None),
        }
    )

    both_meta, both_traces = steps["both"]
    assert both_meta["shape"] == [4, 2] and both_meta["summary"] == 0.6
    assert both_traces["value"].shape == (4, 2)

    summary_meta, summary_traces = steps["summary_only"]
    assert summary_meta["shape"] is None
    assert summary_traces == {}  # nothing bulk, so no member

    raw_meta, _ = steps["raw_only"]
    assert raw_meta["summary"] is None


def test_dataframe_summary_rides_pandas_table_schema() -> None:
    import pandas as pd

    frame = pd.DataFrame({"stat": ["a", "b"], "value": [1 / 3, np.nan]})
    labeled = pd.DataFrame({"v": [10.0]}, index=pd.Index(["x"], name="lab"))
    steps = serialize_postproc_results(
        {
            "desc": Artifact(raw=None, summary=Summary(frame)),
            "idx": Artifact(raw=None, summary=Summary(labeled)),
        }
    )

    for name, original in (("desc", frame), ("idx", labeled)):
        payload = steps[name][0]["summary"]
        json.dumps(payload)  # inline and JSON-safe, no side channel
        back = pd.read_json(StringIO(json.dumps(payload)), orient="table")
        assert dict(back.dtypes) == dict(original.dtypes)
        # A summary is a presentation surface: JSON caps floats at 15 digits and
        # the traces are what carries the exact values.
        pd.testing.assert_frame_equal(back, original, rtol=1e-13, atol=0)
    assert steps["idx"][0]["summary"]["schema"]["primaryKey"] == ["lab"]


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


def test_ui_wire_stays_one_flat_json_document() -> None:
    result = _run()
    wire = serialize_pipeline_result(result)

    assert wire["kind"] == "mc"
    # Derived values the bundle refuses to store are present here, since this is
    # the shape a client renders rather than the shape a run is rebuilt from.
    assert wire["test_summaries"]["jb"]["pval_trace"]
    assert wire["regression_summaries"]["ols"]["r2_trace"]
    json.dumps(wire)


def test_postproc_summary_mapping_inlines() -> None:
    result = _postproc_result(
        {
            "m": Artifact(
                raw=None, summary=Summary(value={"a": [1, 2], "b": np.float64(0.5)})
            )
        }
    )

    wire = serialize_pipeline_result(result)

    assert wire["postproc"]["m"]["summary"]["value"] == {"a": [1, 2], "b": 0.5}
    json.dumps(wire)


@pytest.mark.parametrize("n_rep", [1, 3])
def test_retained_indices_span_the_run(n_rep: int) -> None:
    result = _run(n_rep=n_rep)
    _, traces = serialize_test_results(result.test_summaries)["jb"]

    assert traces["retained_reps"].tolist() == list(range(n_rep))
