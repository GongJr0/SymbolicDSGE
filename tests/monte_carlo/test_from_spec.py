from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable

import pytest
import numpy as np

from SymbolicDSGE.kalman.filter import FilterResult, UnscentedFilterResult
from SymbolicDSGE.monte_carlo import MCPipeline, OpType
from SymbolicDSGE.monte_carlo.custom_op import NumbaCustomFunc
from SymbolicDSGE.monte_carlo.postproc import run_kde as _run_kde
from SymbolicDSGE.monte_carlo.mc_constructs import (
    DYNAMIC_SOURCE_FIELDS,
    FILTER_RAW_SOURCE_FIELDS,
    MC_DATA_SOURCE_FIELDS,
)
from SymbolicDSGE.monte_carlo.step_factories import (
    standardize_step,
    transform_step,
)
from SymbolicDSGE.monte_carlo.spec import (
    OP_TYPES,
    STEP_KINDS,
    PipelineMeta,
    PipelineSpec,
    StepSpec,
    pipeline_meta,
)
from SymbolicDSGE.monte_carlo.traces import _trace_keys
from tests._spec_helpers import _posted_step


def _step(
    step_type: str,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    func: Callable[..., Any] | None = None,
) -> StepSpec:
    """One step, authored flat and resolved the way the client resolves one."""
    meta = _posted_step({"step_type": step_type, "name": name, "params": params or {}})
    meta.pop("code", None)
    return StepSpec(meta=meta, func=func)


def _spec(
    replication_steps: list[StepSpec],
    postproc_steps: list[StepSpec] | None = None,
) -> PipelineSpec:
    return PipelineSpec(
        replication_steps=replication_steps,
        postproc_steps=postproc_steps or [],
    )


def _meta(spec: PipelineSpec) -> PipelineMeta:
    return pipeline_meta(spec)


def test_every_step_kind_declares_an_op_kind() -> None:
    # A step states its own op kind, drawn from this map, so a kind missing from
    # it could never be built.
    assert set(OP_TYPES) == set(STEP_KINDS)


def test_source_kwargs_compile_to_runner_args_once() -> None:
    step = standardize_step(
        "std",
        source="obs",
        field="payload",
        columns=0,
        burn_in=2,
        ddof=1,
    )

    assert dict(step.kwargs) == {"ddof": 1}
    assert len(step.source_args) == 1
    selector = step.source_args[0]
    assert selector.arg == "sample"
    assert selector.source_step == "obs"
    assert selector.columns == (0,)
    assert selector.burn_in == 2


def test_source_arg_compile_validates_static_selection() -> None:
    with pytest.raises(TypeError, match="field"):
        standardize_step("bad_string", source="observables")
    with pytest.raises(ValueError, match="source must be non-empty"):
        standardize_step("bad_payload", source="", field="payload")
    with pytest.raises(ValueError, match="burn_in"):
        standardize_step("bad_burn", source="datagen", field="states", burn_in=-1)


def _custom_copy(sample: np.ndarray, output: np.ndarray) -> int:
    output[:] = sample
    return 0


def test_transform_step_wraps_custom_function_and_compiles_source() -> None:
    step = transform_step(
        "tf",
        _custom_copy,
        source="dat",
        field="observables",
        output_shape=(4, 2),
        columns=(0, 1),
        burn_in=1,
    )

    assert step.step_type == "transform:custom"
    assert isinstance(step.func, NumbaCustomFunc)
    assert step.kwargs == {"output_shape": (4, 2)}
    assert step.source_args[0].source_step == "dat"
    assert step.source_args[0].field == "observables"
    assert step.source_args[0].columns == (0, 1)
    assert step.source_args[0].burn_in == 1


def test_source_fields_match_the_native_output_channels() -> None:
    assert MC_DATA_SOURCE_FIELDS == ("states", "shocks", "observables")
    # ``status`` is a scalar error code, not a selectable array source, so it is
    # excluded from the source-field set. Native lowering resolves the layouts,
    # so source fields no longer carry Python-side positional indices.
    linear_array_fields = tuple(
        f.name for f in fields(FilterResult) if f.name != "status"
    )
    unscented_array_fields = tuple(
        f.name for f in fields(UnscentedFilterResult) if f.name != "status"
    )
    assert FILTER_RAW_SOURCE_FIELDS[: len(linear_array_fields)] == linear_array_fields
    assert FILTER_RAW_SOURCE_FIELDS == unscented_array_fields
    assert DYNAMIC_SOURCE_FIELDS == ("payload",)


def test_terminal_step_kinds_are_the_tests_and_the_regression() -> None:
    terminals = {
        step_type
        for step_type, op_type in OP_TYPES.items()
        if op_type in ("test", "regression")
    }
    assert terminals == {
        "wald",
        "ljung_box",
        "jarque_bera",
        "breusch_pagan",
        "breusch_godfrey",
        "cusum",
        "cusumsq",
        "chow",
        "regression",
    }


def test_validate_orders_steps_with_explicit_filter_source() -> None:
    spec = _spec(
        [
            _step("simulation", "datagen", {"T": 8}),
            _step("filter", "renamed_filter", {}),
            _step(
                "breusch_pagan",
                "diagnostic",
                {
                    "residuals_source": "renamed_filter",
                    "residuals_field": "std_innov",
                    "X_source": "datagen",
                    "X_field": "observables",
                },
            ),
        ]
    )

    pipeline = MCPipeline.from_spec(spec)

    assert [step.name for step in pipeline.replication_steps] == [
        "datagen",
        "renamed_filter",
        "diagnostic",
    ]
    residuals = next(
        selector
        for selector in pipeline.replication_steps[-1].source_args
        if selector.field == "std_innov"
    )
    assert residuals.source_step == "renamed_filter"


def test_validate_binds_multi_source_terminal_from_distinct_producers() -> None:
    # A terminal can read a payload (transform) on one leg and a filter source
    # on another, linking from both producers.
    spec = _spec(
        [
            _step("simulation", "datagen", {"T": 8}),
            _step("filter", "filter", {}),
            _step("standardize", "std", {"source": "datagen", "field": "observables"}),
            _step(
                "breusch_pagan",
                "bp",
                {
                    "residuals_source": "filter",
                    "residuals_field": "std_innov",
                    "X_source": "std",
                    "X_field": "payload",
                },
            ),
        ]
    )

    pipeline = MCPipeline.from_spec(spec)
    bp = next(step for step in pipeline.replication_steps if step.name == "bp")
    producers = {selector.field: selector.source_step for selector in bp.source_args}
    assert producers["std_innov"] == "filter"
    assert producers["payload"] == "std"


def test_validate_resolves_payload_source_by_producer_name() -> None:
    # A terminal selects a transform's payload by producer name; ordering and
    # validation resolve it from the source reference alone.
    spec = _spec(
        [
            _step("simulation", "datagen", {"T": 8}),
            _step("standardize", "std", {"source": "datagen", "field": "observables"}),
            _step("jarque_bera", "jb", {"source": "std", "field": "payload"}),
        ]
    )
    pipeline = MCPipeline.from_spec(spec)
    assert [step.name for step in pipeline.replication_steps] == [
        "datagen",
        "std",
        "jb",
    ]


def test_validate_orders_payload_source_chain() -> None:
    # Transform chain wired purely by key: tf2 reads tf1's payload, tf1 reads
    # tf0's payload. Ordering must place tf0, tf1, then tf2 from the references.
    spec = _spec(
        [
            _step("simulation", "datagen", {"T": 8}),
            _step("log", "tf2", {"source": "tf1", "field": "payload"}),
            _step("log", "tf1", {"source": "tf0", "field": "payload"}),
            _step("standardize", "tf0", {"source": "datagen", "field": "observables"}),
        ]
    )
    pipeline = MCPipeline.from_spec(spec)
    ordered = [step.name for step in pipeline.replication_steps]
    assert ordered.index("tf0") < ordered.index("tf1") < ordered.index("tf2")


def test_validate_rejects_payload_leg_without_producer() -> None:
    spec = _spec(
        [
            _step("simulation", "datagen", {"T": 8}),
            _step("jarque_bera", "jb", {"source": "ghost", "field": "payload"}),
        ]
    )
    with pytest.raises(ValueError, match="unknown producer"):
        MCPipeline.from_spec(spec)


def test_validate_rejects_filter_source_without_filter_link() -> None:
    spec = _spec(
        [
            _step("simulation", "datagen", {"T": 8}),
            _step("ljung_box", "lb", {"source": "filter", "field": "std_innov"}),
        ]
    )
    with pytest.raises(ValueError, match="unknown producer"):
        MCPipeline.from_spec(spec)


def test_from_spec_stamps_op_types_and_keeps_kwargs_verbatim() -> None:
    spec = _spec(
        [
            _step("simulation", "datagen", {"T": 8}),
            _step(
                "regression",
                "reg",
                {
                    "kind": "ols",
                    "y_source": "datagen",
                    "y_field": "observables",
                    "X_source": "datagen",
                    "X_field": "observables",
                },
            ),
        ]
    )
    pipeline = MCPipeline.from_spec(spec)

    assert [s.name for s in pipeline.replication_steps] == ["datagen", "reg"]
    assert pipeline.replication_steps[0].op_type is OpType.DATAGEN
    assert pipeline.replication_steps[1].op_type is OpType.REGRESSION
    # the step's kwargs reach the step untouched; nothing reshapes them here
    assert pipeline.replication_steps[0].kwargs["T"] == 8
    assert pipeline.replication_steps[1].kwargs["kind"] == "ols"
    # source legs are rebuilt from the step's own bindings
    assert [a.arg for a in pipeline.replication_steps[1].source_args] == ["y", "X"]


# --- POSTPROC (post-loop) kind: ordering, separation, compilation ------------


def test_postprocs_are_a_separate_terminal_list() -> None:
    spec = _spec(
        [
            _step("simulation", "datagen", {"T": 8}),
            _step(
                "jarque_bera",
                "jb",
                {"source": "datagen", "field": "observables", "column": 0},
            ),
        ],
        [_step("kde", "kde", {"trace": "test.jb.statistic"}, func=_run_kde)],
    )
    pipeline = MCPipeline.from_spec(spec)
    # The DAG is per-rep only; postprocs are a separate phase, not ordered in.
    assert [step.name for step in pipeline.replication_steps] == ["datagen", "jb"]
    assert [step.name for step in pipeline.postproc_steps] == ["kde"]


def test_postproc_in_replication_steps_is_rejected() -> None:
    # A postproc smuggled into the replication list is rejected when the
    # pipeline is built. Postprocs are not graph nodes.
    spec = _spec(
        [
            _step("simulation", "sim", {"T": 8}),
            _step("kde", "k", {"trace": "test.jb.statistic"}, func=_run_kde),
        ]
    )
    with pytest.raises(ValueError, match="can't be specified under replication_steps"):
        MCPipeline.from_spec(spec)


def test_postproc_custom_takes_its_callable_from_the_spec() -> None:
    def my_summary(*, traces):
        return 1.0

    spec = _spec(
        [_step("simulation", "datagen", {"T": 8})],
        [_step("postproc:custom", "p", {}, func=my_summary)],
    )
    pipeline = MCPipeline.from_spec(spec)
    step = {s.name: s for s in pipeline.postproc_steps}["p"]
    assert step.op_type is OpType.POSTPROC
    assert step.step_type == "postproc:custom"
    assert step.func is my_summary
    # The authoring source is not a runtime kwarg of the op.
    assert "code" not in step.kwargs and "func_ref" not in step.kwargs


# --- #179 trace registry + POSTPROC trace-reference validation ---------------


def test_available_traces_enumerates_producer_keys() -> None:
    spec = _spec(
        [
            _step("simulation", "sim", {"T": 8}),
            _step("filter", "f", {}),
            _step("standardize", "s", {"source": "sim", "field": "observables"}),
            _step(
                "jarque_bera",
                "jb",
                {"source": "sim", "field": "observables", "column": 0},
            ),
            _step(
                "regression",
                "reg",
                {
                    "y_source": "sim",
                    "y_field": "observables",
                    "X_source": "sim",
                    "X_field": "observables",
                },
            ),
        ]
    )
    assert set(_trace_keys(_meta(spec))) == {
        "test.jb.statistic",
        "test.jb.pval",
        "test.jb.status",
        "regression.reg.coef",
        "regression.reg.ssr",
        "regression.reg.sst",
        "regression.reg.se",
        "regression.reg.r2",
        "regression.reg.status",
        "payload.s",  # transform output
    }
    # datagen / filter produce no consumable trace.


def _kde_spec(trace_params: dict) -> PipelineSpec:
    return _spec(
        [
            _step("simulation", "sim", {"T": 8}),
            _step(
                "jarque_bera",
                "jb",
                {"source": "sim", "field": "observables", "column": 0},
            ),
        ],
        [_step("kde", "k", trace_params, func=_run_kde)],
    )


def test_kde_valid_trace_reference_passes() -> None:
    pipeline = MCPipeline.from_spec(_kde_spec({"trace": "test.jb.statistic"}))
    assert [step.name for step in pipeline.replication_steps] == ["sim", "jb"]
    assert [step.name for step in pipeline.postproc_steps] == ["k"]


def test_kde_bogus_trace_reference_raises_listing_available() -> None:
    with pytest.raises(ValueError, match="no step in the pipeline produces"):
        MCPipeline.from_spec(_kde_spec({"trace": "test.ghost.pval"}))


def test_kde_without_a_trace_cannot_be_constructed() -> None:
    # `kde_step` mirrors `run_kde`'s keywords, so the omission is rejected at
    # construction and never reaches a pipeline to be validated.
    from SymbolicDSGE.monte_carlo.step_factories import kde_step

    with pytest.raises(TypeError, match="trace"):
        kde_step("kde")


def test_postproc_custom_trace_refs_not_statically_validated() -> None:
    # A custom postproc references traces in opaque code; it must validate even
    # though we can't statically know which keys it reads.
    spec = _spec(
        [
            _step("simulation", "sim", {"T": 8}),
            _step(
                "jarque_bera",
                "jb",
                {"source": "sim", "field": "observables", "column": 0},
            ),
        ],
        [_step("postproc:custom", "p", {}, func=lambda **kwargs: {})],
    )
    pipeline = MCPipeline.from_spec(spec)
    assert [step.name for step in pipeline.replication_steps] == ["sim", "jb"]
    assert [step.name for step in pipeline.postproc_steps] == ["p"]
