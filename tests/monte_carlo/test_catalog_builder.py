from __future__ import annotations

from typing import Any

import pytest
import numpy as np

from SymbolicDSGE.kalman.filter import FilterRawResult, UnscentedFilterRawResult
from SymbolicDSGE.monte_carlo import OpType
from SymbolicDSGE.monte_carlo.builder import build_pipeline, run_pipeline
from SymbolicDSGE.monte_carlo.custom_op import NumbaCustomFunc
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
    EdgeSpec,
    NodeSpec,
    PipelineSpec,
    PostprocSpec,
)


def _node(
    id: str,
    step_type: str,
    name: str,
    params: dict[str, Any] | None = None,
) -> NodeSpec:
    """One node, authored flat and resolved the way the client resolves one."""
    return _posted_node(
        {"id": id, "step_type": step_type, "name": name, "params": params or {}}
    )


def _edge(source: str, target: str) -> EdgeSpec:
    return EdgeSpec(source=source, target=target)


def _postproc(name: str, step_type: str, params: dict[str, Any]) -> PostprocSpec:
    return PostprocSpec(name=name, step_type=step_type, params=params)


def _spec(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec] | None = None,
    postprocs: list[PostprocSpec] | None = None,
) -> PipelineSpec:
    return PipelineSpec(nodes=nodes, edges=edges or [], postprocs=postprocs or [])


def _factory_probe_kwargs(step_type: str) -> dict:
    if step_type == "simulation":
        return {"T": 1}
    if step_type == "wald":
        return {"source": "filter", "field": "std_innov", "target": np.zeros(1)}
    if step_type in {"ljung_box", "jarque_bera"}:
        return {"source": "datagen", "field": "observables"}
    if step_type in {"breusch_pagan", "breusch_godfrey"}:
        return {
            "residuals_source": "datagen",
            "residuals_field": "observables",
            "X_source": "datagen",
            "X_field": "observables",
        }
    if step_type in {"cusum", "cusumsq", "chow", "regression"}:
        return {
            "y_source": "datagen",
            "y_field": "observables",
            "X_source": "datagen",
            "X_field": "observables",
        }
    if step_type in {
        "standardize",
        "log",
        "log_diff",
        "diff",
        "rolling_mean",
        "rolling_std",
        "rolling_var",
    }:
        return {"source": "datagen", "field": "observables"}
    return {}


def test_every_step_kind_declares_an_op_kind() -> None:
    # A node states its own op kind and `build_pipeline` holds it to this map,
    # so a kind missing from it could never be built.
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


_FIELD_KEYS = {
    "key",
    "label",
    "type",
    "default",
    "required",
    "options",
    "minimum",
    "when",
}


def test_source_fields_match_the_native_output_channels() -> None:
    assert MC_DATA_SOURCE_FIELDS == ("states", "observables")
    # ``status`` is a scalar error code, not a selectable array source, so it is
    # excluded from the source-field set. Native lowering resolves the layouts,
    # so source fields no longer carry Python-side positional indices.
    linear_array_fields = tuple(f for f in FilterRawResult._fields if f != "status")
    unscented_array_fields = tuple(
        f for f in UnscentedFilterRawResult._fields if f != "status"
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
        nodes=[
            _node("sim", "simulation", "datagen", {"T": 8}),
            _node("filter", "filter", "renamed_filter", {}),
            _node(
                "test",
                "breusch_pagan",
                "diagnostic",
                {
                    "residuals_source": "renamed_filter",
                    "residuals_field": "std_innov",
                    "X_source": "datagen",
                    "X_field": "observables",
                },
            ),
        ],
        edges=[_edge("sim", "filter"), _edge("filter", "test")],
    )

    pipeline = build_pipeline(spec)

    assert [step.name for step in pipeline.per_rep_steps] == [
        "datagen",
        "renamed_filter",
        "diagnostic",
    ]
    residuals = next(
        selector
        for selector in pipeline.per_rep_steps[-1].source_args
        if selector.field == "std_innov"
    )
    assert residuals.source_step == "renamed_filter"


def test_validate_binds_multi_source_terminal_from_distinct_producers() -> None:
    # A terminal can now read a payload (transform) on one leg and a filter
    # source on another, linking from both producers.
    spec = _spec(
        nodes=[
            _node("sim", "simulation", "datagen", {"T": 8}),
            _node("filter", "filter", "filter", {}),
            _node(
                "std",
                "standardize",
                "std",
                {"source": "datagen", "field": "observables"},
            ),
            _node(
                "bp",
                "breusch_pagan",
                "bp",
                {
                    "residuals_source": "filter",
                    "residuals_field": "std_innov",
                    "X_source": "std",
                    "X_field": "payload",
                },
            ),
        ],
        edges=[
            _edge("sim", "filter"),
            _edge("sim", "std"),
            _edge("filter", "bp"),
            _edge("std", "bp"),
        ],
    )

    pipeline = build_pipeline(spec)
    bp = next(step for step in pipeline.per_rep_steps if step.name == "bp")
    producers = {selector.field: selector.source_step for selector in bp.source_args}
    assert producers["std_innov"] == "filter"
    assert producers["payload"] == "std"


def test_validate_resolves_payload_source_without_an_edge() -> None:
    # A terminal selects a transform's payload by producer name with no edge
    # linking them; ordering and validation resolve it from the source reference.
    spec = _spec(
        nodes=[
            _node("sim", "simulation", "datagen", {"T": 8}),
            _node(
                "std",
                "standardize",
                "std",
                {"source": "datagen", "field": "observables"},
            ),
            _node(
                "jb",
                "jarque_bera",
                "jb",
                {"source": "std", "field": "payload"},
            ),
        ],
        edges=[_edge("sim", "std"), _edge("sim", "jb")],
    )
    pipeline = build_pipeline(spec)
    assert [step.name for step in pipeline.per_rep_steps] == ["datagen", "std", "jb"]


def test_validate_orders_payload_source_chain_without_edges() -> None:
    # Transform chain wired purely by key: tf2 reads tf1's payload, tf1 reads
    # tf0's payload. Ordering must place tf0, tf1, then tf2 from the references.
    spec = _spec(
        nodes=[
            _node("sim", "simulation", "datagen", {"T": 8}),
            _node("tf2", "log", "tf2", {"source": "tf1", "field": "payload"}),
            _node("tf1", "log", "tf1", {"source": "tf0", "field": "payload"}),
            _node(
                "tf0",
                "standardize",
                "tf0",
                {"source": "datagen", "field": "observables"},
            ),
        ],
        edges=[_edge("sim", "tf0")],
    )
    pipeline = build_pipeline(spec)
    ordered = [step.name for step in pipeline.per_rep_steps]
    assert ordered.index("tf0") < ordered.index("tf1") < ordered.index("tf2")


def test_validate_rejects_payload_leg_without_producer() -> None:
    spec = _spec(
        nodes=[
            _node("sim", "simulation", "datagen", {"T": 8}),
            _node("jb", "jarque_bera", "jb", {"source": "ghost", "field": "payload"}),
        ],
        edges=[_edge("sim", "jb")],
    )
    with pytest.raises(ValueError, match="unknown producer"):
        build_pipeline(spec)


def test_validate_rejects_filter_source_without_filter_link() -> None:
    spec = _spec(
        nodes=[
            _node("sim", "simulation", "datagen", {"T": 8}),
            _node(
                "test", "ljung_box", "lb", {"source": "filter", "field": "std_innov"}
            ),
        ],
        edges=[_edge("sim", "test")],
    )
    with pytest.raises(ValueError, match="unknown producer"):
        build_pipeline(spec)


def test_run_requires_a_reference_model() -> None:
    # A DGP is only required by the steps that read one, and those say so at
    # lowering; the reference is the run's own precondition.
    spec = _spec(nodes=[_node("sim", "simulation", "datagen", {"T": 4})])
    with pytest.raises(ValueError, match="reference model is required"):
        run_pipeline(spec, reference=None, dgp=None, n_rep=1, fail_fast=True)


def test_build_pipeline_stamps_op_types_and_keeps_params_verbatim() -> None:
    spec = _spec(
        nodes=[
            _node("sim", "simulation", "datagen", {"T": 8}),
            _node(
                "reg",
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
        ],
        edges=[_edge("sim", "reg")],
    )
    pipeline = build_pipeline(spec)

    assert [s.name for s in pipeline.per_rep_steps] == ["datagen", "reg"]
    assert pipeline.per_rep_steps[0].op_type is OpType.DATAGEN
    assert pipeline.per_rep_steps[1].op_type is OpType.REGRESSION
    # the node's params reach the step untouched; nothing reshapes them here
    assert pipeline.per_rep_steps[0].kwargs["T"] == 8
    assert pipeline.per_rep_steps[1].kwargs["kind"] == "ols"
    # source legs are rebuilt from the node's own bindings
    assert [a.arg for a in pipeline.per_rep_steps[1].source_args] == ["y", "X"]


def test_build_pipeline_rejects_unknown_step_type() -> None:
    # NodeSpec is a TypedDict and validates nothing, so a bogus kind reaches
    # build_pipeline and must be rejected there.
    node = _node(id="x", step_type="bogus", name="x", params={})
    with pytest.raises(ValueError, match="Unsupported MC step type"):
        build_pipeline(_spec(nodes=[node]))


# --- POSTPROC (post-loop) kind: ordering, edges, compilation -----------------


def test_postprocs_are_a_separate_terminal_list() -> None:
    spec = _spec(
        nodes=[
            _node("sim", "simulation", "datagen", {"T": 8}),
            _node(
                "jb",
                "jarque_bera",
                "jb",
                {"source": "datagen", "field": "observables", "column": 0},
            ),
        ],
        edges=[_edge("sim", "jb")],
        postprocs=[_postproc("kde", "kde", {"trace": "test.jb.statistic"})],
    )
    pipeline = build_pipeline(spec)
    # The DAG is per-rep only; postprocs are a separate phase, not ordered in.
    assert [step.name for step in pipeline.per_rep_steps] == ["datagen", "jb"]
    assert [step.name for step in pipeline.postproc_steps] == ["kde"]


def test_postproc_in_nodes_is_rejected() -> None:
    # A postproc smuggled into `nodes` (rather than `postprocs`) is rejected when
    # the pipeline is built. Postprocs are not graph nodes.
    spec = _spec(
        nodes=[
            _node("sim", "simulation", "sim", {"T": 8}),
            _node("k", "kde", "k", {"trace": "test.jb.statistic"}),
        ]
    )
    with pytest.raises(ValueError, match="can't be specified under per_rep_steps"):
        build_pipeline(spec)


def test_build_postproc_custom_from_resources() -> None:
    def my_summary(*, traces):
        return 1.0

    spec = _spec(
        nodes=[
            _node("sim", "simulation", "datagen", {"T": 8}),
        ],
        edges=[],
        postprocs=[_postproc("p", "postproc:custom", {"func_ref": "p", "code": "..."})],
    )
    pipeline = build_pipeline(spec, resources={"p": my_summary})
    step = {s.name: s for s in pipeline.postproc_steps}["p"]
    assert step.op_type is OpType.POSTPROC
    assert step.step_type == "postproc:custom"
    assert "code" not in step.kwargs and "func_ref" not in step.kwargs


# --- #179 trace registry + POSTPROC trace-reference validation ---------------

from SymbolicDSGE.monte_carlo.traces import _trace_keys
from tests._spec_helpers import _posted_node


def test_available_traces_enumerates_producer_keys() -> None:
    spec = _spec(
        nodes=[
            _node("sim", "simulation", "sim", {"T": 8}),
            _node("f", "filter", "f", {}),
            _node(
                "s",
                "standardize",
                "s",
                {"source": "sim", "field": "observables"},
            ),
            _node(
                "jb",
                "jarque_bera",
                "jb",
                {"source": "sim", "field": "observables", "column": 0},
            ),
            _node(
                "reg",
                "regression",
                "reg",
                {
                    "y_source": "sim",
                    "y_field": "observables",
                    "X_source": "sim",
                    "X_field": "observables",
                },
            ),
        ],
        edges=[],
    )
    assert set(_trace_keys(spec)) == {
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
        nodes=[
            _node("sim", "simulation", "sim", {"T": 8}),
            _node(
                "jb",
                "jarque_bera",
                "jb",
                {"source": "sim", "field": "observables", "column": 0},
            ),
        ],
        edges=[_edge("sim", "jb")],
        postprocs=[_postproc("k", "kde", trace_params)],
    )


def test_kde_valid_trace_reference_passes() -> None:
    pipeline = build_pipeline(_kde_spec({"trace": "test.jb.statistic"}))
    assert [step.name for step in pipeline.per_rep_steps] == ["sim", "jb"]
    assert [step.name for step in pipeline.postproc_steps] == ["k"]


def test_kde_bogus_trace_reference_raises_listing_available() -> None:
    with pytest.raises(ValueError, match="no step in the pipeline produces"):
        build_pipeline(_kde_spec({"trace": "test.ghost.pval"}))


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
        nodes=[
            _node("sim", "simulation", "sim", {"T": 8}),
            _node(
                "jb",
                "jarque_bera",
                "jb",
                {"source": "sim", "field": "observables", "column": 0},
            ),
        ],
        edges=[_edge("sim", "jb")],
        postprocs=[_postproc("p", "postproc:custom", {"func_ref": "p", "code": "..."})],
    )
    pipeline = build_pipeline(spec, resources={"p": lambda **kwargs: {}})
    assert [step.name for step in pipeline.per_rep_steps] == ["sim", "jb"]
    assert [step.name for step in pipeline.postproc_steps] == ["p"]
