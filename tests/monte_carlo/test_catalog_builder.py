from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from sympy import Symbol

from SymbolicDSGE.monte_carlo import (
    STEP_CATALOG,
    TERMINAL_STEP_TYPES,
    OpType,
    build_pipeline,
    catalog_payload,
    validate_pipeline_spec,
)
from SymbolicDSGE.monte_carlo.catalog import _shocks_from_registry
from SymbolicDSGE.monte_carlo.operations.transforms import transform_step
from SymbolicDSGE.monte_carlo.spec import (
    STEP_KINDS,
    EdgeSpec,
    NodeSpec,
    PipelineSpec,
    PostprocSpec,
)


def test_factories_stamp_step_type_matching_catalog() -> None:
    # Every catalog factory must stamp the same step_type as its catalog key,
    # so a live MCPipeline can be compiled back to a PipelineSpec losslessly.
    for step_type, definition in STEP_CATALOG.items():
        step = definition.factory(name="probe")
        assert step.step_type == step_type


def test_catalog_keys_are_all_valid_step_kinds() -> None:
    assert set(STEP_CATALOG) <= STEP_KINDS


# Source leg -> the payload-key kwarg the binder produces for it. Each step's
# run op must accept the key for every source leg it declares, else a key-based
# (or transform-fed) payload arrives as an unexpected kwarg at run time.
_SOURCE_LEG_TO_PAYLOAD_KEY = {
    "source": "payload_key",
    "residual_source": "residual_payload_key",
    "y_source": "y_payload_key",
    "X_source": "x_payload_key",
    "x_source": "x_payload_key",
}


def test_every_source_leg_op_accepts_its_payload_key() -> None:
    for step_type, definition in STEP_CATALOG.items():
        step = definition.factory(name="probe")
        op_params = set(inspect.signature(step.func).parameters)
        for field in definition.fields:
            payload_key = _SOURCE_LEG_TO_PAYLOAD_KEY.get(field.key)
            if payload_key is None:
                continue
            assert payload_key in op_params, (
                f"'{step_type}' run op does not accept '{payload_key}' for leg "
                f"'{field.key}'; a key-based payload would be an unexpected kwarg."
            )


def test_transform_step_stamps_custom_kind() -> None:
    step = transform_step("tf", lambda **_: None)
    assert step.step_type == "transform:custom"


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


def _stub_dgp(*targets: str) -> SimpleNamespace:
    shock_map = {Symbol(f"e_{t}"): Symbol(t) for t in targets}
    return SimpleNamespace(config=SimpleNamespace(shock_map=shock_map))


def test_catalog_payload_shape_and_known_fields() -> None:
    payload = catalog_payload()
    steps = payload["steps"]
    assert len(steps) == len(STEP_CATALOG) == 19

    for step in steps:
        assert set(step) == {
            "step_type",
            "title",
            "default_name",
            "description",
            "category",
            "fields",
        }
        for field in step["fields"]:
            assert set(field) == _FIELD_KEYS

    by_type = {step["step_type"]: step for step in steps}
    sim_fields = {f["key"]: f for f in by_type["simulation"]["fields"]}
    assert sim_fields["T"]["default"] == 100
    assert sim_fields["T"]["required"] is True
    assert sim_fields["T"]["minimum"] == 1

    wald_fields = {f["key"]: f for f in by_type["wald"]["fields"]}
    assert wald_fields["target_vector"]["when"] == ["mean"]
    assert wald_fields["source"]["options"][0] == "states"


def test_catalog_entries_carry_selector_category() -> None:
    by_type = {s["step_type"]: s for s in catalog_payload()["steps"]}
    assert by_type["simulation"]["category"] == "core"
    assert by_type["filter"]["category"] == "core"
    assert by_type["standardize"]["category"] == "transforms"
    assert by_type["jarque_bera"]["category"] == "tests"
    assert by_type["wald"]["category"] == "tests"
    assert by_type["regression"]["category"] == "regressions"
    assert by_type["kde"]["category"] == "postproc"
    assert {s["category"] for s in by_type.values()} == {
        "core",
        "transforms",
        "tests",
        "regressions",
        "postproc",
    }


def test_terminal_step_types_derived_from_catalog() -> None:
    assert TERMINAL_STEP_TYPES == frozenset(
        {
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
    )


def test_validate_orders_steps_and_binds_filter_key() -> None:
    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "datagen", {"T": 8, "observables": True}),
            NodeSpec("filter", "filter", "renamed_filter", {}),
            NodeSpec(
                "test",
                "breusch_pagan",
                "diagnostic",
                {"residual_source": "std_innov", "X_source": "observables"},
            ),
        ],
        edges=[EdgeSpec("sim", "filter"), EdgeSpec("filter", "test")],
    )

    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)

    assert [node.id for node in ordered] == ["sim", "filter", "test"]
    assert ordered[-1].params["filter_key"] == "renamed_filter"


def test_validate_binds_multi_source_terminal_from_distinct_producers() -> None:
    # A terminal can now read a payload (transform) on one leg and a filter
    # source on another, linking from both producers.
    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "datagen", {"T": 8, "observables": True}),
            NodeSpec("filter", "filter", "filter", {}),
            NodeSpec("std", "standardize", "std", {"source": "observables"}),
            NodeSpec(
                "bp",
                "breusch_pagan",
                "bp",
                {"residual_source": "std_innov", "X_source": "payload"},
            ),
        ],
        edges=[
            EdgeSpec("sim", "filter"),
            EdgeSpec("sim", "std"),
            EdgeSpec("filter", "bp"),
            EdgeSpec("std", "bp"),
        ],
    )

    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    bp = next(node for node in ordered if node.id == "bp")
    assert bp.params["filter_key"] == "filter"  # filter leg bound from the filter
    assert bp.params["x_payload_key"] == "std"  # payload leg bound from transform


def test_validate_resolves_payload_by_key_without_an_edge() -> None:
    # A terminal selects a transform's payload by key (payload_key) with no edge
    # linking them; ordering + validation resolve it from the reference.
    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "datagen", {"T": 8, "observables": True}),
            NodeSpec("std", "standardize", "std", {"source": "observables"}),
            NodeSpec(
                "jb",
                "jarque_bera",
                "jb",
                {"source": "payload", "payload_key": "std"},
            ),
        ],
        edges=[EdgeSpec("sim", "std"), EdgeSpec("sim", "jb")],
    )
    ordered, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    assert [node.id for node in ordered] == ["sim", "std", "jb"]


def test_validate_orders_payload_key_chain_without_edges() -> None:
    # Transform chain wired purely by key: tf2 reads tf1's payload, tf1 reads
    # tf0's payload. Ordering must place tf0 -> tf1 -> tf2 from the references.
    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "datagen", {"T": 8, "observables": True}),
            NodeSpec("tf2", "log", "tf2", {"source": "payload", "payload_key": "tf1"}),
            NodeSpec("tf1", "log", "tf1", {"source": "payload", "payload_key": "tf0"}),
            NodeSpec("tf0", "standardize", "tf0", {"source": "observables"}),
        ],
        edges=[EdgeSpec("sim", "tf0")],
    )
    ordered_nodes, _ = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    ordered = [node.id for node in ordered_nodes]
    assert ordered.index("tf0") < ordered.index("tf1") < ordered.index("tf2")


def test_validate_rejects_payload_leg_without_producer() -> None:
    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "datagen", {"T": 8, "observables": True}),
            NodeSpec("jb", "jarque_bera", "jb", {"source": "payload"}),
        ],
        edges=[EdgeSpec("sim", "jb")],
    )
    with pytest.raises(ValueError, match="no .*producer is selected"):
        validate_pipeline_spec(spec, has_reference=True, has_dgp=True)


def test_validate_rejects_filter_source_without_filter_link() -> None:
    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "datagen", {"T": 8, "observables": True}),
            NodeSpec("test", "ljung_box", "lb", {"source": "std_innov"}),
        ],
        edges=[EdgeSpec("sim", "test")],
    )
    with pytest.raises(ValueError, match="must link from a filter"):
        validate_pipeline_spec(spec, has_reference=True, has_dgp=True)


def test_validate_requires_reference_and_dgp() -> None:
    spec = PipelineSpec(nodes=[NodeSpec("sim", "simulation", "datagen", {"T": 4})])
    with pytest.raises(ValueError, match="reference model is required"):
        validate_pipeline_spec(spec, has_reference=False, has_dgp=True)
    with pytest.raises(ValueError, match="DGP model is required"):
        validate_pipeline_spec(spec, has_reference=True, has_dgp=False)


def test_build_pipeline_compiles_via_catalog_and_filters_regression_kwargs() -> None:
    spec = PipelineSpec(
        nodes=[
            NodeSpec(
                "sim",
                "simulation",
                "datagen",
                {
                    "T": 8,
                    "observables": True,
                    "shock_registry": [{"vars": ["u"], "dist": "norm"}],
                },
            ),
            NodeSpec(
                "reg",
                "regression",
                "reg",
                {
                    "kind": "ols",
                    "y_source": "observables",
                    "X_source": "observables",
                    "alpha": 0.9,  # conditional kwarg, invalid for OLS -> dropped
                },
            ),
        ],
        edges=[EdgeSpec("sim", "reg")],
    )
    ordered, postprocs = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)

    pipeline = build_pipeline(ordered, postprocs, dgp=_stub_dgp("u"))

    assert [s.name for s in pipeline.per_rep_steps] == ["datagen", "reg"]
    assert pipeline.per_rep_steps[0].op_type is OpType.DATAGEN
    assert pipeline.per_rep_steps[1].op_type is OpType.REGRESSION
    # simulation compiled the shock registry into a live shocks mapping
    assert "shocks" in pipeline.per_rep_steps[0].kwargs
    # OLS regression dropped the conditional alpha kwarg
    assert "alpha" not in pipeline.per_rep_steps[1].kwargs


def test_build_pipeline_rejects_unknown_step_type() -> None:
    # NodeSpec.__init__ does not validate step_type (only from_dict does), so a
    # bogus kind reaches build_pipeline and must be rejected there.
    node = NodeSpec(id="x", step_type="bogus", name="x", params={})
    with pytest.raises(ValueError, match="Unsupported MC step type"):
        build_pipeline([node], dgp=_stub_dgp("u"))


# --- POSTPROC (post-loop) kind: ordering, edges, compilation -----------------


def test_postprocs_are_a_separate_terminal_list() -> None:
    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "datagen", {"T": 8, "observables": True}),
            NodeSpec("jb", "jarque_bera", "jb", {"source": "observables", "column": 0}),
        ],
        edges=[EdgeSpec("sim", "jb")],
        postprocs=[PostprocSpec("kde", "kde", {"trace": "test.jb.statistic"})],
    )
    ordered, postprocs = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    # The DAG is per-rep only; postprocs are returned separately, not ordered in.
    assert [n.id for n in ordered] == ["sim", "jb"]
    assert [pp.name for pp in postprocs] == ["kde"]


def test_from_dict_rejects_postproc_in_nodes() -> None:
    # A postproc smuggled into `nodes` (rather than `postprocs`) is rejected at
    # deserialization -- postprocs are not graph nodes.
    with pytest.raises(ValueError, match="must be listed under 'postprocs'"):
        PipelineSpec.from_dict(
            {
                "nodes": [
                    {
                        "id": "sim",
                        "step_type": "simulation",
                        "name": "sim",
                        "params": {},
                    },
                    {"id": "k", "step_type": "kde", "name": "k", "params": {}},
                ],
                "edges": [],
            }
        )


def test_kde_catalog_entry_is_postproc_with_trace_field() -> None:
    kde = STEP_CATALOG["kde"]
    assert kde.op_role == "postproc"
    assert kde.category == "postproc"
    field_keys = [f.key for f in kde.fields]
    assert "trace" in field_keys
    # the trace selector must NOT use a per-rep source-leg name
    assert "source" not in field_keys


def test_build_postproc_custom_from_resources() -> None:
    def my_summary(*, traces, reference, dgp):
        return 1.0

    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "datagen", {"T": 8, "observables": True}),
        ],
        edges=[],
        postprocs=[
            PostprocSpec("p", "postproc:custom", {"func_ref": "p", "code": "..."})
        ],
    )
    ordered, postprocs = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    pipeline = build_pipeline(
        ordered, postprocs, dgp=_stub_dgp("u"), resources={"p": my_summary}
    )
    step = {s.name: s for s in pipeline.postproc_steps}["p"]
    assert step.op_type is OpType.POSTPROC
    assert step.step_type == "postproc:custom"
    assert "code" not in step.kwargs and "func_ref" not in step.kwargs


# --- #179 trace registry + POSTPROC trace-reference validation ---------------

from SymbolicDSGE.monte_carlo.traces import available_traces  # noqa: E402


def test_available_traces_enumerates_producer_keys() -> None:
    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "sim", {"T": 8, "observables": True}),
            NodeSpec("f", "filter", "f", {}),
            NodeSpec("s", "standardize", "s", {"source": "observables"}),
            NodeSpec("jb", "jarque_bera", "jb", {"source": "observables", "column": 0}),
            NodeSpec(
                "reg",
                "regression",
                "reg",
                {"y_source": "observables", "X_source": "observables"},
            ),
        ],
        edges=[],
    )
    assert set(available_traces(spec)) == {
        "test.jb.statistic",
        "test.jb.pval",
        "test.jb.status",
        "regression.reg.coef",
        "regression.reg.r2",
        "regression.reg.status",
        "payload.s",  # transform output
    }
    # datagen / filter produce no consumable trace.


def _kde_spec(trace_params: dict) -> PipelineSpec:
    return PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "sim", {"T": 8, "observables": True}),
            NodeSpec("jb", "jarque_bera", "jb", {"source": "observables", "column": 0}),
        ],
        edges=[EdgeSpec("sim", "jb")],
        postprocs=[PostprocSpec("k", "kde", trace_params)],
    )


def test_kde_valid_trace_reference_passes() -> None:
    ordered, postprocs = validate_pipeline_spec(
        _kde_spec({"trace": "test.jb.statistic"}), has_reference=True, has_dgp=True
    )
    assert [n.id for n in ordered] == ["sim", "jb"]
    assert [pp.name for pp in postprocs] == ["k"]


def test_kde_bogus_trace_reference_raises_listing_available() -> None:
    with pytest.raises(ValueError, match="no step in the pipeline produces"):
        validate_pipeline_spec(
            _kde_spec({"trace": "test.ghost.pval"}), has_reference=True, has_dgp=True
        )


def test_kde_missing_trace_reference_raises() -> None:
    with pytest.raises(ValueError, match="must select a trace"):
        validate_pipeline_spec(_kde_spec({}), has_reference=True, has_dgp=True)


def test_postproc_custom_trace_refs_not_statically_validated() -> None:
    # A custom postproc references traces in opaque code; it must validate even
    # though we can't statically know which keys it reads.
    spec = PipelineSpec(
        nodes=[
            NodeSpec("sim", "simulation", "sim", {"T": 8, "observables": True}),
            NodeSpec("jb", "jarque_bera", "jb", {"source": "observables", "column": 0}),
        ],
        edges=[EdgeSpec("sim", "jb")],
        postprocs=[
            PostprocSpec("p", "postproc:custom", {"func_ref": "p", "code": "..."})
        ],
    )
    ordered, postprocs = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)
    assert [n.id for n in ordered] == ["sim", "jb"]
    assert [pp.name for pp in postprocs] == ["p"]


def test_kde_trace_field_is_typed_trace() -> None:
    trace_field = next(f for f in STEP_CATALOG["kde"].fields if f.key == "trace")
    assert trace_field.type == "trace"


# --- shock registry compile -------------------------------------------------
# The simulation compile hook builds shocks from an explicit registry with no
# model involved: one entry compiles to one Shock, joint iff it selects more
# than one variable.


def test_registry_norm_univariate_shapes_loc_kwarg() -> None:
    shocks = _shocks_from_registry([{"vars": ["u"], "dist": "norm", "loc": 0.5}], T=8)
    assert shocks is not None
    shock = shocks["u"]
    assert shock.multivar is False
    assert shock.dist == "norm"
    assert shock.dist_kwargs == {"loc": 0.5}


def test_registry_norm_multivariate_shapes_mean_kwarg() -> None:
    shocks = _shocks_from_registry(
        [{"vars": ["u", "v"], "dist": "norm", "loc": 0.0}], T=8
    )
    assert shocks is not None
    shock = shocks["u,v"]
    assert shock.multivar is True
    assert shock.dist_kwargs == {"mean": [0.0, 0.0]}


def test_registry_t_carries_df() -> None:
    shocks = _shocks_from_registry(
        [{"vars": ["u"], "dist": "t", "loc": 0.0, "df": 7.0}], T=8
    )
    assert shocks is not None
    assert shocks["u"].dist_kwargs == {"loc": 0.0, "df": 7.0}


def test_registry_uniform_single_variable_ok() -> None:
    shocks = _shocks_from_registry([{"vars": ["u"], "dist": "uni"}], T=8)
    assert shocks is not None
    assert shocks["u"].multivar is False
    assert shocks["u"].dist == "uni"


def test_registry_uniform_multiple_variables_raises() -> None:
    with pytest.raises(ValueError, match="univariate"):
        _shocks_from_registry([{"vars": ["u", "v"], "dist": "uni"}], T=8)


def test_registry_empty_compiles_to_none() -> None:
    assert _shocks_from_registry([], T=8) is None


def test_registry_entry_without_variables_raises() -> None:
    with pytest.raises(ValueError, match="at least one variable"):
        _shocks_from_registry([{"vars": [], "dist": "norm"}], T=8)


def test_registry_duplicate_key_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate shock entry"):
        _shocks_from_registry(
            [{"vars": ["u"], "dist": "norm"}, {"vars": ["u"], "dist": "t"}], T=8
        )


def test_registry_unsupported_distribution_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported shock distribution"):
        _shocks_from_registry([{"vars": ["u"], "dist": "cauchy"}], T=8)
