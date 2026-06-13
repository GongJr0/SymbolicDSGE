from __future__ import annotations

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
from SymbolicDSGE.monte_carlo.spec import EdgeSpec, NodeSpec, PipelineSpec

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
    assert len(steps) == len(STEP_CATALOG) == 11

    for step in steps:
        assert set(step) == {
            "step_type",
            "title",
            "default_name",
            "description",
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

    ordered = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)

    assert [node.id for node in ordered] == ["sim", "filter", "test"]
    assert ordered[-1].params["filter_key"] == "renamed_filter"


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
            NodeSpec("sim", "simulation", "datagen", {"T": 8, "observables": True}),
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
    ordered = validate_pipeline_spec(spec, has_reference=True, has_dgp=True)

    pipeline = build_pipeline(ordered, dgp=_stub_dgp("u"))

    assert [s.name for s in pipeline.steps] == ["datagen", "reg"]
    assert pipeline.steps[0].op_type is OpType.DATAGEN
    assert pipeline.steps[1].op_type is OpType.REGRESSION
    # simulation got generated shocks injected
    assert "shocks" in pipeline.steps[0].kwargs
    # OLS regression dropped the conditional alpha kwarg
    assert "alpha" not in pipeline.steps[1].kwargs


def test_build_pipeline_rejects_unknown_step_type() -> None:
    # NodeSpec.__init__ does not validate step_type (only from_dict does), so a
    # bogus kind reaches build_pipeline and must be rejected there.
    node = NodeSpec(id="x", step_type="bogus", name="x", params={})
    with pytest.raises(ValueError, match="Unsupported MC step type"):
        build_pipeline([node], dgp=_stub_dgp("u"))
