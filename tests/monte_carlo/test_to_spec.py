from __future__ import annotations

import json

import numpy as np
import pytest

from SymbolicDSGE.core.shock_generators import Shock
from SymbolicDSGE.monte_carlo import (
    EdgeSpec,
    MCPipeline,
    PipelineSpec,
    build_pipeline,
    validate_pipeline_spec,
)
from SymbolicDSGE.monte_carlo.operations.core import (
    raw_model_data_step,
    reference_filter_step,
    simulation_step,
)
from SymbolicDSGE.monte_carlo.operations.tests import (
    jarque_bera_test_step,
    wald_test_step,
)
from SymbolicDSGE.monte_carlo.operations.transforms import (
    standardize_step,
    transform_step,
)


def _simulation_pipeline() -> MCPipeline:
    return MCPipeline(
        [
            simulation_step(
                "dgp",
                T=8,
                observables=True,
                seed_increment="auto",
                shocks={"u": Shock(dist="norm", seed=0, dist_kwargs={"loc": 0.0})},
            ),
            reference_filter_step("filter"),
            standardize_step("s", source="dgp", field="observables"),
            jarque_bera_test_step("jb", source="s", field="payload"),
            wald_test_step(
                "w",
                source="filter",
                field="std_innov",
                kind="mean",
                target=np.array([0.0]),
                bandwidth=4,
            ),
        ]
    )


def test_to_spec_structure_and_edges() -> None:
    spec = _simulation_pipeline().to_spec()

    assert [n.step_type for n in spec.nodes] == [
        "simulation",
        "filter",
        "standardize",
        "jarque_bera",
        "wald",
    ]
    assert {(e.source, e.target) for e in spec.edges} == {
        ("dgp", "filter"),
        ("dgp", "s"),
        ("s", "jb"),
        ("filter", "w"),
    }

    by_name = {n.name: n for n in spec.nodes}
    assert by_name["jb"].params["source"] == "s"
    assert by_name["jb"].params["field"] == "payload"
    assert by_name["w"].params["source"] == "filter"
    assert by_name["w"].params["field"] == "std_innov"
    # wald target ndarray is inverted to the spec's target_vector field
    assert by_name["w"].params["target_vector"] == [0.0]
    assert "target" not in by_name["w"].params
    # shocks are serialized to JSON-safe dicts
    assert by_name["dgp"].params["shocks"]["u"]["dist"] == "norm"
    json.dumps(spec.to_dict())


def test_to_spec_is_a_fixed_point_under_rebuild() -> None:
    pipe = _simulation_pipeline()
    spec1 = pipe.to_spec()

    ordered, postprocs = validate_pipeline_spec(spec1, has_reference=True, has_dgp=True)
    rebuilt = build_pipeline(ordered, postprocs)

    spec2 = rebuilt.to_spec()
    assert spec2 == spec1


def test_to_spec_rejects_shock_generators_with_actionable_message() -> None:
    # `.shock_generator()` returns an opaque callable the runtime accepts but
    # that cannot be serialized; to_spec must say how to fix it.
    pipe = MCPipeline(
        [
            simulation_step(
                "dgp",
                T=8,
                shocks={"u": Shock(dist="norm", seed=0).shock_generator(8)},
            ),
            jarque_bera_test_step("jb", source="dgp", field="observables"),
        ]
    )
    with pytest.raises(TypeError, match="shock generator"):
        pipe.to_spec()


def test_rebuilt_simulation_recovers_live_shocks() -> None:
    pipe = _simulation_pipeline()
    ordered, postprocs = validate_pipeline_spec(
        pipe.to_spec(), has_reference=True, has_dgp=True
    )
    rebuilt = build_pipeline(ordered, postprocs)

    shock = rebuilt.per_rep_steps[0].kwargs["shocks"]["u"]
    assert isinstance(shock, Shock)
    assert shock.to_dict() == pipe.per_rep_steps[0].kwargs["shocks"]["u"].to_dict()


def test_to_spec_records_raw_model_data_reference_not_arrays() -> None:
    states = np.zeros((4, 5, 2))
    observables = np.zeros((4, 5, 3))
    pipe = MCPipeline(
        [
            raw_model_data_step(
                "dat",
                states=states,
                observables=observables,
                observable_names=("a", "b", "c"),
            ),
            jarque_bera_test_step("jb", source="dat", field="observables"),
        ]
    )
    spec = pipe.to_spec()

    dat = spec.nodes[0]
    assert dat.step_type == "raw_model_data"
    assert dat.params["data_ref"] == "dat"
    assert dat.params["data_shapes"] == {
        "states": [4, 5, 2],
        "observables": [4, 5, 3],
    }
    assert dat.params["observable_names"] == ["a", "b", "c"]
    assert {(e.source, e.target) for e in spec.edges} == {("dat", "jb")}
    # No raw arrays leak into the JSON spec.
    json.dumps(spec.to_dict())
    assert PipelineSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_to_spec_emits_custom_with_func_ref() -> None:
    pipe = MCPipeline(
        [
            raw_model_data_step("dat", observables=np.zeros((4, 5, 3))),
            transform_step("tf", lambda **_: None),
        ]
    )
    spec = pipe.to_spec()

    tf = {n.name: n for n in spec.nodes}["tf"]
    assert tf.step_type == "transform:custom"
    # the callable rides a separate bundle member; the spec only references it
    assert tf.params["func_ref"] == "tf"
    assert "source" not in tf.params
    assert "field" not in tf.params
    assert {(e.source, e.target) for e in spec.edges} == set()


def test_to_spec_emits_postproc_custom_with_func_ref_and_kwargs() -> None:
    from SymbolicDSGE.monte_carlo.operations.postproc import postproc_step

    def my_summary(*, traces, reference, dgp, threshold):
        return float(threshold)

    pipe = MCPipeline(
        [
            raw_model_data_step("dat", observables=np.zeros((4, 5, 3))),
            jarque_bera_test_step("jb", source="dat", field="observables"),
        ],
        [postproc_step("sum", my_summary, threshold=0.5)],
    )
    spec = pipe.to_spec()

    pp = {p.name: p for p in spec.postprocs}["sum"]
    assert pp.step_type == "postproc:custom"
    # callable rides a bundle member; op kwargs survive as plain spec params
    assert pp.params["func_ref"] == "sum"
    assert pp.params["threshold"] == 0.5
    # postprocs are a separate list, never nodes or edges.
    assert "sum" not in {n.name for n in spec.nodes}
    assert all(e.source != "sum" and e.target != "sum" for e in spec.edges)


def test_to_spec_round_trips_a_postproc_pipeline() -> None:
    from SymbolicDSGE.monte_carlo.operations.postproc import kde_step

    pipe = MCPipeline(
        [
            simulation_step(
                "dgp",
                T=8,
                observables=True,
                seed_increment="auto",
                shocks={"u": Shock(dist="norm", seed=0)},
            ),
            jarque_bera_test_step("jb", source="dgp", field="observables", column=0),
        ],
        [kde_step("kde", trace="test.jb.statistic", grid_points=50)],
    )
    spec1 = pipe.to_spec()
    kde_pp = {p.name: p for p in spec1.postprocs}["kde"]
    assert kde_pp.step_type == "kde"
    assert kde_pp.params["trace"] == "test.jb.statistic"
    # postprocs are a separate list, never nodes or edges.
    assert "kde" not in {n.name for n in spec1.nodes}

    ordered, postprocs = validate_pipeline_spec(spec1, has_reference=True, has_dgp=True)
    assert [n.id for n in ordered] == ["dgp", "jb"]
    assert [p.name for p in postprocs] == ["kde"]
    rebuilt = build_pipeline(ordered, postprocs)
    assert rebuilt.to_spec() == spec1  # fixed point
