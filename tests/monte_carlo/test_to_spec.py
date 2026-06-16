from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from SymbolicDSGE.core.shock_generators import Shock
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import (
    EdgeSpec,
    MCPipeline,
    PipelineSpec,
    build_pipeline,
    validate_pipeline_spec,
)
from SymbolicDSGE.monte_carlo.operations.core import (
    raw_data_step,
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
                shocks={"u": Shock(T=8, dist="norm", seed=0, dist_kwargs={"loc": 0.0})},
            ),
            reference_filter_step("filter"),
            standardize_step("s", source="observables"),
            jarque_bera_test_step("jb", source="payload", payload_key="s"),
            wald_test_step(
                "w",
                source="std_innov",
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
    # payload references are kept (the producer is named by key, not an edge)
    assert by_name["jb"].params["payload_key"] == "s"
    # filter_key stays edge-derived (re-bound from the filter edge on rebuild)
    assert "filter_key" not in by_name["w"].params
    # wald target ndarray is inverted to the spec's target_vector field
    assert by_name["w"].params["target_vector"] == [0.0]
    assert "target" not in by_name["w"].params
    # shocks are serialized to JSON-safe dicts
    assert by_name["dgp"].params["shocks"]["u"]["dist"] == "norm"
    json.dumps(spec.to_dict())


def test_to_spec_is_a_fixed_point_under_rebuild() -> None:
    pipe = _simulation_pipeline()
    spec1 = pipe.to_spec()

    stub_dgp = cast(SolvedModel, SimpleNamespace())
    ordered = validate_pipeline_spec(spec1, has_reference=True, has_dgp=True)
    rebuilt = build_pipeline(ordered, dgp=stub_dgp)

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
                shocks={"u": Shock(T=8, dist="norm", seed=0).shock_generator()},
            ),
            jarque_bera_test_step("jb", source="observables"),
        ]
    )
    with pytest.raises(TypeError, match="shock generator"):
        pipe.to_spec()


def test_rebuilt_simulation_recovers_live_shocks() -> None:
    pipe = _simulation_pipeline()
    stub_dgp = cast(SolvedModel, SimpleNamespace())
    ordered = validate_pipeline_spec(pipe.to_spec(), has_reference=True, has_dgp=True)
    rebuilt = build_pipeline(ordered, dgp=stub_dgp)

    shock = rebuilt.steps[0].kwargs["shocks"]["u"]
    assert isinstance(shock, Shock)
    assert shock.to_dict() == pipe.steps[0].kwargs["shocks"]["u"].to_dict()


def test_to_spec_records_raw_data_reference_not_arrays() -> None:
    states = np.zeros((4, 5, 2))
    observables = np.zeros((4, 5, 3))
    pipe = MCPipeline(
        [
            raw_data_step(
                "dat",
                states=states,
                observables=observables,
                n_exog=1,
                observable_names=("a", "b", "c"),
            ),
            jarque_bera_test_step("jb", source="observables"),
        ]
    )
    spec = pipe.to_spec()

    dat = spec.nodes[0]
    assert dat.step_type == "raw_data"
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
            raw_data_step("dat", observables=np.zeros((4, 5, 3))),
            transform_step("tf", lambda **_: None, source="observables"),
        ]
    )
    spec = pipe.to_spec()

    tf = {n.name: n for n in spec.nodes}["tf"]
    assert tf.step_type == "custom"
    # the callable rides a separate bundle member; the spec only references it
    assert tf.params["func_ref"] == "tf"
    assert tf.params["source"] == "observables"
    assert {(e.source, e.target) for e in spec.edges} == {("dat", "tf")}
