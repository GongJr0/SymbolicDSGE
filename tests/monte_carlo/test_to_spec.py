from __future__ import annotations

import json

import numpy as np
import pytest

from SymbolicDSGE.core.shock.generators import Shock
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.custom_op import NumbaCustomFunc
from SymbolicDSGE.monte_carlo.spec import pipeline_meta
from SymbolicDSGE.monte_carlo.step_factories import (
    jarque_bera_test_step,
    raw_model_data_step,
    filter_step,
    simulation_step,
    standardize_step,
    transform_step,
    wald_test_step,
)


def _copy_transform(sample: np.ndarray, output: np.ndarray) -> int:
    output[:, :] = sample
    return 0


def _simulation_pipeline() -> MCPipeline:
    return MCPipeline(
        [
            simulation_step(
                "dgp",
                target="dgp",
                T=8,
                observables=True,
                shocks={"u": Shock(dist="norm", seed=0, loc=0.0)},
            ),
            filter_step(
                "filter", target="reference", obs_source="dgp", obs_field="observables"
            ),
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


def test_to_spec_structure_and_sources() -> None:
    spec = _simulation_pipeline().to_spec()

    assert [step.meta["step_type"] for step in spec.steps] == [
        "simulation",
        "filter",
        "standardize",
        "jarque_bera",
        "wald",
    ]

    by_name = {step.meta["name"]: step.meta for step in spec.steps}
    # source legs are their own objects, not flattened into kwargs
    assert by_name["jb"]["source_args"] == [
        {
            "arg": "sample",
            "source_step": "s",
            "field": "payload",
            "columns": None,
            "burn_in": 0,
        }
    ]
    assert by_name["w"]["source_args"][0]["source_step"] == "filter"
    assert by_name["w"]["source_args"][0]["field"] == "std_innov"
    # kwargs are stored as the step holds them; no form-shaped renaming
    assert by_name["w"]["kwargs"]["target"] == [0.0]


def test_to_spec_is_a_fixed_point_under_rebuild() -> None:
    pipe = _simulation_pipeline()
    spec1 = pipe.to_spec()

    rebuilt = MCPipeline.from_spec(spec1)

    assert pipeline_meta(rebuilt.to_spec()) == pipeline_meta(spec1)


def test_to_spec_lifts_bulk_arrays_out_of_the_meta() -> None:
    states = np.zeros((4, 6, 3))
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

    dat = spec.steps[0]
    assert dat.meta["step_type"] == "raw_model_data"
    assert dat.meta["kwargs"]["observable_names"] == ["a", "b", "c"]
    # Bulk arrays ride their own slot, under the kwarg names they were passed
    # with.
    assert set(dat.arrays) == {"states", "observables"}
    assert dat.arrays["states"].shape == (4, 6, 3)
    assert dat.arrays["observables"].shape == (4, 5, 3)
    assert "states" not in dat.meta["kwargs"]
    assert "observables" not in dat.meta["kwargs"]
    # No bulk arrays leak into the JSON document.
    json.dumps(pipeline_meta(spec))

    # They go back under the names they came from.
    rebuilt = MCPipeline.from_spec(spec)
    np.testing.assert_array_equal(rebuilt.steps[0].kwargs["observables"], observables)


def test_small_array_kwargs_stay_inline() -> None:
    pipe = _simulation_pipeline()
    w = {step.meta["name"]: step for step in pipe.to_spec().steps}["w"]
    # `target` is a 1-element array: too small to be worth its own member.
    assert w.arrays == {}
    assert w.meta["kwargs"]["target"] == [0.0]


def test_to_spec_carries_a_custom_transform_callable_beside_its_meta() -> None:
    pipe = MCPipeline(
        [
            raw_model_data_step("dat", observables=np.zeros((4, 5, 3))),
            transform_step(
                "tf",
                _copy_transform,
                source="dat",
                field="observables",
                output_shape=(5, 3),
            ),
        ]
    )
    spec = pipe.to_spec()

    tf = {step.meta["name"]: step for step in spec.steps}["tf"]
    assert tf.meta["step_type"] == "transform:custom"
    # the callable rides its own slot; the meta stays JSON
    assert isinstance(tf.func, NumbaCustomFunc)
    assert tf.meta["kwargs"] == {"output_shape": [5, 3]}
    assert tf.meta["source_args"][0]["source_step"] == "dat"
    assert tf.meta["source_args"][0]["field"] == "observables"
    json.dumps(pipeline_meta(spec))


def test_to_spec_rejects_a_callable_no_step_kind_can_restore() -> None:
    from SymbolicDSGE.monte_carlo.mc_constructs import MCStep, OpType

    step = MCStep(
        name="jb",
        op_type=OpType.TEST,
        func=_copy_transform,
        step_type="jarque_bera",
    )
    with pytest.raises(ValueError, match="carries a callable"):
        step.to_spec()
