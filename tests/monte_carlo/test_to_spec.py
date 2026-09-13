from __future__ import annotations

import json

import numpy as np
import pytest

from SymbolicDSGE.core.shock_generators import Shock
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.custom_op import NumbaCustomFunc, PandasCustomFunc
from SymbolicDSGE.monte_carlo.postproc import run_kde
from SymbolicDSGE.monte_carlo.spec import pipeline_meta
from SymbolicDSGE.monte_carlo.step_factories import (
    jarque_bera_test_step,
    raw_model_data_step,
    reference_filter_step,
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
                T=8,
                observables=True,
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


def test_to_spec_structure_and_sources() -> None:
    spec = _simulation_pipeline().to_spec()

    assert [step.meta["step_type"] for step in spec.replication_steps] == [
        "simulation",
        "filter",
        "standardize",
        "jarque_bera",
        "wald",
    ]
    assert spec.postproc_steps == []

    by_name = {step.meta["name"]: step.meta for step in spec.replication_steps}
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
    # shocks are serialized to JSON-safe dicts
    assert by_name["dgp"]["kwargs"]["shocks"]["u"]["dist"] == "norm"
    # the meta half is the document a bundle writes, and it is JSON on its own
    json.dumps(pipeline_meta(spec))


def test_to_spec_is_a_fixed_point_under_rebuild() -> None:
    pipe = _simulation_pipeline()
    spec1 = pipe.to_spec()

    rebuilt = MCPipeline.from_spec(spec1)

    assert pipeline_meta(rebuilt.to_spec()) == pipeline_meta(spec1)


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
    with pytest.raises(TypeError, match="callable"):
        pipe.to_spec()


def test_rebuilt_simulation_recovers_live_shocks() -> None:
    pipe = _simulation_pipeline()
    rebuilt = MCPipeline.from_spec(pipe.to_spec())

    shock = rebuilt.replication_steps[0].kwargs["shocks"]["u"]
    assert isinstance(shock, Shock)
    assert shock.to_dict() == pipe.replication_steps[0].kwargs["shocks"]["u"].to_dict()


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

    dat = spec.replication_steps[0]
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
    np.testing.assert_array_equal(
        rebuilt.replication_steps[0].kwargs["observables"], observables
    )


def test_small_array_kwargs_stay_inline() -> None:
    pipe = _simulation_pipeline()
    w = {step.meta["name"]: step for step in pipe.to_spec().replication_steps}["w"]
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

    tf = {step.meta["name"]: step for step in spec.replication_steps}["tf"]
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


def test_to_spec_emits_a_postproc_custom_op_with_its_kwargs() -> None:
    from SymbolicDSGE.monte_carlo.step_factories import postproc_step

    pipe = MCPipeline(
        [
            raw_model_data_step("dat", observables=np.zeros((4, 5, 3))),
            jarque_bera_test_step("jb", source="dat", field="observables"),
        ],
        [postproc_step("sum", _my_summary, threshold=0.5)],
    )
    spec = pipe.to_spec()

    pp = {step.meta["name"]: step for step in spec.postproc_steps}["sum"]
    assert pp.meta["step_type"] == "postproc:custom"
    # the callable rides its own slot; op kwargs survive as plain meta kwargs
    assert pp.func is not None
    assert pp.meta["kwargs"]["threshold"] == 0.5
    # post-loop ops are a separate list, never replication steps
    assert "sum" not in {step.meta["name"] for step in spec.replication_steps}


def _my_summary(*, traces, threshold):
    return float(threshold)


def test_to_spec_round_trips_a_postproc_pipeline() -> None:
    from SymbolicDSGE.monte_carlo.step_factories import kde_step

    pipe = MCPipeline(
        [
            simulation_step(
                "dgp",
                T=8,
                observables=True,
                shocks={"u": Shock(dist="norm", seed=0)},
            ),
            jarque_bera_test_step("jb", source="dgp", field="observables", column=0),
        ],
        [kde_step("kde", trace="test.jb.statistic", grid_points=50)],
    )
    spec1 = pipe.to_spec()
    kde_pp = {step.meta["name"]: step for step in spec1.postproc_steps}["kde"]
    assert kde_pp.meta["step_type"] == "kde"
    assert kde_pp.meta["kwargs"]["trace"] == "test.jb.statistic"
    # a built-in post-loop kind carries the library callable it runs
    assert kde_pp.func is run_kde
    # post-loop ops are a separate list, never replication steps
    assert "kde" not in {step.meta["name"] for step in spec1.replication_steps}

    rebuilt = MCPipeline.from_spec(spec1)
    assert [step.name for step in rebuilt.replication_steps] == ["dgp", "jb"]
    assert [step.name for step in rebuilt.postproc_steps] == ["kde"]
    assert pipeline_meta(rebuilt.to_spec()) == pipeline_meta(spec1)  # fixed point


def test_postproc_custom_func_survives_a_rebuild() -> None:
    from SymbolicDSGE.monte_carlo.step_factories import postproc_step

    pipe = MCPipeline(
        [
            raw_model_data_step("dat", observables=np.zeros((4, 5, 3))),
            jarque_bera_test_step("jb", source="dat", field="observables"),
        ],
        [postproc_step("sum", PandasCustomFunc(_my_summary), threshold=0.5)],
    )
    rebuilt = MCPipeline.from_spec(pipe.to_spec())
    assert isinstance(rebuilt.postproc_steps[0].func, PandasCustomFunc)
