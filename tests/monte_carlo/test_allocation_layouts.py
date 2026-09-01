"""Output-layout planning: field shapes, arena sizes, and layout rejections."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE._ckernels.monte_carlo._offsets import ArenaOffset
from SymbolicDSGE.monte_carlo.allocation import (
    ArenaSize,
    _compile_field_layout,
    _FieldSpec,
    _resolve_input_asize,
    is_absent,
    resolve_output_specs,
)
from SymbolicDSGE.monte_carlo.mc_constructs import MCStep
from SymbolicDSGE.monte_carlo.step_factories import (
    add_payload_step,
    diff_step,
    log_diff_step,
    postproc_step,
    raw_model_data_step,
    reference_filter_step,
    regression_step,
    rolling_mean_step,
    rolling_std_step,
    rolling_var_step,
    simulation_step,
    standardize_step,
)

T = 12
N_REP = 2


@pytest.fixture(scope="module")
def solved() -> SolvedModel:
    model, kalman = ModelParser("MODELS/POST82.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


def _plan(steps: list[MCStep], reference: object = None) -> Any:
    """Resolve output layouts without lowering or allocating."""
    pipeline = MCPipeline(steps)
    return pipeline._resolve_output_specs(
        cast(SolvedModel, reference if reference is not None else object()), None
    )


def _data(*, rows: int = T, columns: int = 2) -> np.ndarray:
    rng = np.random.default_rng(20260801)
    return rng.normal(size=(N_REP, rows, columns))


def _with_datagen(*steps: MCStep) -> list[MCStep]:
    """Prepend the DATAGEN step MCPipeline requires at the head of a pipeline."""
    return [raw_model_data_step("data", observables=_data()), *steps]


# --------------------------------------------------------------------------
# add_payload_step
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ([1.0, 2.0, 3.0], (3, 1)),
        ([[1.0, 2.0], [3.0, 4.0]], (2, 2)),
        (np.zeros((N_REP, 5, 3)), (5, 3)),
    ],
    ids=["1d", "2d", "3d-batched"],
)
def test_a_payload_carries_its_own_shape(
    payload: object, expected: tuple[int, int]
) -> None:
    """A payload has no source, so its value alone fixes the layout."""
    plans = _plan(_with_datagen(add_payload_step("const", payload=cast(Any, payload))))

    assert plans["const"].out_fields["payload"].shape == expected


def test_a_payload_needs_no_input_arena() -> None:
    """Nothing is staged for a payload: the value is static backing."""
    plans = _plan(_with_datagen(add_payload_step("const", payload=[1.0, 2.0])))

    assert plans["const"].input_size == ArenaSize(0, 0)
    assert plans["const"].output_size == ArenaSize(n_float=2, n_int=0)


def test_a_payload_offsets_land_in_the_float_lane() -> None:
    plans = _plan(_with_datagen(add_payload_step("const", payload=[[1.0, 2.0]])))
    payload = plans["const"].out_fields["payload"]

    assert payload.dtype == np.dtype(np.float64)
    assert payload.offset == 0
    assert payload.flat_count == 2


def test_a_payload_feeds_a_downstream_transform() -> None:
    """The payload's resolved shape is what the consumer plans against."""
    plans = _plan(
        _with_datagen(
            add_payload_step("const", payload=np.zeros((6, 2))),
            log_diff_step("growth", source="const", field="payload"),
        )
    )

    assert plans["growth"].out_fields["payload"].shape == (5, 2)


@pytest.mark.parametrize("ndim", [0, 4])
def test_a_payload_must_be_1d_2d_or_3d(ndim: int) -> None:
    payload = np.zeros((2,) * ndim)

    with pytest.raises(ValueError, match="Payload must be 1D, 2D, or 3D"):
        _plan(_with_datagen(add_payload_step("const", payload=cast(Any, payload))))


# --------------------------------------------------------------------------
# transform output shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "kwargs", "expected_rows"),
    [
        (standardize_step, {}, T),
        (log_diff_step, {}, T - 1),
        (diff_step, {"order": 1}, T - 1),
        (diff_step, {"order": 3}, T - 3),
        (rolling_mean_step, {"window": 4}, T - 3),
        (rolling_std_step, {"window": 4}, T - 3),
        (rolling_var_step, {"window": 1}, T),
    ],
)
def test_row_shrinking_transforms_resolve_their_row_count(
    factory: Any, kwargs: dict[str, Any], expected_rows: int
) -> None:
    plans = _plan(
        [
            raw_model_data_step("data", observables=np.exp(_data())),
            factory("out", source="data", field="observables", **kwargs),
        ]
    )

    assert plans["out"].out_fields["payload"].shape == (expected_rows, 2)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [(diff_step, {"order": T + 5}), (rolling_mean_step, {"window": T + 5})],
)
def test_a_transform_wider_than_its_source_clamps_to_zero_rows(
    factory: Any, kwargs: dict[str, Any]
) -> None:
    """The layout stays non-negative; the kernel rejects the run later."""
    plans = _plan(
        [
            raw_model_data_step("data", observables=np.exp(_data())),
            factory("out", source="data", field="observables", **kwargs),
        ]
    )

    assert plans["out"].out_fields["payload"].shape == (0, 2)


def test_an_unknown_transform_type_has_no_output_shape() -> None:
    step = standardize_step("out", source="data", field="observables")
    steps = [
        raw_model_data_step("data", observables=_data()),
        dataclasses.replace(step, step_type="fourier"),
    ]

    with pytest.raises(NotImplementedError, match="transform step type"):
        _plan(steps)


def test_a_transform_takes_exactly_one_source() -> None:
    step = standardize_step("out", source="data", field="observables")
    steps = [
        raw_model_data_step("data", observables=_data()),
        dataclasses.replace(step, source_args=step.source_args * 2),
    ]

    with pytest.raises(ValueError, match="must have one source argument"):
        _plan(steps)


# --------------------------------------------------------------------------
# datagen shapes
# --------------------------------------------------------------------------


def test_one_dimensional_raw_data_gains_a_column_axis() -> None:
    plans = _plan([raw_model_data_step("data", observables=np.zeros(T))])

    assert plans["data"].out_fields["observables"].shape == (T, 1)


def test_raw_data_beyond_three_dimensions_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be 1D, 2D, or 3D"):
        _plan([raw_model_data_step("data", observables=np.zeros((2, 2, 2, 2)))])


def test_an_unknown_datagen_type_has_no_output_layout() -> None:
    step = raw_model_data_step("data", observables=_data())

    with pytest.raises(NotImplementedError, match="datagen step type"):
        _plan([dataclasses.replace(step, step_type="bootstrap")])


# --------------------------------------------------------------------------
# source selection
# --------------------------------------------------------------------------


def test_selecting_a_field_the_producer_does_not_emit_is_rejected() -> None:
    steps = [
        raw_model_data_step("data", observables=_data()),
        standardize_step("out", source="data", field="states"),
    ]

    with pytest.raises(ValueError, match="does not produce source field"):
        _plan(steps)


def test_a_non_two_dimensional_source_field_cannot_be_selected(
    solved: SolvedModel,
) -> None:
    """Filters emit rank-3 covariance fields that no transform can consume."""
    steps = [
        simulation_step("sim", target="reference", T=T, observables=True),
        reference_filter_step("filter"),
        standardize_step("out", source="filter", field="P_pred"),
    ]

    with pytest.raises(ValueError, match="must be 2D"):
        _plan(steps, reference=solved)


def test_a_filter_needs_datagen_observables(solved: SolvedModel) -> None:
    steps = [
        simulation_step("sim", target="reference", T=T, observables=False),
        reference_filter_step("filter"),
    ]

    with pytest.raises(ValueError, match="Filter output planning requires"):
        _plan(steps, reference=solved)


# --------------------------------------------------------------------------
# defensive paths, reached by calling the planner directly
# --------------------------------------------------------------------------


def test_a_postproc_step_has_no_per_replication_layout() -> None:
    """MCPipeline rejects these upstream, so the planner is called directly."""
    step = postproc_step("summary", func=lambda **_: None)

    with pytest.raises(NotImplementedError, match="Output-layout resolution"):
        resolve_output_specs([step], [[]], cast(SolvedModel, object()), None)


def test_a_negative_field_dimension_is_rejected() -> None:
    """No producer emits one; the check guards the shared layout compiler."""
    fields = {"payload": _FieldSpec(shape=(4, -1), dtype=np.float64)}
    offsets = ArenaOffset(foffset=(0,), fwidth=(0,), ioffset=(), iwidth=())

    with pytest.raises(ValueError, match="has a negative dimension"):
        _compile_field_layout(fields, offsets)


def test_a_postproc_step_has_no_input_arena() -> None:
    """Output resolution rejects it first, so the arena sizer is called direct."""
    step = postproc_step("summary", func=lambda **_: None)

    with pytest.raises(NotImplementedError, match="Input arena resolution"):
        _resolve_input_asize(step, [], {}, [step], cast(SolvedModel, object()), None)


# --------------------------------------------------------------------------
# regression layouts
# --------------------------------------------------------------------------


def _regression(**overrides: Any) -> MCStep:
    kwargs: dict[str, Any] = {
        "y_source": "data",
        "y_field": "observables",
        "y_column": [0],
        "X_source": "data",
        "X_field": "observables",
        "X_columns": [1],
    }
    kwargs.update(overrides)
    return regression_step("fit", **kwargs)


def test_a_regression_lays_out_one_coefficient_per_regressor() -> None:
    plans = _plan(_with_datagen(_regression(intercept=True)))
    fields = plans["fit"].out_fields

    assert fields["coef"].shape == (2,)
    assert fields["se"].shape == (2,)
    assert fields["status"].dtype == np.dtype(np.int64)


def test_a_non_ols_regression_reports_no_standard_errors() -> None:
    """The field keeps its name and its place, at the width the layout gave it."""
    plans = _plan(_with_datagen(_regression(kind="ridge", intercept=True)))

    assert is_absent(plans["fit"].out_fields["se"])


def test_a_regression_needs_both_a_response_and_a_design() -> None:
    step = _regression()
    steps = _with_datagen(dataclasses.replace(step, source_args=step.source_args[:1]))

    with pytest.raises(ValueError, match="must have response and design sources"):
        _plan(steps)


def test_a_regression_response_must_be_one_column() -> None:
    with pytest.raises(ValueError, match="response must resolve to one column"):
        _plan(_with_datagen(_regression(y_column=[0, 1])))


def test_a_regression_response_and_design_must_share_row_counts() -> None:
    steps = [
        raw_model_data_step("data", observables=np.exp(_data())),
        log_diff_step("growth", source="data", field="observables", columns=[1]),
        _regression(X_source="growth", X_field="payload", X_columns=None),
    ]

    with pytest.raises(ValueError, match="must have the same number of rows"):
        _plan(steps)


def test_a_regression_with_no_regressor_and_no_intercept_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires a regressor or an intercept"):
        _plan(_with_datagen(_regression(X_columns=[], intercept=False)))


def test_a_custom_transform_output_shape_must_be_two_non_negative_dimensions() -> None:
    step = standardize_step("out", source="data", field="observables")
    steps = _with_datagen(
        dataclasses.replace(
            step,
            step_type="transform:custom",
            kwargs={**step.kwargs, "output_shape": (T, -1)},
        )
    )

    with pytest.raises(ValueError, match="two non-negative dimensions"):
        _plan(steps)
