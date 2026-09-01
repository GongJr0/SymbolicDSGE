"""Rejection paths a pipeline the native runner cannot execute takes."""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.allocation import FieldLayout
from SymbolicDSGE.monte_carlo.mc_constructs import MCStep
from SymbolicDSGE.monte_carlo.native_lowering.filters import _filter_y_binding
from SymbolicDSGE.monte_carlo.step_factories import (
    breusch_pagan_test_step,
    jarque_bera_test_step,
    ljung_box_test_step,
    log_diff_step,
    raw_model_data_step,
    reference_filter_step,
    simulation_step,
    wald_test_step,
)

T = 12
N_REP = 2


@pytest.fixture(scope="module")
def solved() -> SolvedModel:
    model, kalman = ModelParser("MODELS/POST82.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


def _two_column_data() -> np.ndarray:
    rng = np.random.default_rng(20260801)
    return rng.normal(size=(N_REP, T, 2))


def _lower(steps: list[MCStep], reference: object = None) -> None:
    """Lower a pipeline far enough to reach the step compilers."""
    MCPipeline(steps).lower_native(
        reference=cast(SolvedModel, reference if reference is not None else object()),
        n_rep=N_REP,
        n_jobs=1,
    )


# --------------------------------------------------------------------------
# diagnostics.py
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [ljung_box_test_step, jarque_bera_test_step])
def test_single_column_diagnostics_reject_a_wide_source(factory: object) -> None:
    """Arena planning sizes the source, so it is what refuses a second column."""
    step = cast(MCStep, factory("diag", source="data", field="observables"))  # type: ignore[operator]
    steps = [
        raw_model_data_step("data", observables=_two_column_data()),
        step,
    ]

    with pytest.raises(ValueError, match="requires a single-column source"):
        _lower(steps)


def test_an_unknown_diagnostic_kind_has_no_arena() -> None:
    """Only the kinds with a native kernel can be planned, let alone lowered."""
    step = ljung_box_test_step("diag", source="data", field="observables", column=0)
    steps = [
        raw_model_data_step("data", observables=_two_column_data()),
        dataclasses.replace(step, step_type="not_a_diagnostic"),
    ]

    with pytest.raises(NotImplementedError, match="not implemented for test step type"):
        _lower(steps)


def test_an_unknown_wald_kind_has_no_arena() -> None:
    """The kind picks the native arena, so an unknown one fails before lowering."""
    step = wald_test_step(
        "wald", source="data", field="observables", target=np.zeros(2)
    )
    steps = [
        raw_model_data_step("data", observables=_two_column_data()),
        dataclasses.replace(step, kwargs={**step.kwargs, "kind": "median"}),
    ]

    with pytest.raises(ValueError, match="Unsupported native diagnostic kind"):
        _lower(steps)


def test_two_source_diagnostics_reject_a_wide_response() -> None:
    steps = [
        raw_model_data_step("data", observables=_two_column_data()),
        breusch_pagan_test_step(
            "bp",
            residuals_source="data",
            residuals_field="observables",
            X_source="data",
            X_field="observables",
        ),
    ]

    with pytest.raises(ValueError, match="requires a single-column source"):
        _lower(steps)


def test_two_source_diagnostics_reject_mismatched_row_counts() -> None:
    """The response and the regressors are staged into one arena, row by row."""
    steps = [
        raw_model_data_step("data", observables=np.exp(_two_column_data())),
        # A log difference drops a row, so this source is one shorter than the raw.
        log_diff_step("growth", source="data", field="observables", columns=1),
        breusch_pagan_test_step(
            "bp",
            residuals_source="data",
            residuals_field="observables",
            residual_col=0,
            X_source="growth",
            X_field="payload",
        ),
    ]

    with pytest.raises(ValueError, match="must have matching row counts"):
        _lower(steps)


def test_a_two_source_diagnostic_needs_both_sources() -> None:
    step = breusch_pagan_test_step(
        "bp",
        residuals_source="data",
        residuals_field="observables",
        residual_col=0,
        X_source="data",
        X_field="observables",
        X_columns=1,
    )
    steps = [
        raw_model_data_step("data", observables=_two_column_data()),
        dataclasses.replace(step, source_args=step.source_args[:1]),
    ]

    with pytest.raises(ValueError, match="requires two source arguments"):
        _lower(steps)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kernel": "epanechnikov"}, "Unsupported native Wald configuration"),
        ({"bandwidth": True}, "bandwidth must be an integer, mode, or None"),
        ({"bandwidth": "newey_west"}, "Unsupported native Wald bandwidth mode"),
    ],
)
def test_wald_rejects_configurations_without_a_kernel(
    overrides: dict[str, object], message: str
) -> None:
    step = wald_test_step(
        "wald",
        source="data",
        field="observables",
        target=np.zeros(2),
    )
    steps = [
        raw_model_data_step("data", observables=_two_column_data()),
        dataclasses.replace(step, kwargs={**step.kwargs, **overrides}),
    ]

    with pytest.raises(ValueError, match=message):
        _lower(steps)


@pytest.mark.parametrize(
    ("kind", "target", "message"),
    [
        ("mean", np.zeros(3), "one value per source column"),
        ("covariance", np.zeros((3, 3)), "match the source column count"),
        ("covariance", np.zeros(2), "match the source column count"),
    ],
)
def test_wald_targets_must_match_the_source_width(
    kind: str, target: np.ndarray, message: str
) -> None:
    steps = [
        raw_model_data_step("data", observables=_two_column_data()),
        wald_test_step(
            "wald",
            source="data",
            field="observables",
            kind=cast(object, kind),  # type: ignore[arg-type]
            target=target,
        ),
    ]

    with pytest.raises(ValueError, match=message):
        _lower(steps)


# --------------------------------------------------------------------------
# filters.py
# --------------------------------------------------------------------------


def test_filter_rejects_a_datagen_whose_width_it_cannot_match(
    solved: SolvedModel,
) -> None:
    """Unnamed raw observables must line up positionally, so the count must match."""
    n_obs = len(solved.compiled.observable_names)
    steps = [
        raw_model_data_step(
            "data", observables=np.zeros((N_REP, T, n_obs + 1), dtype=np.float64)
        ),
        reference_filter_step("filter"),
    ]

    with pytest.raises(ValueError, match="do not match the DATAGEN output"):
        _lower(steps, reference=solved)


def test_unscented_filtering_cannot_return_shocks(solved: SolvedModel) -> None:
    steps = [
        simulation_step("sim", target="reference", T=T, observables=True),
        reference_filter_step("filter", filter_mode="unscented", return_shocks=True),
    ]

    with pytest.raises(ValueError, match="does not support return_shocks"):
        _lower(steps, reference=solved)


def test_an_unknown_filter_mode_is_rejected(solved: SolvedModel) -> None:
    """The interface resolves the mode before lowering picks a kernel for it."""
    step = reference_filter_step("filter")
    steps = [
        simulation_step("sim", target="reference", T=T, observables=True),
        dataclasses.replace(step, kwargs={**step.kwargs, "filter_mode": "particle"}),
    ]

    with pytest.raises(ValueError, match="Unrecognized filter mode"):
        _lower(steps, reference=solved)


def test_a_filter_on_dgp_simulated_data_needs_the_dgp(solved: SolvedModel) -> None:
    """Planning sizes the simulation first, so it is what misses the model."""
    pipeline = MCPipeline(
        [
            simulation_step("sim", target="dgp", T=T, observables=True),
            reference_filter_step("filter"),
        ]
    )

    with pytest.raises(ValueError, match="Simulation output planning requires"):
        pipeline.lower_native(reference=solved, dgp=None, n_rep=N_REP, n_jobs=1)


def test_filter_observables_must_be_unique(solved: SolvedModel) -> None:
    name = solved.compiled.observable_names[0]
    steps = [
        simulation_step("sim", target="reference", T=T, observables=True),
        reference_filter_step("filter", observables=[name, name]),
    ]

    with pytest.raises(ValueError, match="must be unique"):
        _lower(steps, reference=solved)


def test_filter_observables_must_exist_on_the_reference(solved: SolvedModel) -> None:
    steps = [
        simulation_step("sim", target="reference", T=T, observables=True),
        reference_filter_step("filter", observables=["not_an_observable"]),
    ]

    with pytest.raises(ValueError, match="Unknown reference observables"):
        _lower(steps, reference=solved)


def test_filter_observables_must_be_present_in_the_datagen_output(
    solved: SolvedModel,
) -> None:
    """Named raw data can carry a subset the filter does not fully cover."""
    all_names = tuple(solved.compiled.observable_names)
    steps = [
        raw_model_data_step(
            "data",
            observables=np.zeros((N_REP, T, 1), dtype=np.float64),
            observable_names=all_names[:1],
        ),
        reference_filter_step("filter", observables=list(all_names[:2])),
    ]

    with pytest.raises(ValueError, match="missing filter observables"):
        _lower(steps, reference=solved)


def test_filter_x0_must_cover_every_state(solved: SolvedModel) -> None:
    n_var = len(solved.compiled.var_names)
    steps = [
        simulation_step("sim", target="reference", T=T, observables=True),
        reference_filter_step("filter", x0=np.zeros(n_var - 1, dtype=np.float64)),
    ]

    with pytest.raises(ValueError, match=f"x0 must have length {n_var}"):
        _lower(steps, reference=solved)


def test_the_observation_binding_must_match_the_source_layout() -> None:
    """An internal consistency check between the staged rows and the plan."""
    layout = FieldLayout(
        shape=(T, 2), flat_count=T * 2, dtype=np.dtype(np.float64), offset=0
    )

    with pytest.raises(ValueError, match="do not match their input layout"):
        _filter_y_binding(layout, T + 1, np.asarray([0, 1], dtype=np.int64), 0, 2)

    with pytest.raises(ValueError, match="do not match their input layout"):
        _filter_y_binding(layout, T, np.asarray([0], dtype=np.int64), 0, 2)
