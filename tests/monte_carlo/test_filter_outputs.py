"""What a run reports back from its filter steps.

The kernels themselves are checked against the reference filter elsewhere; what
is under test here is the compiler that reads a filter's retained arena into an
:class:`MCFilterResult`, which is where the mode, the field shapes, and the
mode-specific presence rules are decided.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE._ckernels.monte_carlo._arenas import resolve_retention
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.config import KalmanConfig
from SymbolicDSGE.kalman.filter import FilterResult, UnscentedFilterResult
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.mc_constructs import MCPipelineResult
from SymbolicDSGE.monte_carlo.step_factories import (
    raw_model_data_step,
    reference_filter_step,
)

T = 8
N_REP = 4

#: A run that keeps a strict subset, so the two counts cannot be confused.
PARTIAL_N_REP = 8
PARTIAL_N_RETAIN = 3

FilterMode = Literal["linear", "extended", "unscented"]

#: Every array field the linear and extended layouts open a buffer for, paired
#: with the axes each one carries past the replication axis.
_SHARED_FIELDS = (
    ("x_pred", ("T", "n_var")),
    ("x_filt", ("T", "n_var")),
    ("y_pred", ("T", "n_obs")),
    ("y_filt", ("T", "n_obs")),
    ("innov", ("T", "n_obs")),
    ("std_innov", ("T", "n_obs")),
    ("S", ("T", "n_obs", "n_obs")),
)

_UNSCENTED_ONLY = ("x1_pred", "x2_pred", "x1_filt", "x2_filt")


@pytest.fixture(scope="module")
def linear() -> SolvedModel:
    model, kalman = ModelParser("MODELS/POST82.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


@pytest.fixture(scope="module")
def second_order() -> SolvedModel:
    model, _ = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, KalmanConfig(R=np.array([[0.01]], dtype=np.float64)))
    return solver.solve(solver.compile(), order=2)


def _observations(solved: SolvedModel, n_rep: int = N_REP) -> np.ndarray:
    """Distinct data per replication, so a rep reading another's shows up."""
    rng = np.random.default_rng(20260908)
    n_obs = len(solved.compiled.observable_names)
    return rng.normal(scale=0.01, size=(n_rep, T, n_obs))


def _run(
    solved: SolvedModel,
    y: np.ndarray,
    *,
    n_rep: int = N_REP,
    name: str = "filt",
    **filter_kwargs: object,
) -> MCPipelineResult:
    pipeline = MCPipeline(
        [
            raw_model_data_step(
                "data",
                observables=y,
                observable_names=tuple(solved.compiled.observable_names),
            ),
            reference_filter_step(name, **filter_kwargs),  # type: ignore[arg-type]
        ]
    )
    return pipeline.run(solved, n_rep=n_rep, verbosity=0)


# --------------------------------------------------------------------------
# The mode, which nothing the run leaves behind records
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["linear", "extended"])
def test_the_result_carries_the_mode_that_selected_the_kernel(
    linear: SolvedModel, mode: str
) -> None:
    """Lowering consumes the mode to pick a kernel and keeps no note of it."""
    y = _observations(linear)
    result = _run(linear, y, filter_mode=mode)

    assert result.filter_outputs["filt"].filter_mode == mode


def test_the_mode_defaults_the_way_the_planner_defaults_it(
    linear: SolvedModel,
) -> None:
    """A step that never named a mode still reports the one it ran under."""
    y = _observations(linear)
    result = _run(linear, y)

    assert result.filter_outputs["filt"].filter_mode == "linear"


def test_unscented_reports_its_own_mode(second_order: SolvedModel) -> None:
    y = _observations(second_order)
    result = _run(second_order, y, filter_mode="unscented")

    assert result.filter_outputs["filt"].filter_mode == "unscented"


# --------------------------------------------------------------------------
# Field shapes: the arena lane is flat, so every axis is restored from the plan
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["linear", "extended"])
def test_every_field_is_restored_to_its_planned_shape(
    linear: SolvedModel, mode: str
) -> None:
    comp = linear.compiled
    dims = {"T": T, "n_var": comp.n_var, "n_obs": len(comp.observable_names)}
    y = _observations(linear)

    filt = _run(linear, y, filter_mode=mode).filter_outputs["filt"]

    for field, axes in _SHARED_FIELDS:
        expected = (N_REP, *(dims[axis] for axis in axes))
        assert getattr(filt, field).shape == expected, field
    for field in ("P_pred", "P_filt"):
        assert getattr(filt, field).shape == (N_REP, T, comp.n_var, comp.n_var), field


def test_a_scalar_field_lands_on_the_replication_axis_alone(
    linear: SolvedModel,
) -> None:
    """``loglik`` is planned with no axes, so it must not keep a trailing one."""
    y = _observations(linear)

    filt = _run(linear, y).filter_outputs["filt"]

    assert filt.loglik.shape == (N_REP,)


def test_unscented_covariances_are_in_the_augmented_frame(
    second_order: SolvedModel,
) -> None:
    """The pruned state doubles the covariance frame but not the reported state."""
    comp = second_order.compiled
    n_z = 2 * comp.n_state
    y = _observations(second_order)

    filt = _run(second_order, y, filter_mode="unscented").filter_outputs["filt"]

    assert filt.P_pred.shape == (N_REP, T, n_z, n_z)
    assert filt.P_filt.shape == (N_REP, T, n_z, n_z)
    assert filt.x_pred.shape == (N_REP, T, comp.n_var)
    for field in _UNSCENTED_ONLY:
        assert getattr(filt, field).shape == (N_REP, T, comp.n_state), field


# --------------------------------------------------------------------------
# Presence: the layout differs by mode, and by whether shocks were asked for
# --------------------------------------------------------------------------


def test_shocks_are_absent_unless_the_step_asked_for_them(
    linear: SolvedModel,
) -> None:
    """The layout sizes ``eps_hat`` to nothing, which reads as "not produced"."""
    y = _observations(linear)

    filt = _run(linear, y, return_shocks=False).filter_outputs["filt"]

    assert filt.eps_hat is None


def test_requested_shocks_come_back_shaped_by_the_exogenous_count(
    linear: SolvedModel,
) -> None:
    y = _observations(linear)

    filt = _run(linear, y, return_shocks=True).filter_outputs["filt"]

    assert filt.eps_hat is not None
    assert filt.eps_hat.shape == (N_REP, T, linear.compiled.n_exog)


@pytest.mark.parametrize("mode", ["linear", "extended"])
@pytest.mark.parametrize("field", _UNSCENTED_ONLY)
def test_the_pruned_state_is_unavailable_off_the_unscented_path(
    linear: SolvedModel, mode: str, field: str
) -> None:
    """Those layouts reserve no buffer at all, so the result must not invent one."""
    y = _observations(linear)

    filt = _run(linear, y, filter_mode=mode).filter_outputs["filt"]

    with pytest.raises(AttributeError, match="only available for unscented"):
        getattr(filt, field)


def test_the_unscented_layout_reserves_nothing_for_shocks(
    second_order: SolvedModel,
) -> None:
    """The unscented kernel cannot return shocks, so the field is simply absent."""
    y = _observations(second_order)

    filt = _run(second_order, y, filter_mode="unscented").filter_outputs["filt"]

    assert filt.eps_hat is None


# --------------------------------------------------------------------------
# Values: each retained replication carries its own filter, not a neighbour's
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["linear", "extended"])
def test_each_replication_matches_the_reference_filter_on_its_own_data(
    linear: SolvedModel, mode: str
) -> None:
    y = _observations(linear)

    filt = _run(linear, y, filter_mode=mode, return_shocks=True).filter_outputs["filt"]

    for rep in range(N_REP):
        expected = linear.kalman(
            y=y[rep], filter_mode=cast(FilterMode, mode), return_shocks=True
        )
        for field, _ in _SHARED_FIELDS:
            np.testing.assert_allclose(
                getattr(filt, field)[rep],
                getattr(expected, field),
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"{field} on replication {rep}",
            )
        assert filt.eps_hat is not None and expected.eps_hat is not None
        np.testing.assert_allclose(
            filt.eps_hat[rep], expected.eps_hat, rtol=1e-9, atol=1e-12
        )
        assert float(filt.loglik[rep]) == pytest.approx(float(expected.loglik))


def test_unscented_replications_match_the_reference_filter(
    second_order: SolvedModel,
) -> None:
    y = _observations(second_order)

    filt = _run(second_order, y, filter_mode="unscented").filter_outputs["filt"]

    for rep in range(N_REP):
        expected = second_order.kalman(y=y[rep], filter_mode="unscented")
        for field in ("x_pred", "x_filt", "P_pred", "P_filt", *_UNSCENTED_ONLY):
            np.testing.assert_allclose(
                getattr(filt, field)[rep],
                getattr(expected, field),
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"{field} on replication {rep}",
            )
        assert float(filt.loglik[rep]) == pytest.approx(float(expected.loglik))


# --------------------------------------------------------------------------
# replication(): one rep's slice, as the filter classes themselves
# --------------------------------------------------------------------------


def test_a_replication_reads_back_as_a_plain_filter_result(
    linear: SolvedModel,
) -> None:
    y = _observations(linear)

    filt = _run(linear, y).filter_outputs["filt"]
    one = filt.replication(2)

    assert isinstance(one, FilterResult)
    np.testing.assert_array_equal(one.x_filt, filt.x_filt[2])
    np.testing.assert_array_equal(one.S, filt.S[2])
    assert one.eps_hat is None


def test_a_replication_of_an_unscented_run_carries_the_pruned_state(
    second_order: SolvedModel,
) -> None:
    y = _observations(second_order)

    filt = _run(second_order, y, filter_mode="unscented").filter_outputs["filt"]
    one = filt.replication(1)

    assert isinstance(one, UnscentedFilterResult)
    np.testing.assert_array_equal(one.x1_filt, filt.x1_filt[1])
    np.testing.assert_array_equal(one.x2_pred, filt.x2_pred[1])


@pytest.mark.parametrize("idx", [-1, PARTIAL_N_RETAIN, PARTIAL_N_REP])
def test_oob_index_retained_is_refused(linear: SolvedModel, idx: int) -> None:
    """The bound is what was retained, not the run's replication count.

    Retention is partial so the counts differ. ``PARTIAL_N_RETAIN`` is the
    discriminating case: a replication the run produced and did not keep, which
    a bound on ``n_rep`` would let through.
    """
    n_rep, n_retain = PARTIAL_N_REP, PARTIAL_N_RETAIN
    y = _observations(linear, n_rep=n_rep)

    filt = _run(linear, y, n_rep=n_rep, n_retain=n_retain).filter_outputs["filt"]

    with pytest.raises(IndexError, match="retained replications"):
        filt.replication(idx)


def test_retention_indices_match_the_run_when_everything_is_kept(
    linear: SolvedModel,
) -> None:
    """Retaining everything makes ``retained_reps`` the identity"""
    y = _observations(linear)

    filt = _run(linear, y).filter_outputs["filt"]

    for rep in range(N_REP):
        np.testing.assert_array_equal(filt.replication(rep).x_filt, filt.x_filt[rep])


# --------------------------------------------------------------------------
# Retention, which decides how many replications there are to read at all
# --------------------------------------------------------------------------


def test_partial_retention_reports_only_the_rows_it_kept(
    linear: SolvedModel,
) -> None:
    """The leading axis counts retained replications, not run replications."""
    n_rep, n_retain = PARTIAL_N_REP, PARTIAL_N_RETAIN
    y = _observations(linear, n_rep=n_rep)

    result = _run(linear, y, n_rep=n_rep, n_retain=n_retain)
    filt = result.filter_outputs["filt"]

    assert filt.x_filt.shape[0] == n_retain
    assert filt.loglik.shape == (n_retain,)
    assert result.meta.n_retained_by_step["filt"] == n_retain


def test_each_retained_entry_holds_the_replication_it_was_sampled_from(
    linear: SolvedModel,
) -> None:
    """Storage is compact, so entry ``i`` is the filter for ``retained_reps[i]``."""
    n_rep, n_retain = PARTIAL_N_REP, PARTIAL_N_RETAIN
    y = _observations(linear, n_rep=n_rep)

    filt = _run(linear, y, n_rep=n_rep, n_retain=n_retain).filter_outputs["filt"]

    # Which replications get sampled is the allocator's rule, so it is asked
    # rather than restated here, and what the result reports has to agree.
    retained_reps, _ = resolve_retention(n_retain, n_rep)
    np.testing.assert_array_equal(filt.retained_reps, retained_reps)
    for idx, rep in enumerate(retained_reps):
        np.testing.assert_allclose(
            filt.x_filt[idx], linear.kalman(y=y[rep]).x_filt, rtol=1e-9, atol=1e-12
        )


def test_retaining_nothing_still_reports_the_planned_shapes(
    linear: SolvedModel,
) -> None:
    """An empty run retained nothing, and the axes still have to be right."""
    comp = linear.compiled
    y = _observations(linear)

    filt = _run(linear, y, n_retain=0).filter_outputs["filt"]

    assert filt.x_filt.shape == (0, T, comp.n_var)
    assert filt.P_pred.shape == (0, T, comp.n_var, comp.n_var)
    assert filt.loglik.shape == (0,)


# --------------------------------------------------------------------------
# The mapping itself
# --------------------------------------------------------------------------


def test_filters_are_keyed_by_step_name(linear: SolvedModel) -> None:
    y = _observations(linear)
    pipeline = MCPipeline(
        [
            raw_model_data_step(
                "data",
                observables=y,
                observable_names=tuple(linear.compiled.observable_names),
            ),
            reference_filter_step("kf", filter_mode="linear"),
            reference_filter_step("ekf", filter_mode="extended"),
        ]
    )

    result = pipeline.run(linear, n_rep=N_REP, verbosity=0)

    assert set(result.filter_outputs) == {"kf", "ekf"}
    assert result.filter_outputs["kf"].filter_mode == "linear"
    assert result.filter_outputs["ekf"].filter_mode == "extended"


def test_a_pipeline_without_filters_reports_no_filter_outputs(
    linear: SolvedModel,
) -> None:
    """Emptiness is how every step kind reports "none of these"."""
    y = _observations(linear)
    pipeline = MCPipeline(
        [
            raw_model_data_step(
                "data",
                observables=y,
                observable_names=tuple(linear.compiled.observable_names),
            )
        ]
    )

    result = pipeline.run(linear, n_rep=N_REP, verbosity=0)

    assert result.filter_outputs == {}
