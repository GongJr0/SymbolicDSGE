"""Filter outputs across a bundle write and read.

A filter step is the only result kind whose arrays are neither one shape nor one
rank: the covariance blocks are 4-D, ``loglik`` is one value per replication,
and which fields exist at all depends on the mode and on whether shocks were
asked for. What is under test is that the meta carries enough to put all of
that back exactly as the run produced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.bundle import BundleBuilder, build_from
from SymbolicDSGE.core.solved_model import SolvedModel
from SymbolicDSGE.kalman.config import KalmanConfig
from SymbolicDSGE.monte_carlo import MCPipeline
from SymbolicDSGE.monte_carlo.mc_constructs import MCFilterResult
from SymbolicDSGE.monte_carlo.step_factories import (
    raw_model_data_step,
    reference_filter_step,
)

T = 6
N_REP = 4

_REQUIRED = (
    "x_pred",
    "x_filt",
    "P_pred",
    "P_filt",
    "y_pred",
    "y_filt",
    "S",
    "innov",
    "std_innov",
    "loglik",
)
_PRUNED = ("x1_pred", "x2_pred", "x1_filt", "x2_filt")
#: Fields a linear or extended run leaves unpopulated, read through the guards.
_OPTIONAL = ("eps_hat", "_x1_pred", "_x2_pred", "_x1_filt", "_x2_filt")


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


def _pipeline(solved: SolvedModel, *filters: object) -> MCPipeline:
    rng = np.random.default_rng(20260909)
    y = rng.normal(scale=0.01, size=(N_REP, T, len(solved.compiled.observable_names)))
    return MCPipeline(
        [
            raw_model_data_step(
                "data",
                observables=y,
                observable_names=tuple(solved.compiled.observable_names),
            ),
            *filters,  # type: ignore[list-item]
        ]
    )


def _roundtrip(
    solved: SolvedModel,
    pipeline: MCPipeline,
    tmp_path,
    *,
    as_parquet: bool = True,
) -> tuple[dict[str, MCFilterResult], dict[str, MCFilterResult]]:
    """Run, bundle, reload; return the run's filters beside the reloaded ones."""
    result = pipeline.run(solved, n_rep=N_REP, verbosity=0)
    target = (
        BundleBuilder(created_by="filter-roundtrip")
        .add_mc(pipeline, result=result, as_parquet=as_parquet)
        .write(tmp_path / "filters.sdsge")
    )
    loaded = build_from(target)
    assert loaded.mc is not None and loaded.mc.result is not None
    return dict(result.filter_outputs), dict(loaded.mc.result.filter_outputs)


def _assert_same(before: MCFilterResult, after: MCFilterResult) -> None:
    """Every field, by whatever equality its type has. Nothing is skipped."""
    for name in before.__dataclass_fields__:
        original, restored = getattr(before, name), getattr(after, name)
        if original is None or isinstance(original, (str, int)):
            assert restored == original, name
            continue
        assert restored is not None, name
        assert restored.shape == original.shape, name
        assert restored.dtype == original.dtype, name
        np.testing.assert_array_equal(restored, original, err_msg=name)


# --------------------------------------------------------------------------
# One filter step, per mode
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["linear", "extended"])
def test_a_filter_round_trips_every_field_it_produced(
    linear: SolvedModel, mode: str, tmp_path
) -> None:
    pipeline = _pipeline(linear, reference_filter_step("filt", filter_mode=mode))

    before, after = _roundtrip(linear, pipeline, tmp_path)

    _assert_same(before["filt"], after["filt"])
    assert after["filt"].filter_mode == mode


def test_the_unscented_pruned_state_survives_the_bundle(
    second_order: SolvedModel, tmp_path
) -> None:
    """Only this mode opens those four buffers, and the mode is what says so."""
    pipeline = _pipeline(
        second_order, reference_filter_step("filt", filter_mode="unscented")
    )

    before, after = _roundtrip(second_order, pipeline, tmp_path)

    _assert_same(before["filt"], after["filt"])
    for name in _PRUNED:
        np.testing.assert_array_equal(
            getattr(after["filt"], name), getattr(before["filt"], name)
        )


def test_the_mode_is_read_back_rather_than_inferred(
    linear: SolvedModel, tmp_path
) -> None:
    """Linear and extended write identical field sets, so only the meta separates them."""
    pipeline = _pipeline(linear, reference_filter_step("filt", filter_mode="extended"))

    _, after = _roundtrip(linear, pipeline, tmp_path)

    assert after["filt"].filter_mode == "extended"
    # Nothing mode-specific was produced, so the optional block stays absent and
    # only the shared fields come back populated.
    assert all(getattr(after["filt"], name, None) is None for name in _OPTIONAL)


# --------------------------------------------------------------------------
# Presence, which the meta spells by omission
# --------------------------------------------------------------------------


def test_requested_shocks_come_back_and_absent_ones_stay_absent(
    linear: SolvedModel, tmp_path
) -> None:
    pipeline = _pipeline(
        linear,
        reference_filter_step("with_shocks", return_shocks=True),
        reference_filter_step("without_shocks", return_shocks=False),
    )

    before, after = _roundtrip(linear, pipeline, tmp_path)

    assert after["with_shocks"].eps_hat is not None
    np.testing.assert_array_equal(
        after["with_shocks"].eps_hat, before["with_shocks"].eps_hat
    )
    assert after["without_shocks"].eps_hat is None


@pytest.mark.parametrize("name", _PRUNED)
def test_a_reloaded_linear_filter_still_refuses_the_pruned_state(
    linear: SolvedModel, name: str, tmp_path
) -> None:
    """A field with no member must come back absent, not as an empty array."""
    pipeline = _pipeline(linear, reference_filter_step("filt"))

    _, after = _roundtrip(linear, pipeline, tmp_path)

    with pytest.raises(AttributeError, match="only available for unscented"):
        getattr(after["filt"], name)


# --------------------------------------------------------------------------
# Shapes the flat storage has to restore
# --------------------------------------------------------------------------


def test_the_covariance_blocks_keep_their_rank(linear: SolvedModel, tmp_path) -> None:
    """Storage flattens to ``(-1, last)``, so 4-D has to come back off the meta."""
    n_var = linear.compiled.n_var
    n_obs = len(linear.compiled.observable_names)
    pipeline = _pipeline(linear, reference_filter_step("filt"))

    _, after = _roundtrip(linear, pipeline, tmp_path)

    assert after["filt"].P_pred.shape == (N_REP, T, n_var, n_var)
    assert after["filt"].P_filt.shape == (N_REP, T, n_var, n_var)
    assert after["filt"].S.shape == (N_REP, T, n_obs, n_obs)


def test_a_per_replication_scalar_keeps_its_single_axis(
    linear: SolvedModel, tmp_path
) -> None:
    """``loglik`` must not pick up a trailing axis on the way through storage."""
    pipeline = _pipeline(linear, reference_filter_step("filt"))

    _, after = _roundtrip(linear, pipeline, tmp_path)

    loglik = after["filt"].loglik
    assert loglik.shape == (N_REP,)
    # A trailing axis would survive comparison and fail only here.
    assert isinstance(float(loglik[0]), float)


def test_retaining_nothing_still_restores_the_shapes(
    linear: SolvedModel, tmp_path
) -> None:
    """Retaining nothing means no trace members, and the axes come from the meta."""
    n_var = linear.compiled.n_var
    pipeline = _pipeline(linear, reference_filter_step("filt", 0))

    _, after = _roundtrip(linear, pipeline, tmp_path)

    assert after["filt"].x_filt.shape == (0, T, n_var)
    assert after["filt"].P_pred.shape == (0, T, n_var, n_var)
    assert after["filt"].loglik.shape == (0,)


# --------------------------------------------------------------------------
# The mapping, and the non-parquet member format
# --------------------------------------------------------------------------


def test_filters_round_trip_keyed_by_step_name(linear: SolvedModel, tmp_path) -> None:
    pipeline = _pipeline(
        linear,
        reference_filter_step("kf", filter_mode="linear"),
        reference_filter_step("ekf", filter_mode="extended"),
    )

    before, after = _roundtrip(linear, pipeline, tmp_path)

    assert set(after) == {"kf", "ekf"}
    assert after["kf"].filter_mode == "linear"
    assert after["ekf"].filter_mode == "extended"
    for name in after:
        _assert_same(before[name], after[name])


def test_filters_round_trip_through_csv_members(linear: SolvedModel, tmp_path) -> None:
    pipeline = _pipeline(linear, reference_filter_step("filt", return_shocks=True))

    before, after = _roundtrip(linear, pipeline, tmp_path, as_parquet=False)

    for name in (*_REQUIRED, "eps_hat"):
        np.testing.assert_allclose(
            getattr(after["filt"], name), getattr(before["filt"], name)
        )


def test_a_step_that_retained_nothing_reloads_its_index_map_empty(
    linear: SolvedModel, tmp_path
) -> None:
    """Retaining nothing means no member for the index map, and that is not corruption."""
    pipeline = _pipeline(linear, reference_filter_step("filt", 0))

    before, after = _roundtrip(linear, pipeline, tmp_path)

    assert after["filt"].n_retained == 0
    assert after["filt"].retained_reps.shape == (0,)
    assert after["filt"].n_rep == before["filt"].n_rep


def test_the_index_map_says_which_replication_each_entry_holds(
    linear: SolvedModel, tmp_path
) -> None:
    """Rows are compact under partial retention, so the mapping is the only link."""
    pipeline = _pipeline(linear, reference_filter_step("filt", 2))

    before, after = _roundtrip(linear, pipeline, tmp_path)

    assert after["filt"].retained_reps.dtype == np.int64
    np.testing.assert_array_equal(
        after["filt"].retained_reps, before["filt"].retained_reps
    )
    assert after["filt"].retained_reps.shape == (2,)
    assert after["filt"].n_rep == N_REP


def test_a_pipeline_without_filters_reloads_with_none(
    linear: SolvedModel, tmp_path
) -> None:
    """A kind with no steps writes no member, and absence has to read as empty."""
    pipeline = _pipeline(linear)

    _, after = _roundtrip(linear, pipeline, tmp_path)

    assert after == {}
