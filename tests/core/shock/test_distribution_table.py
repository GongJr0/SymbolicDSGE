"""The quantile table: what it refuses, what it standardizes to, and how it reads.

``DistributionTable`` is reached two ways, and they answer to different things. The
constructor takes knots as given and only checks they could have come from a quantile
function, which is the one route to most of its refusals. The tabulating routes build
the knots themselves and standardize them, so what they own is the measure the kernel
samples with: mean zero and unit variance under a clamped standard normal, not under
the distribution the knots came from.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import cauchy, laplace, norm, poisson, t as scipy_t

from SymbolicDSGE.core.shock.generators import Shock
from SymbolicDSGE.core.shock.spec import resolve_shock_plan
from SymbolicDSGE.core.shock.table import (
    ZMAX,
    DistributionTable,
    _cell_weights,
    _knot_probabilities,
)

#: Enough draws to see the standardization, few enough to stay instant. The seed is
#: fixed, so the tolerances below bound one known sample rather than a distribution.
_DRAWS = 200_000


def _mc(table: DistributionTable) -> tuple[float, float]:
    """Mean and variance of the table as the kernel reads it, over normal draws."""
    v = table.evaluate(np.random.default_rng(0).standard_normal(_DRAWS))
    return float(v.mean()), float(v.var())


# --- what the constructor refuses -------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        np.zeros((4, 2)),
        np.linspace(-1.0, 1.0, 8).reshape(2, 4),
        np.array([0.0]),
        np.array([]),
    ],
)
def test_knots_must_be_a_run_of_at_least_two(q) -> None:
    # Two knots is the minimum an interpolation can read between.
    with pytest.raises(ValueError, match="one-dimensional run of at least two"):
        DistributionTable(q=q)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_knot_names_its_index(bad) -> None:
    with pytest.raises(ValueError, match="Knot 1 of 3 is not finite"):
        DistributionTable(q=np.array([-1.0, bad, 1.0]))


def test_knots_must_not_decrease() -> None:
    # A quantile function is non-decreasing by definition, so a dip means the knots
    # did not come from one.
    with pytest.raises(ValueError, match="Knots decrease at index 1"):
        DistributionTable(q=np.array([-1.0, 1.0, 0.0]))


@pytest.mark.parametrize("zmax", [0.0, -6.0, np.nan, np.inf])
def test_zmax_must_be_finite_and_positive(zmax) -> None:
    with pytest.raises(ValueError, match="zmax must be finite and positive"):
        DistributionTable(q=np.array([-1.0, 1.0]), zmax=zmax)


def test_knots_that_repeat_refuse_to_pass_as_continuous() -> None:
    # Ties at this rate mean a discrete support, which interpolation would smear
    # across the gaps between its points.
    q = np.repeat(np.linspace(-1.0, 1.0, 4), 16)
    with pytest.raises(ValueError, match="appears to be discrete"):
        DistributionTable(q=q)
    assert DistributionTable(q=q, discrete=True).n_knots == 64


# --- what the tabulating routes refuse --------------------------------------


def test_a_ppf_must_answer_one_value_per_probability() -> None:
    with pytest.raises(ValueError, match="return one value per probability"):
        DistributionTable.from_callable(lambda u: np.zeros(3))


@pytest.mark.parametrize(
    "ppf",
    [
        lambda u: np.zeros_like(u),
        lambda u: np.where(u > 0.9, np.nan, 0.0),
    ],
    ids=["no spread", "non-finite"],
)
def test_knots_with_no_usable_spread_cannot_be_standardized(ppf) -> None:
    # Standardizing divides by this spread, so the table would be the division's
    # result rather than a distribution.
    with pytest.raises(ValueError, match="cannot be standardized"):
        DistributionTable.from_callable(ppf)


def test_a_rounded_ppf_is_refused_through_the_shock_route_too() -> None:
    # The ties check sits in the constructor, so it fires wherever the knots are
    # built, not only where they are handed over.
    with pytest.raises(ValueError, match="appears to be discrete"):
        Shock.from_ppf(lambda u: np.round(norm.ppf(u)))


def test_the_callable_route_cannot_ask_and_so_does_not_refuse() -> None:
    # Same distribution, no object to query: from_callable standardizes by whatever
    # spread the outermost knots happen to have. The caller owns that question, which
    # is why Shock.from_ppf accepts a df the family route rejects.
    assert DistributionTable.from_callable(cauchy.ppf).n_knots == 1024
    assert Shock.from_ppf(scipy_t, df=1.0).joint("e_u").target == ("e_u",)


# --- what standardization guarantees ----------------------------------------


@pytest.mark.parametrize(
    "dist", [norm, laplace, scipy_t(5)], ids=["norm", "laplace", "t5"]
)
def test_a_tabulated_distribution_reads_back_standardized(dist) -> None:
    # The scale a shock is drawn at comes from the model's calibration, so the table
    # itself must carry none.
    mean, var = _mc(DistributionTable.from_callable(dist.ppf))
    assert abs(mean) < 0.01
    assert abs(var - 1.0) < 0.02


@pytest.mark.parametrize(
    "ppf",
    [lambda u: np.round(norm.ppf(u)), lambda u: poisson.ppf(u, mu=3)],
    ids=["rounded norm", "poisson"],
)
def test_a_discrete_distribution_standardizes_under_its_cell_weights(ppf) -> None:
    # A discrete table is read by nearest knot, so the moments it is standardized by
    # are the knots' own cells rather than the trapezoids between them.
    mean, var = _mc(DistributionTable.from_callable(ppf, discrete=True))
    assert abs(mean) < 0.01
    assert abs(var - 1.0) < 0.02


def test_a_location_baked_into_the_ppf_washes_out() -> None:
    # Standardizing subtracts the mean the knots carry, which is what lets Shock keep
    # loc as a kwarg and apply it once at resolution instead.
    shifted = DistributionTable.from_callable(lambda u: laplace.ppf(u, loc=7.5))
    np.testing.assert_allclose(
        shifted.q, DistributionTable.from_callable(laplace.ppf).q, atol=1e-12
    )


def test_the_normal_table_is_its_own_grid() -> None:
    # Phi^-1 sampled on a probit grid is that grid, and standardizing something
    # already standard leaves it alone. What is left is the knot spacing.
    table = DistributionTable.from_callable(norm.ppf)
    np.testing.assert_allclose(table.q, table.z, atol=1e-4)


@pytest.mark.parametrize("n_knots", [2, 17, 4096])
def test_the_knot_count_is_whatever_the_caller_asked_for(n_knots) -> None:
    assert DistributionTable.from_callable(norm.ppf, n_knots=n_knots).n_knots == n_knots


# --- the grid the knots sit on ----------------------------------------------


def test_the_grid_spans_zmax_at_the_declared_resolution() -> None:
    table = DistributionTable(q=np.linspace(-2.0, 2.0, 5))
    assert table.zmax == ZMAX
    assert table.z[0] == -ZMAX
    assert table.z[-1] == ZMAX
    assert table.z.shape == (5,)
    assert table.dz == 3.0


def test_the_lookup_factor_inverts_the_spacing() -> None:
    # The kernel multiplies by inv_dz where this side divides by dz, so the two have
    # to agree exactly rather than nearly.
    table = DistributionTable(q=np.linspace(-1.0, 1.0, 64))
    assert table.dz * table.inv_dz == 1.0


# --- reading the table ------------------------------------------------------


def test_reading_at_a_knot_returns_that_knot() -> None:
    table = DistributionTable.from_callable(laplace.ppf, n_knots=128)
    np.testing.assert_allclose(table.evaluate(table.z), table.q, atol=1e-12)


@pytest.mark.parametrize("z", [-ZMAX, -1e3, -np.inf])
def test_a_draw_past_the_grid_clamps_to_the_end_knot(z) -> None:
    # The clamp is what bounds the table: a draw in the far tail reads the outermost
    # knot instead of extrapolating past it.
    table = DistributionTable.from_callable(laplace.ppf, n_knots=128)
    assert table.evaluate(z) == table.q[0]
    assert table.evaluate(-z) == table.q[-1]


@pytest.mark.parametrize("shape", [(), (5,), (3, 4), (2, 3, 4)])
def test_a_read_keeps_the_shape_it_was_given(shape) -> None:
    table = DistributionTable.from_callable(norm.ppf, n_knots=32)
    assert table.evaluate(np.zeros(shape)).shape == shape


def test_a_discrete_table_snaps_where_a_continuous_one_interpolates() -> None:
    q = np.array([-1.0, 0.0, 1.0])
    between = 1.0  # a sixth of the way from the middle knot to the last
    assert DistributionTable(q=q, discrete=True).evaluate(between) == 0.0
    assert DistributionTable(q=q).evaluate(between) == pytest.approx(1.0 / 6.0)


# --- the stored form --------------------------------------------------------


def test_the_stored_form_carries_the_grid_and_an_id_over_the_knots() -> None:
    params = DistributionTable.from_callable(laplace.ppf, n_knots=64).to_params()
    assert params["zmax"] == ZMAX
    assert params["discrete"] is False
    assert len(params["table_id"]) == 16


def test_the_id_follows_the_knots_and_nothing_else() -> None:
    # Two authors tabulating the same distribution share one member in a bundle,
    # which is what the id is for.
    table = DistributionTable.from_callable(laplace.ppf, n_knots=64)
    same = DistributionTable.from_callable(laplace.ppf, n_knots=64)
    assert same.to_params()["table_id"] == table.to_params()["table_id"]

    for other in (
        DistributionTable.from_callable(laplace.ppf, n_knots=65),
        DistributionTable.from_callable(norm.ppf, n_knots=64),
    ):
        assert other.to_params()["table_id"] != table.to_params()["table_id"]


def test_the_repr_names_the_kind_and_the_resolution() -> None:
    q = np.linspace(-1.0, 1.0, 8)
    assert repr(DistributionTable(q=q)) == "DistributionTable(continuous, n_knots=8)"
    assert (
        repr(DistributionTable(q=q, discrete=True))
        == "DistributionTable(discrete, n_knots=8)"
    )


# --- through a solved model -------------------------------------------------


def test_a_tabulated_shock_drives_a_simulation_deterministically(solved_test) -> None:
    shock = Shock.from_ppf(laplace, seed=3).joint("e_u")
    first = solved_test.sim(T=12, shocks=[shock]).states["u"]
    np.testing.assert_allclose(first, solved_test.sim(T=12, shocks=[shock]).states["u"])
    # The table is the whole distribution, so the same seed through the analytic
    # family is a different path rather than the same one.
    family = solved_test.sim(T=12, shocks=[Shock("norm", seed=3).joint("e_u")])
    assert not np.allclose(first, family.states["u"])


def test_the_resolved_entry_points_at_the_table_it_was_built_from(
    solved_test,
) -> None:
    # One table serves every entry that names it; the plan carries the reference
    # rather than a copy, which is what the bundle's pooling relies on.
    shock = Shock.from_ppf(laplace, seed=1).joint("e_u")
    (entry,) = resolve_shock_plan(
        solved_test.config, solved_test.compiled, [shock], 8
    ).entries
    assert entry.table is shock.dist


def test_one_template_shares_its_table_across_every_bind() -> None:
    template = Shock.from_ppf(laplace, seed=0)
    first, second = template.independent("e_u", "e_v")
    assert first.dist is second.dist is template.dist


# --- the measure the moments are taken under --------------------------------


@pytest.mark.parametrize("knots", [3, 64, 1024])
def test_the_nearest_knot_cells_partition_the_normal_measure(knots) -> None:
    # A discrete read assigns each knot its own cell, and the end knots absorb the
    # clamped tails. Weights that did not sum to one would standardize against a
    # measure the kernel does not sample from.
    z, _ = _knot_probabilities(ZMAX, knots)
    assert float(_cell_weights(z).sum()) == pytest.approx(1.0, abs=1e-12)


def test_the_grid_maps_onto_the_probabilities_it_tabulates() -> None:
    z, u = _knot_probabilities(ZMAX, 128)
    assert z[0] == -ZMAX
    assert z[-1] == ZMAX
    assert np.all(np.diff(u) > 0.0)
    assert 0.0 < u[0]
    assert u[-1] < 1.0
