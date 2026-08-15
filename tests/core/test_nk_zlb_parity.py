"""Dynare parity for the one-constraint OccBin NK model: the whole simulated path.

``tests/fixtures/models/nk_zlb_1_constraint.yaml`` is our twin of
``nk_zlb_1_constraint.mod``, Willi Mutschler's Dynare implementation of the model
Guerrieri and Iacoviello (2015) plot in figure 5, and
:mod:`_oracles.dynare_nk_zlb_1_constraint` carries what Dynare 7.1 produced from
it: 30 periods around a discount-factor shock that drives the nominal rate into
the zero lower bound, the piecewise path, the unconstrained path, and the
realized regime at every date.

This is the one-constraint half of what
:mod:`tests.core.test_occbin_parity` does with two. The two-constraint fixture
reaches what a single bound cannot, interacting constraints and a mask that
changes twelve times; this one reaches what that fixture cannot, a run whose
constraint pins a variable to an exact value while it binds and where Dynare
takes the ``constraint_nbr == 1`` branch of its own solver rather than the
two-constraint one. Neither substitutes for the other.

The run is driven the way a user drives it, ``solve`` then ``sim``, and compared
against the golden with no conversion on either side: Dynare reports levels and
so does :attr:`SimResult.X`.

The horizon is Dynare's ``simul_check_ahead_periods=200``, plus the date a
binding guess appends, so 201. Dynare leaves ``max_check_ahead_periods`` at
infinity, so its guess may grow; ours is capped, and the run is checked to never
have needed the room.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser
from _oracles import dynare_nk_zlb_1_constraint as golden

_MODEL = "tests/fixtures/models/nk_zlb_1_constraint.yaml"

#: Our name for each of the oracle's ``COLUMNS``, in that order. The two models
#: declare their variables in the same order, so this is a copy rather than a
#: permutation, but the path is still read by name.
_OUR_COLUMNS = (
    "bet",
    "c",
    "y",
    "l",
    "w",
    "mc",
    "r",
    "g",
    "pie",
    "pie_star",
    "x1",
    "x2",
    "v",
    "pie_an",
    "r_an",
    "yhat",
)

#: ``simul_check_ahead_periods`` on the golden run.
_CHECK_AHEAD = 200

#: Room for the growth Dynare allows and this run turns out not to need.
_MAX_CHECK_AHEAD = _CHECK_AHEAD + 50

#: The buffer the two above imply: the horizon plus the appended release date.
_T0 = _CHECK_AHEAD + 1

#: Neither side holds an exact steady state to be compared against: ``x1`` and
#: ``x2`` are ill conditioned in double precision, so Dynare's own values are 63
#: and 75 ulps from a 50-digit evaluation of its ``steady_state_model`` and ours
#: are further out still. The oracle's docstring carries the arithmetic. That
#: conditioning is the floor here, together with the 400x and 100x rescalings
#: ``r_an``, ``pie_an`` and ``yhat`` apply, which turn ``r``'s 4e-16 into 2e-13
#: by arithmetic rather than by error. Every other column agrees to 5e-15 or
#: better, so these numbers are not measuring the dynamics: the worst element
#: sits at 4.5e-13 and the tolerances below leave it a factor of six.
_RTOL = 1e-13
_ATOL = 1e-12


@pytest.fixture(scope="module")
def solved():
    """The model a user gets from ``solve``: the constraint makes it piecewise."""
    model, kalman = ModelParser(_MODEL).get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


@pytest.fixture(scope="module")
def result(solved):
    """Dynare's surprise sequence run through the public simulation path."""
    return solved.sim(
        golden.T,
        shocks={"epsi": np.ravel(golden.SHOCKS)},
        check_ahead_periods=_CHECK_AHEAD,
        max_check_ahead_periods=_MAX_CHECK_AHEAD,
    )


def _columns(levels, compiled):
    """Our path restricted to the oracle's columns, in the oracle's order."""
    idx = compiled.layout.idx
    return np.stack([levels[:, idx[name]] for name in _OUR_COLUMNS], axis=1)


def test_the_reference_steady_state_is_dynares(solved):
    idx = solved.compiled.layout.idx
    ss = solved.policy.steady_state
    ours = np.array([ss[idx[name]] for name in _OUR_COLUMNS])

    np.testing.assert_allclose(ours, golden.YS, rtol=_RTOL, atol=_ATOL)


def test_the_unconstrained_path_is_dynares_linear_simulation(solved):
    # oo_.occbin.simul.linear: the same surprise sequence with the constraint
    # ignored, so it settles the reference pencil before any regime logic runs.
    # `sim_reference` is that regime driven as an ordinary first-order model,
    # which is the same object Dynare builds this path from.
    ours = _columns(
        solved.sim_reference(golden.T, shocks={"epsi": np.ravel(golden.SHOCKS)}).X,
        solved.compiled,
    )

    np.testing.assert_allclose(ours, golden.LINEAR, rtol=_RTOL, atol=_ATOL)


def test_the_piecewise_path_is_dynares(solved, result):
    ours = _columns(result.X, solved.compiled)

    np.testing.assert_allclose(ours, golden.PIECEWISE, rtol=_RTOL, atol=_ATOL)


def test_the_realized_regime_is_dynares(result):
    # Date 0 of each period's accepted guess is what that period realizes; the
    # rest is the expectation the guess was solved under.
    np.testing.assert_array_equal(result.regimes[:, 0], golden.REALIZED_MASK)


def test_the_bound_holds_exactly_while_it_binds(solved, result):
    # What a single constraint reaches that the two-constraint fixture does not:
    # the binding regime replaces the policy rule with `r = ZLB` outright, so on
    # a binding date the rate is the bound to the last bit rather than merely
    # near it. Anything else means the patched row is not the one being solved.
    binds = result.regimes[:, 0] == 1
    rate = result.X[:, solved.compiled.layout.idx["r"]]
    zlb = solved.compiled.config.calibration.parameters["ZLB"]

    assert binds.any()
    np.testing.assert_array_equal(rate[binds], np.full(binds.sum(), zlb))
    assert (rate[~binds] > zlb).all()


def test_the_run_settles_well_inside_dynares_check_ahead(result):
    # Dynare would have grown the horizon past 200 rather than clip it, so a
    # period that reached the cap would make the two runs incomparable.
    diag = result.diagnostics

    np.testing.assert_array_equal(diag.T_used, _T0)
    assert 1 < diag.iters.max() < 30
