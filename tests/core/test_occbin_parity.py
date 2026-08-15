"""Dynare parity for the two-constraint OccBin RBC: the whole simulated path.

``tests/fixtures/models/rbc_occbin.yaml`` is our twin of Dynare's own
``rbc_occbin.mod``, and :mod:`_oracles.dynare_rbc_occbin` carries what Dynare 7.1
produced from it: 169 periods of unforeseen shocks, the piecewise path, the
unconstrained path, and the realized regime at every date.

This is the only place the OccBin driver meets an answer it did not compute
itself. The kernel tests check that the accepted guess is a fixed point of the
latch and that the path is the forward pass of that guess, which any
self-consistent solver satisfies; only a golden says the fixed point is the same
one Dynare lands on. It reaches what synthetic fixtures cannot: two constraints
interacting, a mask that changes 12 times over the run, and shock periods that
inherit a guess from the period before rather than starting relaxed.

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
from _oracles import dynare_rbc_occbin as golden


_MODEL = "tests/fixtures/models/rbc_occbin.yaml"

#: Our name for each of the oracle's ``COLUMNS``, in that order.
_OUR_COLUMNS = ("a", "c", "invest", "k", "lam", "log_k", "log_invest", "log_c")

#: ``simul_check_ahead_periods`` on the golden run.
_CHECK_AHEAD = 200

#: Room for the growth Dynare allows and this run turns out not to need.
_MAX_CHECK_AHEAD = _CHECK_AHEAD + 50

#: The buffer the two above imply: the horizon plus the appended release date.
_T0 = _CHECK_AHEAD + 1


@pytest.fixture(scope="module")
def solved():
    """The model a user gets from ``solve``: constraints make it piecewise."""
    model, kalman = ModelParser(_MODEL).get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


@pytest.fixture(scope="module")
def result(solved):
    """Dynare's surprise sequence run through the public simulation path."""
    return solved.sim(
        golden.T,
        shocks={"eps": np.ravel(golden.SHOCKS)},
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

    np.testing.assert_allclose(ours, golden.YS, rtol=1e-14, atol=1e-14)


def test_the_unconstrained_path_is_dynares_linear_simulation(solved):
    # oo_.occbin.simul.linear: the same surprise sequence with the constraints
    # ignored, so it settles the reference pencil before any regime logic runs.
    # `ref` is that regime as an ordinary first-order solution, so the path is
    # its own state space, `y_t = A y_{t-1} + B eps_t`.
    ref = solved.policy.ref
    head = golden.LINEAR_HEAD.shape[0]
    eps = np.ravel(golden.SHOCKS)[:head, None]

    y = np.zeros(solved.compiled.n_var)
    rows = []
    for t in range(head):
        y = ref.A @ y + ref.B @ eps[t]
        rows.append(y)

    ours = _columns(np.array(rows) + ref.steady_state, solved.compiled)

    np.testing.assert_allclose(ours, golden.LINEAR_HEAD, rtol=1e-12, atol=1e-12)


def test_the_piecewise_path_is_dynares(solved, result):
    ours = _columns(result.X, solved.compiled)

    np.testing.assert_allclose(ours, golden.PIECEWISE, rtol=1e-11, atol=1e-11)


def test_the_realized_regime_is_dynares(result):
    # Date 0 of each period's accepted guess is what that period realizes; the
    # rest is the expectation the guess was solved under.
    np.testing.assert_array_equal(result.regimes[:, 0], golden.REALIZED_MASK)


def test_the_run_settles_well_inside_dynares_check_ahead(result):
    # Dynare would have grown the horizon past 200 rather than clip it, so a
    # period that reached the cap would make the two runs incomparable.
    diag = result.diagnostics

    np.testing.assert_array_equal(diag.T_used, _T0)
    assert 1 < diag.iters.max() < 30
