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

The horizon is Dynare's ``simul_check_ahead_periods=200``, plus the date a
binding guess appends, so 201. Dynare leaves ``max_check_ahead_periods`` at
infinity, so its guess may grow; ours is capped, and the run is checked to never
have needed the room.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE._ckernels.core import klein_preprocess, klein_solve1
from SymbolicDSGE._ckernels.occbin import occbin_sim, regime_pencil
from SymbolicDSGE.core import DSGESolver, ModelParser
from _oracles import dynare_rbc_occbin as golden

_MODEL = "tests/fixtures/models/rbc_occbin.yaml"

#: Our name for each of the oracle's ``COLUMNS``, in that order.
_OUR_COLUMNS = ("a", "c", "invest", "k", "lam", "log_k", "log_invest", "log_c")

#: ``simul_check_ahead_periods`` on the golden run, and the appended date.
_T0 = 201

#: Room for the growth Dynare allows and this run turns out not to need.
_T_CAP = _T0 + 50


@pytest.fixture(scope="module")
def compiled():
    model, kalman = ModelParser(_MODEL).get_all()
    return DSGESolver(model, kalman).compile()


@pytest.fixture(scope="module")
def par(compiled):
    calib = compiled.config.calibration.parameters
    return np.array([float(calib[p]) for p in compiled.calib_params])


@pytest.fixture(scope="module")
def solved(compiled, par):
    """(ss, a_ref, b_ref, f_ref, p_ref) for the all-relaxed reference regime."""
    seed = DSGESolver._resolve_ss_seed(None, compiled)
    addr = compiled.construct_objective_cfunc().address
    ss, f_ref, p_ref, *_ = klein_solve1(
        addr,
        seed,
        par,
        compiled.n_state,
        compiled.n_exog,
    )
    # The solve keeps the pencil internal, so take it at the steady state the
    # solve resolved: the same linearization, one call later.
    a_ref, b_ref = klein_preprocess(addr, ss, par, compiled.n_var)
    return ss, a_ref, b_ref, f_ref, p_ref


@pytest.fixture(scope="module")
def table(compiled, par, solved):
    """(a, b, c) stacked by bitmask, IRR at bit 0 and INEG at bit 1."""
    ss, a_ref, b_ref, _, _ = solved
    func = compiled.construct_regime_pencil_func()
    a = [a_ref]
    b = [b_ref]
    c = [np.zeros(a_ref.shape[0])]
    for mask in (1, 2, 3):
        a_m, b_m, c_m = regime_pencil(
            func.address(mask), func.rows[mask], ss, par, a_ref, b_ref
        )
        a.append(a_m)
        b.append(b_m)
        c.append(c_m)
    return np.stack(a), np.stack(b), np.stack(c)


@pytest.fixture(scope="module")
def shocks(solved):
    """Dynare's surprise sequence, in the slot the TFP innovation enters."""
    out = np.zeros((golden.T, solved[3].shape[1]))
    out[:, 0] = np.ravel(golden.SHOCKS)
    return out


@pytest.fixture(scope="module")
def piecewise(compiled, table, solved, par, shocks):
    """(levels, regimes, diag) from the native driver over the golden shocks."""
    a, b, c = table
    ss, _, _, f_ref, _ = solved
    cf = compiled.construct_constraint_func()
    out, regimes, diag = occbin_sim(
        a,
        b,
        c,
        f_ref,
        ss,
        par,
        cf.address,
        cf.n_constraint,
        cf.inclusive,
        shocks,
        np.zeros(f_ref.shape[1]),
        T0=_T0,
        T_cap=_T_CAP,
    )
    return out + ss, regimes, diag


def _columns(levels, compiled):
    """Our path restricted to the oracle's columns, in the oracle's order."""
    idx = compiled.layout.idx
    return np.stack([levels[:, idx[name]] for name in _OUR_COLUMNS], axis=1)


def test_the_reference_steady_state_is_dynares(compiled, solved):
    ss = solved[0]
    idx = compiled.layout.idx
    ours = np.array([ss[idx[name]] for name in _OUR_COLUMNS])

    np.testing.assert_allclose(ours, golden.YS, rtol=1e-14, atol=1e-14)


def test_the_unconstrained_path_is_dynares_linear_simulation(compiled, solved, shocks):
    # oo_.occbin.simul.linear: the same surprise sequence with the constraints
    # ignored, so it settles the reference pencil before any regime logic runs.
    ss, _, _, f_ref, p_ref = solved
    x = np.zeros(f_ref.shape[1])
    rows = []
    for t in range(golden.LINEAR_HEAD.shape[0]):
        x = x + shocks[t]
        rows.append(np.concatenate([x, f_ref @ x]))
        x = p_ref @ x

    ours = _columns(np.array(rows) + ss, compiled)

    np.testing.assert_allclose(ours, golden.LINEAR_HEAD, rtol=1e-12, atol=1e-12)


def test_the_piecewise_path_is_dynares(compiled, piecewise):
    levels, _, _ = piecewise

    ours = _columns(levels, compiled)

    np.testing.assert_allclose(ours, golden.PIECEWISE, rtol=1e-11, atol=1e-11)


def test_the_realized_regime_is_dynares(piecewise):
    # Date 0 of each period's accepted guess is what that period realizes; the
    # rest is the expectation the guess was solved under.
    _, regimes, _ = piecewise

    np.testing.assert_array_equal(regimes[:, 0], golden.REALIZED_MASK)


def test_the_run_settles_well_inside_dynares_check_ahead(piecewise):
    # Dynare would have grown the horizon past 200 rather than clip it, so a
    # period that reached the cap would make the two runs incomparable.
    _, _, diag = piecewise

    np.testing.assert_array_equal(diag["T_used"], _T0)
    assert 1 < diag["iters"].max() < 30
