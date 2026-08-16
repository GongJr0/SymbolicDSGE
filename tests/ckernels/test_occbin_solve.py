# type: ignore
"""Native ``occbin_forward`` and ``occbin_sim``: the path and the shock loop.

The forward pass has an oracle, so it is checked against one. The solve is
checked three ways here instead. The accepted mask must equal the condition read
off the path that mask produced, which is what being a fixed point of the latch
means. The path must equal ``occbin_forward`` over ``occbin_recursion`` on that
same mask, which ties the driver to the two kernels it is built from. And a
period with no shock must reproduce, bit for bit, what the period before it
already projected for that date. Whether the fixed point is the same one Dynare
lands on is ``tests/core/test_occbin_parity.py``.

The fixture puts the threshold at a nonzero TFP level rather than at zero. With
``z(t) < 0`` an AR(1) hit decays toward the boundary without ever crossing it,
so a guess binds to the horizon cap and never settles; a threshold gives a spell
whose length the model decides. That length is also knowable here: the regime
replaces the Euler equation only, so TFP is exogenous to the regime and its path
is the same under every guess.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pytest
import sympy as sp
from numba import carray, cfunc, types

from SymbolicDSGE._ckernels.core import klein_preprocess, klein_solve1
from SymbolicDSGE._ckernels.occbin._occbin import (
    occbin_forward,
    occbin_recursion,
    occbin_sim,
    occbin_sim_arena_size,
    regime_pencil,
)
from SymbolicDSGE.core import DSGESolver, ModelParser

from SymbolicDSGE.core.config import Constraint

t = sp.Symbol("t", integer=True)

LOW = 0b1

#: TFP level the regime switches on, and a hit that carries the path past it.
THRESHOLD = -0.005
SHOCK = -0.01

#: Long enough that the spell ends well inside it, and a start that must grow.
HORIZON = 29
SHORT = 7

#: The buffer widths those horizons imply: one release date past the guess.
#: Masks, ``T_used`` and any path meant to line up with a mask are this long.
HORIZON_T = HORIZON + 1
SHORT_T = SHORT + 1

# A binding run, a relapse, and a relaxed tail.
MIXED = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)


@pytest.fixture(scope="module")
def compiled(rbc_second_order_test_model_path):
    """Levels RBC where TFP below a threshold shuts investment off."""
    model, kalman = ModelParser(rbc_second_order_test_model_path).get_all()
    conf = copy.deepcopy(model)
    _, k, z = conf.variables.variables
    delta = sp.Symbol("delta")

    conf.equations.constraint = {
        "low": Constraint(bind=z(t) < THRESHOLD, relax=z(t) >= THRESHOLD)
    }
    conf.equations.regime = {
        frozenset({"low"}): {"euler": sp.Eq(k(t + 1), (1 - delta) * k(t))}
    }
    return DSGESolver(conf, kalman).compile()


@pytest.fixture(scope="module")
def par(compiled):
    calib = compiled.config.calibration.parameters
    return np.array([float(calib[p]) for p in compiled.calib_params])


@pytest.fixture(scope="module")
def cf(compiled):
    return compiled.construct_constraint_func()


@pytest.fixture(scope="module")
def solved(compiled, par):
    """(ss, reference blocks, ghx, B) for the reference regime."""
    seed = DSGESolver._resolve_ss_seed(None, compiled)
    addr = compiled.construct_objective_cfunc().address
    ss, f, p, _, _, _, B = klein_solve1(
        addr,
        seed,
        par,
        compiled._incidence,
        compiled.n_state,
        compiled.n_exog,
    )
    # The solve keeps the pencil internal, so take it at the steady state the
    # solve resolved: the same linearization, one call later.
    ref = klein_preprocess(addr, ss, par, compiled.n_var, compiled.n_exog)
    return ss, ref, np.vstack([p, f]), B


@pytest.fixture(scope="module")
def table(compiled, par, solved):
    """(a, b, c, d, cst) by bitmask: slot 0 the reference, slot 1 the regime."""
    ss, ref, _, _ = solved
    func = compiled.construct_regime_pencil_func()
    slots = [
        regime_pencil(0, np.empty(0, dtype=np.int64), ss, par, *ref),
        regime_pencil(func.address(LOW), func.rows[LOW], ss, par, *ref),
    ]
    return tuple(np.stack([lo[i] for lo in slots]) for i in range(5))


@pytest.fixture(scope="module")
def z_pos(compiled):
    """Where TFP sits in the current-variable vector the constraint reads."""
    return compiled.layout.idx["z"]


def _shocks(solved, n_period, size=SHOCK):
    """``(n_period, n_exog)`` carrying one hit, in the leading innovation."""
    out = np.zeros((n_period, solved[3].shape[1]))
    out[0, 0] = size
    return out


def _binding(out, solved, z_pos):
    """The condition as the cfunc sees it: a level, not a deviation."""
    ss = solved[0]
    return (out[:, z_pos] + ss[z_pos] < THRESHOLD).astype(np.int8)


def _solve(table, solved, par, cond_addr, n_constraint, inclusive, shocks, **kw):
    # Growth is off unless a test asks for it. Letting a guess grow moves
    # iteration counts and switches cycle detection off, which is noise for
    # every test here except the two that are about growth.
    if "check_ahead_periods" in kw:
        kw.setdefault("max_check_ahead_periods", kw["check_ahead_periods"])
    ss, _, ghx, _ = solved
    return occbin_sim(
        *table,
        ghx,
        ss,
        par,
        cond_addr,
        n_constraint,
        inclusive,
        shocks,
        np.zeros(ghx.shape[1]),
        **kw,
    )


def _run(table, solved, par, cf, shocks, **kw):
    return _solve(
        table, solved, par, cf.address, cf.n_constraint, cf.inclusive, shocks, **kw
    )


def test_the_path_is_each_dates_block_on_the_date_before(table, solved):
    # Row t is the whole of y_t out of date t's block, and its state half is
    # what date t + 1 reads back as its lag. The innovation lands on date 0
    # alone, so every later date runs on the state carry only.
    _, _, ghx, B = solved
    n_state, n_exog = ghx.shape[1], B.shape[1]
    rule = occbin_recursion(*table, MIXED, ghx)
    x0 = np.arange(1.0, n_state + 1.0)
    eps0 = np.full(n_exog, SHOCK)

    path = occbin_forward(rule, x0, eps0)

    x = x0.copy()
    for date in range(rule.shape[0]):
        eps = eps0 if date == 0 else np.zeros(n_exog)
        y = (
            rule[date, :, :n_state] @ x
            + rule[date, :, n_state : n_state + n_exog] @ eps
            + rule[date, :, -1]
        )
        np.testing.assert_allclose(path[date], y, rtol=1e-9, atol=1e-11)
        x = y[:n_state]


def test_an_all_relaxed_path_walks_the_reference_state_space(table, solved):
    # With no regime binding the path is the reference model's own recursion,
    # `y_t = ghx x_{t-1} + B eps_t`, which the solve produced independently.
    _, _, ghx, B = solved
    n_state, n_exog = ghx.shape[1], B.shape[1]
    rule = occbin_recursion(*table, np.zeros(8, dtype=np.int8), ghx)
    x0 = np.arange(1.0, n_state + 1.0)
    eps0 = np.full(n_exog, SHOCK)

    path = occbin_forward(rule, x0, eps0)

    y = ghx @ x0 + B @ eps0
    for date in range(8):
        np.testing.assert_allclose(path[date], y, rtol=1e-9, atol=1e-11)
        y = ghx @ y[:n_state]


def test_forward_writes_out_in_place(table, solved):
    _, _, ghx, _ = solved
    rule = occbin_recursion(*table, MIXED, ghx)
    n_var = rule.shape[1]
    buf = np.zeros((MIXED.size, n_var))

    got = occbin_forward(rule, np.ones(ghx.shape[1]), out=buf)

    assert got is buf
    assert np.abs(buf).max() > 0.0


def test_forward_rejects_an_initial_state_wider_than_the_model(table, solved):
    _, _, ghx, _ = solved
    rule = occbin_recursion(*table, MIXED, ghx)

    with pytest.raises(ValueError, match="more than rule's"):
        occbin_forward(rule, np.ones(rule.shape[1] + 1))


def test_forward_rejects_an_innovation_of_the_wrong_width(table, solved):
    _, _, ghx, B = solved
    rule = occbin_recursion(*table, MIXED, ghx)

    with pytest.raises(ValueError, match="to match rule's shock columns"):
        occbin_forward(rule, np.ones(ghx.shape[1]), np.ones(B.shape[1] + 1))


def test_an_empty_rule_stack_is_a_no_op(table, solved):
    _, _, ghx, B = solved
    n_var, n_state, n_exog = ghx.shape[0], ghx.shape[1], B.shape[1]

    path = occbin_forward(np.empty((0, n_var, n_state + n_exog + 1)), np.ones(n_state))

    assert path.shape == (0, n_var)


def test_the_steady_state_is_a_fixed_point(table, solved, par, cf):
    # Nothing moves, so the constraint is read at the steady state, where the
    # distance is exactly zero and a strict condition does not fire.
    n_exog = solved[3].shape[1]

    out, regimes, diag = _run(
        table, solved, par, cf, np.zeros((1, n_exog)), check_ahead_periods=SHORT
    )

    np.testing.assert_array_equal(out, np.zeros_like(out))
    np.testing.assert_array_equal(regimes, np.zeros_like(regimes))
    # One pass to read the path, none to correct it.
    np.testing.assert_array_equal(diag["iters"], [1])
    np.testing.assert_array_equal(diag["max_err"], [0.0])
    np.testing.assert_array_equal(diag["T_used"], [SHORT_T])


def test_the_mask_is_the_condition_read_off_its_own_path(table, solved, par, cf, z_pos):
    # The accepted guess is a fixed point of the latch, so date by date it has
    # to agree with the condition evaluated on the path it produced.

    out, regimes, diag = _run(
        table,
        solved,
        par,
        cf,
        _shocks(solved, 1),
        check_ahead_periods=HORIZON,
        n_periods=HORIZON_T,
    )

    binding = _binding(out, solved, z_pos)
    # The spell has to start and end inside the horizon for this to say much.
    assert 3 < binding.sum() < HORIZON_T
    np.testing.assert_array_equal(regimes[0], binding)
    np.testing.assert_array_equal(diag["T_used"], [HORIZON_T])
    # TFP is exogenous to the regime, so the first latch already lands the
    # answer and the second only confirms it.
    np.testing.assert_array_equal(diag["iters"], [2])


def test_the_path_is_the_forward_pass_of_the_accepted_guess(table, solved, par, cf):
    _, _, ghx, _ = solved
    shocks = _shocks(solved, 1)

    out, regimes, diag = _run(
        table,
        solved,
        par,
        cf,
        shocks,
        check_ahead_periods=HORIZON,
        n_periods=HORIZON_T,
    )

    used = int(diag["T_used"][0])
    rule = occbin_recursion(*table, regimes[0][:used], ghx)
    x0 = np.zeros(ghx.shape[1])

    # The same two kernels on the same inputs, so this is exact up to the LU.
    np.testing.assert_allclose(
        out, occbin_forward(rule, x0, shocks[0]), rtol=1e-13, atol=1e-13
    )


def test_a_period_without_a_shock_shifts_instead_of_re_solving(table, solved, par, cf):
    # Dynare's shockless branch. The previous guess and the path it implies
    # already describe this period, so shifting must reproduce them exactly.
    periods = 5

    many_out, many_reg, many_diag = _run(
        table,
        solved,
        par,
        cf,
        _shocks(solved, periods),
        check_ahead_periods=HORIZON,
        n_periods=periods,
    )
    one_out, one_reg, one_diag = _run(
        table,
        solved,
        par,
        cf,
        _shocks(solved, 1),
        check_ahead_periods=HORIZON,
        n_periods=periods,
    )

    # memmove only: no arithmetic stands between these two.
    np.testing.assert_array_equal(many_out, one_out)
    np.testing.assert_array_equal(
        many_diag["iters"], [one_diag["iters"][0], 0, 0, 0, 0]
    )
    for s in range(periods):
        np.testing.assert_array_equal(many_reg[s][: HORIZON_T - s], one_reg[0][s:])
        np.testing.assert_array_equal(many_reg[s][HORIZON_T - s :], 0)


def test_a_short_horizon_grows_into_the_same_answer(table, solved, par, cf, z_pos):
    shocks = _shocks(solved, 1)

    long_out, _, _ = _run(
        table,
        solved,
        par,
        cf,
        shocks,
        check_ahead_periods=HORIZON,
        n_periods=HORIZON_T,
    )
    short_out, _, short_diag = _run(
        table,
        solved,
        par,
        cf,
        shocks,
        check_ahead_periods=SHORT,
        max_check_ahead_periods=HORIZON,
        n_periods=SHORT_T,
    )

    # Growth stops the first date the guess relaxes, and the spell runs from
    # date 0, so the settled horizon is one past its length.
    spell = int(_binding(long_out, solved, z_pos).sum())
    assert SHORT < spell + 1 < HORIZON
    np.testing.assert_array_equal(short_diag["T_used"], [spell + 1])
    # A relaxed tail is the reference rule's own fixed point, so where the two
    # horizons close does not reach the dates either of them reports.
    np.testing.assert_allclose(
        short_out, long_out[: short_out.shape[0]], rtol=1e-8, atol=1e-10
    )


_COND_SIG = types.void(
    types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.CPointer(types.float64),
)


@cfunc(_COND_SIG)
def _always_fires(cur, par, err):
    """Every condition holds, so every date flips on every pass."""
    e = carray(err, (2,))
    e[0] = 1.0
    e[1] = 1.0


def test_a_guess_that_never_settles_reports_the_iteration_cap(table, solved, par):

    with pytest.raises(RuntimeError, match="did not converge in 3 iterations"):
        _solve(
            table,
            solved,
            par,
            _always_fires.address,
            1,
            0,
            _shocks(solved, 1),
            check_ahead_periods=SHORT,
            max_iter=3,
        )


def test_an_incomplete_pencil_stack_is_rejected(table, solved, par, cf):
    # The latch emits every mask over its constraints and the kernel indexes the
    # table with it, so a missing slot would be read past the end.
    reference_only = tuple(blk[:1] for blk in table)

    with pytest.raises(ValueError, match=re.escape("cover every mask over 1")):
        _run(
            reference_only,
            solved,
            par,
            cf,
            _shocks(solved, 1),
            check_ahead_periods=SHORT,
        )


def test_a_horizon_of_zero_is_rejected(table, solved, par, cf):
    # A guess needs a date to release on, and the shift reads the one before it.

    with pytest.raises(ValueError, match="at least 1"):
        _run(table, solved, par, cf, _shocks(solved, 1), check_ahead_periods=0)


def test_a_tail_longer_than_the_path_is_rejected(table, solved, par, cf):

    with pytest.raises(ValueError, match="at most S"):
        _run(
            table,
            solved,
            par,
            cf,
            _shocks(solved, 1),
            check_ahead_periods=SHORT,
            n_periods=SHORT_T + 2,
        )


def test_shocks_of_the_wrong_width_are_rejected(table, solved, par, cf):
    n_exog = solved[3].shape[1]

    with pytest.raises(ValueError, match="match the pencils' shock block"):
        _run(
            table,
            solved,
            par,
            cf,
            np.zeros((1, n_exog + 1)),
            check_ahead_periods=SHORT,
        )


def test_a_truncated_algorithm_accepts_the_guess_it_ran_out_on(table, solved, par):
    # Dynare's algo_truncation. A max_iter at or below it means the caller
    # asked for a truncated solve, so the last guess stands instead of raising.

    out, regimes, diag = _solve(
        table,
        solved,
        par,
        _always_fires.address,
        1,
        0,
        _shocks(solved, 1),
        check_ahead_periods=SHORT,
        max_iter=1,
        algo_truncation=1,
    )

    # One pass, and the guess it entered on is the relaxed one it started from.
    np.testing.assert_array_equal(diag["iters"], [1])
    np.testing.assert_array_equal(diag["T_used"], [SHORT_T])
    np.testing.assert_array_equal(regimes, np.zeros_like(regimes))
    np.testing.assert_array_equal(diag["periodic"], [0])
    assert np.abs(out).max() > 0.0


def test_a_max_iter_above_the_truncation_still_raises(table, solved, par):

    with pytest.raises(RuntimeError, match="did not converge in 1 iterations"):
        _solve(
            table,
            solved,
            par,
            _always_fires.address,
            1,
            0,
            _shocks(solved, 1),
            check_ahead_periods=SHORT,
            max_iter=1,
            algo_truncation=0,
        )


def test_seeding_the_answer_settles_in_one_pass(table, solved, par, cf):
    shocks = _shocks(solved, 1)
    kw = {"check_ahead_periods": HORIZON, "n_periods": HORIZON_T}

    out, regimes, diag = _run(table, solved, par, cf, shocks, **kw)
    seeded_out, seeded_reg, seeded_diag = _run(
        table, solved, par, cf, shocks, init_regime=regimes[:1], **kw
    )

    np.testing.assert_array_equal(seeded_out, out)
    np.testing.assert_array_equal(seeded_reg, regimes)
    # The latch reads the seed back unchanged, so there is nothing to correct.
    np.testing.assert_array_equal(diag["iters"], [2])
    np.testing.assert_array_equal(seeded_diag["iters"], [1])


def test_a_reset_regime_throws_the_seed_away(table, solved, par, cf):
    shocks = _shocks(solved, 1)
    kw = {"check_ahead_periods": HORIZON, "n_periods": HORIZON_T}
    _, regimes, _ = _run(table, solved, par, cf, shocks, **kw)

    _, reset_reg, reset_diag = _run(
        table, solved, par, cf, shocks, init_regime=regimes[:1], reset_regime=True, **kw
    )

    np.testing.assert_array_equal(reset_reg, regimes)
    # Back to reading the answer off the relaxed guess and confirming it.
    np.testing.assert_array_equal(reset_diag["iters"], [2])


def test_curb_retrench_gives_up_one_date_per_pass(table, solved, par, cf):
    # Seeded with every date binding, the latch wants to relax the whole tail
    # at once. Damping concedes the last date only, and lands the same answer.
    shocks = _shocks(solved, 1)
    kw = {"check_ahead_periods": HORIZON, "n_periods": HORIZON_T}
    seed = np.ones((1, HORIZON_T), dtype=np.int8)

    plain_out, plain_reg, plain_diag = _run(
        table, solved, par, cf, shocks, init_regime=seed, **kw
    )
    curbed_out, curbed_reg, curbed_diag = _run(
        table, solved, par, cf, shocks, init_regime=seed, curb_retrench=True, **kw
    )

    np.testing.assert_allclose(curbed_out, plain_out, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(curbed_reg, plain_reg)
    assert curbed_diag["iters"][0] > plain_diag["iters"][0]


def test_an_initial_guess_past_the_growth_cap_is_rejected(table, solved, par, cf):
    # Shorter rows relax into the tail and longer ones raise the horizon, so the
    # only illegal width is one the arena never reserved.
    shocks = _shocks(solved, 1)
    _, regimes, _ = _run(table, solved, par, cf, shocks, check_ahead_periods=HORIZON)
    too_long = np.zeros((1, regimes.shape[1] + 1), dtype=np.int8)

    with pytest.raises(ValueError, match="more than the"):
        _run(
            table,
            solved,
            par,
            cf,
            shocks,
            check_ahead_periods=HORIZON,
            init_regime=too_long,
        )


def test_an_initial_guess_outside_the_table_is_rejected(table, solved, par, cf):
    seed = np.zeros((1, SHORT_T), dtype=np.int8)
    seed[0, 0] = 2

    with pytest.raises(ValueError, match=re.escape("outside 0..1")):
        _run(
            table,
            solved,
            par,
            cf,
            _shocks(solved, 1),
            check_ahead_periods=SHORT,
            init_regime=seed,
        )


#: A shock deep enough that the spell the flip condition wants is unambiguous,
#: on a horizon short enough to read whole and capped so no guess can grow into
#: a period that stops looking for cycles.
CYCLE_SHOCK = -0.1
CYCLE_HORIZON = 11
CYCLE_T = CYCLE_HORIZON + 1

#: Where ``_flip`` reads capital and TFP. A cfunc cannot close over a fixture,
#: so the positions are baked in and ``flip_par`` checks the layout agrees.
_K_POS = 0
_Z_POS = 1


@cfunc(_COND_SIG)
def _flip(cur, par, err):
    """Bind while TFP is low, relax once the regime has driven capital high.

    Contrarian on purpose: the regime shuts investment off, which sends capital
    up, which is what the relax condition reads. A date that binds then wants to
    relax and a date that relaxes wants to bind, so the guess swings instead of
    settling. The bind side reads TFP, which no regime touches, so the swing is
    over a fixed set of leading dates and its width is the threshold's to pick.
    """
    lev = carray(cur, (_Z_POS + 1,))
    p = carray(par, (2,))
    e = carray(err, (2,))
    e[0] = p[0] - lev[_Z_POS]
    e[1] = lev[_K_POS] - p[1]


@pytest.fixture(scope="module")
def flip_par(compiled, table, solved):
    """``m -> (bind, relax)``: the levels that make ``m`` leading dates bind."""
    ss, _, ghx, _ = solved
    assert (compiled.layout.idx["k"], compiled.layout.idx["z"]) == (_K_POS, _Z_POS)

    rule = occbin_recursion(*table, np.zeros(CYCLE_HORIZON, dtype=np.int8), ghx)
    path = occbin_forward(rule, np.zeros(ghx.shape[1]), _cycle_eps(solved))
    z = path[:, _Z_POS] + ss[_Z_POS]

    def par(m):
        # Between date m - 1 and date m of a TFP path that only recovers, so
        # exactly the first m dates read as low.
        return np.array([0.5 * (z[m - 1] + z[m]), ss[_K_POS]])

    return par


def _cycle_eps(solved):
    """The innovation date 0 of the swinging period arrives on."""
    eps = np.zeros(solved[3].shape[1])
    eps[0] = CYCLE_SHOCK
    return eps


def _flip_run(table, solved, par, **kw):
    return _solve(
        table,
        solved,
        par,
        _flip.address,
        1,
        0,
        _shocks(solved, 1, CYCLE_SHOCK),
        check_ahead_periods=CYCLE_HORIZON,
        max_check_ahead_periods=CYCLE_HORIZON,
        **kw,
    )


def test_a_guess_that_swings_by_one_date_is_a_two_cycle(table, solved, flip_par):
    # The relaxed guess wants date 0 binding, and binding it sends capital up,
    # which is the relax condition, so the pass after hands the relaxed guess
    # straight back.
    with pytest.raises(RuntimeError, match="loops between two regimes"):
        _flip_run(table, solved, flip_par(1))


def test_a_guess_that_swings_by_more_is_a_loop_over_guesses(table, solved, flip_par):
    # The same swing over three dates. A repeat counts as periodic only where
    # the guess barely moved, and three dates is past the default of one.
    with pytest.raises(RuntimeError, match="loops over guess regimes"):
        _flip_run(table, solved, flip_par(3))


def test_the_periodic_threshold_counts_dates(table, solved, flip_par):
    # Not a distance the error is weighed against: raising it to the width of
    # the swing is what turns that same loop back into a two-cycle.
    with pytest.raises(RuntimeError, match="loops between two regimes"):
        _flip_run(table, solved, flip_par(3), periodic_threshold=3)


def test_periodic_solution_accepts_the_half_of_the_cycle_that_settles(
    table, solved, flip_par
):
    # Of the two guesses that repeat, the one weighed is the pass that wanted no
    # new binding date, which here is the one with date 0 already binding.
    ghx = solved[2]

    out, regimes, diag = _flip_run(
        table,
        solved,
        flip_par(1),
        n_periods=CYCLE_T,
        periodic_solution=True,
    )

    accepted = np.zeros(CYCLE_T, dtype=np.int8)
    accepted[0] = 1
    np.testing.assert_array_equal(regimes[0], accepted)
    # Flagged with the code it would otherwise have raised: 1 is Dynare's 310.
    np.testing.assert_array_equal(diag["periodic"], [1])
    np.testing.assert_array_equal(diag["iters"], [2])

    rule = occbin_recursion(*table, accepted, ghx)
    np.testing.assert_allclose(
        out,
        occbin_forward(rule, np.zeros(ghx.shape[1]), _cycle_eps(solved)),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize("strict", [True, False])
def test_periodic_solution_does_not_rescue_a_wider_loop(
    table, solved, flip_par, strict
):

    # Only a two-cycle is weighed by default. Dropping strictness weighs a loop
    # as well, but only one where some pass moved the guess no further than the
    # threshold, and none of this one's did.
    with pytest.raises(RuntimeError, match="loops over guess regimes"):
        _flip_run(
            table,
            solved,
            flip_par(3),
            periodic_solution=True,
            periodic_strict=strict,
        )


def test_a_reset_drops_a_grown_horizon_only_with_its_companion(table, solved, par, cf):
    # The first shock grows the horizon to hold its spell. The second clears the
    # spell outright, so the relaxed guess is the answer: a reset lands it in one
    # pass, and only the companion flag hands back the dates the guess grew.
    shocks = np.zeros((2, solved[3].shape[1]))
    shocks[0, 0] = SHOCK
    shocks[1, 0] = 0.005
    kw = {
        "check_ahead_periods": SHORT,
        "max_check_ahead_periods": HORIZON,
        "n_periods": 2,
    }

    plain_out, _, plain = _run(table, solved, par, cf, shocks, **kw)
    reset_out, reset_reg, reset = _run(
        table, solved, par, cf, shocks, reset_regime=True, **kw
    )
    both_out, _, both = _run(
        table,
        solved,
        par,
        cf,
        shocks,
        reset_regime=True,
        reset_check_ahead=True,
        **kw,
    )

    assert SHORT_T < plain["T_used"][0] < HORIZON_T
    # A relaxed tail is the reference rule's own fixed point, so the horizon the
    # period is solved on cannot reach the dates it reports.
    np.testing.assert_array_equal(reset_out, plain_out)
    # ``both`` solves the same mask on the shorter horizon, so its relaxed tail
    # is a few LU steps shorter and the two agree in exact arithmetic only.
    np.testing.assert_allclose(both_out, plain_out, rtol=1e-12, atol=1e-12)

    np.testing.assert_array_equal(reset["T_used"], plain["T_used"])
    np.testing.assert_array_equal(both["T_used"], [plain["T_used"][0], SHORT_T])
    assert plain["iters"][1] > reset["iters"][1] == both["iters"][1] == 1
    np.testing.assert_array_equal(reset_reg[1][SHORT_T:], 0)


def test_the_arena_formula_is_pinned():
    # The kernel carves nine regions out of two buffers; nothing else states
    # how big they are.
    n_var, n_state, n_ctrl, n_exog, T_cap, max_iter = 7, 3, 4, 2, 11, 5
    n_rhs = n_state + n_exog + 1
    recursion = n_var * n_var + 2 * n_var * n_rhs
    masks = (max_iter + 3) * T_cap + max_iter

    n_float, n_int = occbin_sim_arena_size(
        n_var, n_state, n_ctrl, n_exog, T_cap, max_iter
    )

    assert n_float == n_state + T_cap * n_var + max_iter + recursion
    assert n_int == n_var + 2 * max_iter + (masks + 7) // 8
