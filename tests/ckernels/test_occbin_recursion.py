# type: ignore
"""Native ``occbin_recursion``: backward decision rules for a regime guess.

The rules are checked against the pencils they were built from rather than
against a stored path. Each block is the affine map from ``x_t`` to ``[x_{t+1};
u_t]``, so substituting it back into ``a y_{t+1} = b y_t - c`` is an identity in
``x_t``, which a residual on the ``(n_var, n_state + 1)`` block tests directly
and without an oracle. The date-``T`` closure is the same substitution with
``f_ref`` and a zero constant.

The fixture is the levels RBC of ``test_regime_pencil``, whose binding regime
carries a nonzero constant. A gap model would pass the constant column for free.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.core import klein_preprocess, klein_solve1
from SymbolicDSGE._ckernels.occbin._occbin import (
    occbin_recursion,
    occbin_recursion_arena_size,
    regime_pencil,
)
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.config import Constraint

t = sp.Symbol("t", integer=True)

LOW = 0b1

# A binding run, a relapse, and a relaxed tail; the all-binding case is separate.
MIXED = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)


@pytest.fixture(scope="module")
def compiled(rbc_second_order_test_model_path):
    """Levels RBC where a bad-TFP regime shuts investment off."""
    model, kalman = ModelParser(rbc_second_order_test_model_path).get_all()
    conf = copy.deepcopy(model)
    _, k, z = conf.variables.variables
    delta = sp.Symbol("delta")

    conf.equations.constraint = {"low": Constraint(bind=z(t) < 0, relax=z(t) >= 0)}
    conf.equations.regime = {
        frozenset({"low"}): {"euler": sp.Eq(k(t + 1), (1 - delta) * k(t))}
    }
    return DSGESolver(conf, kalman).compile()


@pytest.fixture(scope="module")
def par(compiled):
    calib = compiled.config.calibration.parameters
    return np.array([float(calib[p]) for p in compiled.calib_params])


@pytest.fixture(scope="module")
def solved(compiled, par):
    """(ss, a_ref, b_ref, f_ref, p_ref) for the reference regime."""
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
    # The whole point of a levels fixture: the expansion point is not the origin.
    assert np.abs(ss).max() > 1.0
    return ss, a_ref, b_ref, f_ref, p_ref


@pytest.fixture(scope="module")
def table(compiled, par, solved):
    """(a, b, c) stacked by bitmask: slot 0 the reference, slot 1 the regime."""
    ss, a_ref, b_ref, _, _ = solved
    func = compiled.construct_regime_pencil_func()
    a_low, b_low, c_low = regime_pencil(
        func.address(LOW), func.rows[LOW], ss, par, a_ref, b_ref
    )
    # The regime is only a test of the constant path if it carries one.
    assert np.abs(c_low).max() > 0.5

    a = np.stack([a_ref, a_low])
    b = np.stack([b_ref, b_low])
    c = np.stack([np.zeros_like(c_low), c_low])
    return a, b, c


def _worst_residual(table, f_ref, mask, out):
    """max |a y_{t+1} - b y_t + c| over dates, relative to the terms it cancels.

    ``y`` is affine in ``[x_t; 1]``, so every quantity here is an ``(n_var,
    n_state + 1)`` matrix and the residual is an identity, not a sample.
    """
    a, b, c = table
    T, n_var, n_rhs = out.shape
    n_state = n_rhs - 1
    n_ctrl = n_var - n_state

    cur_state = np.zeros((n_state, n_rhs))
    cur_state[:, :n_state] = np.eye(n_state)

    worst = 0.0
    for date in range(T):
        state, ctrl = out[date][:n_state], out[date][n_state:]
        if date + 1 < T:
            rule_next = out[date + 1][n_state:]
        else:
            rule_next = np.zeros((n_ctrl, n_rhs))
            rule_next[:, :n_state] = f_ref

        ctrl_next = rule_next[:, :n_state] @ state
        ctrl_next[:, n_state] += rule_next[:, n_state]

        m = mask[date]
        lead = a[m] @ np.vstack([state, ctrl_next])
        lag = b[m] @ np.vstack([cur_state, ctrl])
        resid = lead - lag
        resid[:, n_state] += c[m]

        scale = max(np.abs(lead).max(), np.abs(lag).max(), 1.0)
        worst = max(worst, np.abs(resid).max() / scale)
    return worst


def test_an_all_relaxed_guess_is_the_reference_rule(table, solved):
    # f_ref seeds the recursion, so the reference rule is its fixed point and
    # every date reproduces it: a partition or sign error moves this at once.
    _, _, _, f_ref, p_ref = solved
    a, b, c = table
    n_state = f_ref.shape[1]

    out = occbin_recursion(a, b, c, np.zeros(6, dtype=np.int8), f_ref)

    for date in range(out.shape[0]):
        np.testing.assert_allclose(out[date][:n_state, :n_state], p_ref, atol=1e-8)
        np.testing.assert_allclose(out[date][n_state:, :n_state], f_ref, atol=1e-8)
    # Exact: a zero constant column never leaves zero through the LU solve.
    np.testing.assert_array_equal(out[:, :, n_state], np.zeros(out.shape[:2]))


def test_every_block_solves_the_pencil_of_its_own_date(table, solved):
    _, _, _, f_ref, _ = solved

    out = occbin_recursion(*table, MIXED, f_ref)

    assert _worst_residual(table, f_ref, MIXED, out) < 1e-9


def test_a_binding_terminal_date_still_closes_on_the_reference_rule(table, solved):
    # Nothing in the kernel forces the guess to relax before the horizon ends,
    # and the seed is what the last date binds against.
    _, _, _, f_ref, _ = solved
    mask = np.ones(4, dtype=np.int8)

    out = occbin_recursion(*table, mask, f_ref)

    assert _worst_residual(table, f_ref, mask, out) < 1e-9


def test_a_binding_date_moves_the_rule_off_the_reference(table, solved):
    # Guards the residual check above from passing on a kernel that ignores the
    # mask: the two rules would then agree everywhere.
    _, _, _, f_ref, _ = solved
    a, b, c = table

    relaxed = occbin_recursion(a, b, c, np.zeros_like(MIXED), f_ref)
    mixed = occbin_recursion(a, b, c, MIXED, f_ref)

    binding = np.flatnonzero(MIXED)
    assert np.abs(mixed[binding] - relaxed[binding]).max() > 1e-6
    # The affine part is live only where a regime's constant reaches.
    assert np.abs(mixed[:, :, -1]).max() > 1e-6


def test_out_is_written_in_place(table, solved):
    _, _, _, f_ref, _ = solved
    a, b, c = table
    n_var, n_rhs = a.shape[1], f_ref.shape[1] + 1
    out = np.zeros((MIXED.size, n_var, n_rhs))

    returned = occbin_recursion(a, b, c, MIXED, f_ref, out)

    assert returned is out
    np.testing.assert_array_equal(out, occbin_recursion(a, b, c, MIXED, f_ref))


def test_an_empty_guess_returns_an_empty_stack(table, solved):
    _, _, _, f_ref, _ = solved
    a, _, _ = table

    out = occbin_recursion(*table, np.empty(0, dtype=np.int8), f_ref)

    assert out.shape == (0, a.shape[1], f_ref.shape[1] + 1)


def test_the_arena_holds_the_pencil_the_rhs_and_the_seed(table, solved):
    # Three blocks and the pivots. The sizer is the only statement of that
    # layout the kernel does not make itself, so it is pinned here.
    _, _, _, f_ref, _ = solved
    n_ctrl, n_state = f_ref.shape
    n_var = table[0].shape[1]
    n_rhs = n_state + 1

    n_float, n_int = occbin_recursion_arena_size(n_var, n_state, n_ctrl)

    assert n_float == n_var * n_var + n_var * n_rhs + n_ctrl * n_rhs
    assert n_int == n_var


def test_the_recursion_stays_inside_its_arena(table, solved):
    # Sized exactly and followed by sentinels: an overrun lands in the guard
    # instead of in whatever the live path put after the arena.
    _, _, _, f_ref, _ = solved
    n_ctrl, n_state = f_ref.shape
    n_float, n_int = occbin_recursion_arena_size(table[0].shape[1], n_state, n_ctrl)
    guard, fill, ifill = 8, -1.5e300, -(2**60)

    arena = np.full(n_float + guard, fill)
    iarena = np.full(n_int + guard, ifill, dtype=np.int64)
    out = occbin_recursion(*table, MIXED, f_ref, None, arena, iarena)

    np.testing.assert_array_equal(arena[n_float:], np.full(guard, fill))
    np.testing.assert_array_equal(iarena[n_int:], np.full(guard, ifill))
    # Vacuous unless the buffers passed in are the ones the kernel scratched on.
    assert np.any(arena[:n_float] != fill)
    np.testing.assert_array_equal(out, occbin_recursion(*table, MIXED, f_ref))


def test_a_short_arena_is_rejected(table, solved):
    _, _, _, f_ref, _ = solved
    n_ctrl, n_state = f_ref.shape
    n_float, _ = occbin_recursion_arena_size(table[0].shape[1], n_state, n_ctrl)

    with pytest.raises(ValueError, match="needs"):
        occbin_recursion(*table, MIXED, f_ref, None, np.empty(n_float - 1))


def test_a_singular_date_is_named(table, solved):
    # The LU is the only failure the recursion can report, and the date is the
    # single piece of information the caller cannot recover on its own.
    _, _, _, f_ref, _ = solved
    a, b, c = table
    dead = (
        np.concatenate([a, np.zeros_like(a[:1])]),
        np.concatenate([b, np.zeros_like(b[:1])]),
        np.concatenate([c, np.zeros_like(c[:1])]),
    )
    mask = np.array([0, 0, 2, 0], dtype=np.int8)

    with pytest.raises(RuntimeError, match=r"singular pencil at date 2\."):
        occbin_recursion(*dead, mask, f_ref)


def test_a_mask_outside_the_table_is_rejected(table, solved):
    # The kernel indexes table[mask[t]] unguarded, so an out-of-range bit reads
    # past the regime array instead of raising.
    _, _, _, f_ref, _ = solved
    mask = np.array([0, 1, 2, 0], dtype=np.int8)

    with pytest.raises(ValueError, match=re.escape("mask[2] is 2, outside 0..1")):
        occbin_recursion(*table, mask, f_ref)


def test_a_pencil_that_does_not_partition_is_rejected(table, solved):
    # n_state and n_ctrl come from f_ref alone; if they miss n_var the kernel
    # splits every row at the wrong column.
    _, _, _, f_ref, _ = solved

    with pytest.raises(ValueError, match="expected"):
        occbin_recursion(*table, np.zeros(3, dtype=np.int8), f_ref[:, :-1])
