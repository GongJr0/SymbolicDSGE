# type: ignore
"""Native ``occbin_recursion``: backward decision rules for a regime guess.

The rules are checked against the pencils they were built from rather than
against a stored path. Each block is the affine map from ``[x_{t-1}; eps_t; 1]``
to the whole of ``y_t``, so substituting it back into
``a E_t[y_{t+1}] = b y_t + c y_{t-1} + d eps_t - cst`` is an identity in those
free variables, which a residual on the ``(n_var, n_state + n_exog + 1)`` block
tests directly and without an oracle. The date-``T`` closure is the same
substitution with ``ghx_ref``, no innovation and a zero constant.

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
    """(ss, reference pencil, ghx, B) for the reference regime."""
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
    # The whole point of a levels fixture: the expansion point is not the origin.
    assert np.abs(ss).max() > 1.0
    return ss, ref, np.vstack([p, f]), B


@pytest.fixture(scope="module")
def table(compiled, par, solved):
    """(a, b, c, d, cst) by bitmask: slot 0 the reference, slot 1 the regime."""
    ss, ref, _, _ = solved
    n_state = compiled.n_state
    func = compiled.construct_regime_pencil_func()

    slots = [
        regime_pencil(0, np.empty(0, dtype=np.int64), ss, par, *ref),
        regime_pencil(func.address(LOW), func.rows[LOW], ss, par, *ref),
    ]
    # The regime is only a test of the constant path if it carries one, and the
    # reference is the linearization the steady state solves, so it carries none.
    assert np.abs(slots[LOW][4]).max() > 0.5
    np.testing.assert_array_equal(slots[0][4], np.zeros(compiled.n_var))
    # The recursion reads only the state columns of the lag block, on the
    # grounds that a control does not occur at t-1.
    for blocks in slots:
        np.testing.assert_array_equal(
            blocks[2][:, n_state:], np.zeros((compiled.n_var, compiled.n_ctrl))
        )

    return tuple(np.stack([lo[i] for lo in slots]) for i in range(5))


def _worst_residual(table, ghx, mask, out):
    """max |a y_{t+1} - b y_t - c y_{t-1} - d eps_t + cst|, relatively.

    Every ``y`` is affine in ``[x_{t-1}; eps_t; 1]``, so each quantity here is an
    ``(n_var, n_state + n_exog + 1)`` matrix and the residual is an identity
    rather than a sample.
    """
    a, b, c, d, cst = table
    T, n_var, n_rhs = out.shape
    n_exog = d.shape[2]
    n_state = n_rhs - n_exog - 1

    # y_{t-1} and eps_t as matrices in the free variables the rules use. The
    # control rows of y_{t-1} stay zero: `c` has no column that reads them.
    prev = np.zeros((n_var, n_rhs))
    prev[:n_state, :n_state] = np.eye(n_state)
    eps = np.zeros((n_exog, n_rhs))
    eps[:, n_state : n_state + n_exog] = np.eye(n_exog)

    worst = 0.0
    for date in range(T):
        cur = out[date]
        if date + 1 < T:
            nxt_rule = out[date + 1]
        else:
            nxt_rule = np.zeros((n_var, n_rhs))
            nxt_rule[:, :n_state] = ghx

        # E_t[y_{t+1}]: the next date's rule on this date's state, its own
        # innovation zero in expectation.
        nxt = nxt_rule[:, :n_state] @ cur[:n_state]
        nxt[:, -1] += nxt_rule[:, -1]

        m = mask[date]
        lead = a[m] @ nxt
        lag = b[m] @ cur + c[m] @ prev + d[m] @ eps
        lag[:, -1] -= cst[m]

        scale = max(np.abs(lead).max(), np.abs(lag).max(), 1.0)
        worst = max(worst, np.abs(lead - lag).max() / scale)
    return worst


def test_an_all_relaxed_guess_is_the_reference_rule(table, solved):
    # ghx_ref seeds the recursion, so the reference rule is its fixed point and
    # every date reproduces it: a partition or sign error moves this at once.
    # `B` is the same regime's impact matrix out of the solve, which pins the
    # shock columns against something the recursion did not compute.
    _, _, ghx, B = solved
    n_state = ghx.shape[1]

    out = occbin_recursion(*table, np.zeros(6, dtype=np.int8), ghx)

    for date in range(out.shape[0]):
        np.testing.assert_allclose(out[date][:, :n_state], ghx, atol=1e-8)
        np.testing.assert_allclose(out[date][:, n_state:-1], B, atol=1e-8)
    # Exact: a zero constant column never leaves zero through the LU solve.
    np.testing.assert_array_equal(out[:, :, -1], np.zeros(out.shape[:2]))


def test_every_block_solves_the_pencil_of_its_own_date(table, solved):
    _, _, ghx, _ = solved

    out = occbin_recursion(*table, MIXED, ghx)

    assert _worst_residual(table, ghx, MIXED, out) < 1e-9


def test_a_binding_terminal_date_still_closes_on_the_reference_rule(table, solved):
    # Nothing in the kernel forces the guess to relax before the horizon ends,
    # and the seed is what the last date binds against.
    _, _, ghx, _ = solved
    mask = np.ones(4, dtype=np.int8)

    out = occbin_recursion(*table, mask, ghx)

    assert _worst_residual(table, ghx, mask, out) < 1e-9


def test_a_binding_date_moves_the_rule_off_the_reference(table, solved):
    # Guards the residual check above from passing on a kernel that ignores the
    # mask: the two rules would then agree everywhere.
    _, _, ghx, _ = solved

    relaxed = occbin_recursion(*table, np.zeros_like(MIXED), ghx)
    mixed = occbin_recursion(*table, MIXED, ghx)

    binding = np.flatnonzero(MIXED)
    assert np.abs(mixed[binding] - relaxed[binding]).max() > 1e-6
    # The affine part is live only where a regime's constant reaches.
    assert np.abs(mixed[:, :, -1]).max() > 1e-6


def test_out_is_written_in_place(table, solved):
    _, _, ghx, _ = solved
    a, _, _, d, _ = table
    n_var, n_rhs = a.shape[1], ghx.shape[1] + d.shape[2] + 1
    out = np.zeros((MIXED.size, n_var, n_rhs))

    returned = occbin_recursion(*table, MIXED, ghx, out)

    assert returned is out
    np.testing.assert_array_equal(out, occbin_recursion(*table, MIXED, ghx))


def test_an_empty_guess_returns_an_empty_stack(table, solved):
    _, _, ghx, _ = solved
    a, _, _, d, _ = table

    out = occbin_recursion(*table, np.empty(0, dtype=np.int8), ghx)

    assert out.shape == (0, a.shape[1], ghx.shape[1] + d.shape[2] + 1)


def test_the_arena_holds_the_pencil_the_rhs_and_the_seed(table, compiled):
    # Three blocks and the pivots. The sizer is the only statement of that
    # layout the kernel does not make itself, so it is pinned here.
    n_var, n_state, n_exog = compiled.n_var, compiled.n_state, compiled.n_exog
    n_rhs = n_state + n_exog + 1

    n_float, n_int = occbin_recursion_arena_size(n_var, n_state, n_exog)

    assert n_float == n_var * n_var + 2 * n_var * n_rhs
    assert n_int == n_var


def test_the_recursion_stays_inside_its_arena(table, solved, compiled):
    # Sized exactly and followed by sentinels: an overrun lands in the guard
    # instead of in whatever the live path put after the arena.
    _, _, ghx, _ = solved
    n_float, n_int = occbin_recursion_arena_size(
        compiled.n_var, compiled.n_state, compiled.n_exog
    )
    guard, fill, ifill = 8, -1.5e300, -(2**60)

    arena = np.full(n_float + guard, fill)
    iarena = np.full(n_int + guard, ifill, dtype=np.int64)
    out = occbin_recursion(*table, MIXED, ghx, None, arena, iarena)

    np.testing.assert_array_equal(arena[n_float:], np.full(guard, fill))
    np.testing.assert_array_equal(iarena[n_int:], np.full(guard, ifill))
    # Vacuous unless the buffers passed in are the ones the kernel scratched on.
    assert np.any(arena[:n_float] != fill)
    np.testing.assert_array_equal(out, occbin_recursion(*table, MIXED, ghx))


def test_a_short_arena_is_rejected(table, solved, compiled):
    _, _, ghx, _ = solved
    n_float, _ = occbin_recursion_arena_size(
        compiled.n_var, compiled.n_state, compiled.n_exog
    )

    with pytest.raises(ValueError, match="needs"):
        occbin_recursion(*table, MIXED, ghx, None, np.empty(n_float - 1))


def test_a_singular_date_is_named(table, solved):
    # The LU is the only failure the recursion can report, and the date is the
    # single piece of information the caller cannot recover on its own.
    _, _, ghx, _ = solved
    dead = tuple(np.concatenate([blk, np.zeros_like(blk[:1])]) for blk in table)
    mask = np.array([0, 0, 2, 0], dtype=np.int8)

    with pytest.raises(RuntimeError, match=r"singular pencil at date 2\."):
        occbin_recursion(*dead, mask, ghx)


def test_a_mask_outside_the_table_is_rejected(table, solved):
    # The kernel indexes table[mask[t]] unguarded, so an out-of-range bit reads
    # past the regime array instead of raising.
    _, _, ghx, _ = solved
    mask = np.array([0, 1, 2, 0], dtype=np.int8)

    with pytest.raises(ValueError, match=re.escape("mask[2] is 2, outside 0..1")):
        occbin_recursion(*table, mask, ghx)


def test_a_pencil_that_does_not_partition_is_rejected(table, solved):
    # n_state comes from ghx_ref alone; if its rows miss n_var the kernel reads
    # the seed off the end of the array it was handed.
    _, _, ghx, _ = solved

    with pytest.raises(ValueError, match="expected"):
        occbin_recursion(*table, np.zeros(3, dtype=np.int8), ghx[:-1])
