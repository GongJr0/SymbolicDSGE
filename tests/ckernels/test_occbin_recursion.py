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

from SymbolicDSGE._ckernels.occbin._occbin import (
    occbin_recursion,
    occbin_recursion_arena_size,
)
from SymbolicDSGE.core import DSGESolver
from SymbolicDSGE.core.config import Constraint

t = sp.Symbol("t", integer=True)

LOW = 0b1

# A binding run, a relapse, and a relaxed tail; the all-binding case is separate.
MIXED = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8)


@pytest.fixture(scope="module")
def solver(parsed_rbc_so):
    """Levels RBC where a bad-TFP regime shuts investment off."""
    model, kalman = parsed_rbc_so
    conf = copy.deepcopy(model)
    _, k, z = conf.variables.variables
    delta = sp.Symbol("delta")

    conf.equations.constraints = {"low": Constraint(bind=z(t) < 0, relax=z(t) >= 0)}
    conf.equations.regimes = {
        frozenset({"low"}): {"euler": sp.Eq(k(t + 1), (1 - delta) * k(t))}
    }
    return DSGESolver(conf, kalman)


@pytest.fixture(scope="module")
def compiled(solver):
    return solver.compile()


@pytest.fixture(scope="module")
def policy(solver, compiled):
    """The piecewise solve at the calibration.

    A constrained model solves to a ``PiecewiseSolution``, which is where the
    regime pencils and the reference rule come from: the kernel under test
    consumes exactly what the production path hands it.
    """
    pol = solver.solve(compiled).policy
    # The whole point of a levels fixture: the expansion point is not the origin.
    assert np.abs(pol.steady_state).max() > 1.0
    return pol


@pytest.fixture(scope="module")
def table(policy):
    """(a, b, c, d, cst) by bitmask: slot 0 the reference, slot 1 the regime."""
    return policy.a, policy.b, policy.c, policy.d, policy.cst


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


def test_an_all_relaxed_guess_is_the_reference_rule(table, policy):
    # ghx_ref seeds the recursion, so the reference rule is its fixed point and
    # every date reproduces it: a partition or sign error moves this at once.
    # `B` is the same regime's impact matrix out of the solve, which pins the
    # shock columns against something the recursion did not compute.
    ghx, B = policy.ghx_ref, policy.B
    n_state = ghx.shape[1]

    out = occbin_recursion(*table, np.zeros(6, dtype=np.int8), ghx)

    for date in range(out.shape[0]):
        np.testing.assert_allclose(out[date][:, :n_state], ghx, atol=1e-8)
        np.testing.assert_allclose(out[date][:, n_state:-1], B, atol=1e-8)
    # Exact: a zero constant column never leaves zero through the LU solve.
    np.testing.assert_array_equal(out[:, :, -1], np.zeros(out.shape[:2]))


def test_every_block_solves_the_pencil_of_its_own_date(table, policy):
    ghx = policy.ghx_ref

    out = occbin_recursion(*table, MIXED, ghx)

    assert _worst_residual(table, ghx, MIXED, out) < 1e-9


def test_a_binding_terminal_date_still_closes_on_the_reference_rule(table, policy):
    # Nothing in the kernel forces the guess to relax before the horizon ends,
    # and the seed is what the last date binds against.
    ghx = policy.ghx_ref
    mask = np.ones(4, dtype=np.int8)

    out = occbin_recursion(*table, mask, ghx)

    assert _worst_residual(table, ghx, mask, out) < 1e-9


def test_a_binding_date_moves_the_rule_off_the_reference(table, policy):
    # Guards the residual check above from passing on a kernel that ignores the
    # mask: the two rules would then agree everywhere.
    ghx = policy.ghx_ref

    relaxed = occbin_recursion(*table, np.zeros_like(MIXED), ghx)
    mixed = occbin_recursion(*table, MIXED, ghx)

    binding = np.flatnonzero(MIXED)
    assert np.abs(mixed[binding] - relaxed[binding]).max() > 1e-6
    # The affine part is live only where a regime's constant reaches.
    assert np.abs(mixed[:, :, -1]).max() > 1e-6


def test_out_is_written_in_place(table, policy):
    ghx = policy.ghx_ref
    a, _, _, d, _ = table
    n_var, n_rhs = a.shape[1], ghx.shape[1] + d.shape[2] + 1
    out = np.zeros((MIXED.size, n_var, n_rhs))

    returned = occbin_recursion(*table, MIXED, ghx, out)

    assert returned is out
    np.testing.assert_array_equal(out, occbin_recursion(*table, MIXED, ghx))


def test_an_empty_guess_returns_an_empty_stack(table, policy):
    ghx = policy.ghx_ref
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


def test_the_recursion_stays_inside_its_arena(table, policy, compiled):
    # Sized exactly and followed by sentinels: an overrun lands in the guard
    # instead of in whatever the live path put after the arena.
    ghx = policy.ghx_ref
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


def test_a_short_arena_is_rejected(table, policy, compiled):
    ghx = policy.ghx_ref
    n_float, _ = occbin_recursion_arena_size(
        compiled.n_var, compiled.n_state, compiled.n_exog
    )

    with pytest.raises(ValueError, match="needs"):
        occbin_recursion(*table, MIXED, ghx, None, np.empty(n_float - 1))


def test_a_singular_date_is_named(table, policy):
    # The LU is the only failure the recursion can report, and the date is the
    # single piece of information the caller cannot recover on its own.
    ghx = policy.ghx_ref
    dead = tuple(np.concatenate([blk, np.zeros_like(blk[:1])]) for blk in table)
    mask = np.array([0, 0, 2, 0], dtype=np.int8)

    with pytest.raises(RuntimeError, match=r"singular pencil at date 2\."):
        occbin_recursion(*dead, mask, ghx)


def test_a_mask_outside_the_table_is_rejected(table, policy):
    # The kernel indexes table[mask[t]] unguarded, so an out-of-range bit reads
    # past the regime array instead of raising.
    ghx = policy.ghx_ref
    mask = np.array([0, 1, 2, 0], dtype=np.int8)

    with pytest.raises(ValueError, match=re.escape("mask[2] is 2, outside 0..1")):
        occbin_recursion(*table, mask, ghx)


def test_a_pencil_that_does_not_partition_is_rejected(table, policy):
    # n_state comes from ghx_ref alone; if its rows miss n_var the kernel reads
    # the seed off the end of the array it was handed.
    ghx = policy.ghx_ref

    with pytest.raises(ValueError, match="expected"):
        occbin_recursion(*table, np.zeros(3, dtype=np.int8), ghx[:-1])
