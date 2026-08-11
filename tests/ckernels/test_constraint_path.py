"""Native ``constraint_path``: the regime latch over a path of variable levels.

The fixture deliberately avoids complementary bind/relax pairs, because when
``relax`` is ``not(bind)`` the latch collapses to ``next = bind`` and every
hysteresis bug hides. ``band`` leaves a deadband on g where neither condition
holds, so its bit is decided only by the incoming regime; ``over`` overlaps on z
where both hold at once, so its bit flips on every pass. Declaration order puts
band on bit 0 and over on bit 1.
"""

from __future__ import annotations

import copy
import ctypes

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE._ckernels.occbin._occbin import MAX_CONSTRAINTS, constraint_path
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.config import Constraint

t = sp.Symbol("t", integer=True)

BAND, OVER = 0b01, 0b10

# g and z values placing each constraint in a chosen (bind, relax) cell.
BAND_ON, BAND_DEAD, BAND_OFF = -2.0, 0.0, 2.0
OVER_ON, OVER_BOTH, OVER_OFF = -2.0, 0.0, 2.0


@pytest.fixture(scope="module")
def compiled():
    model, kalman = ModelParser("MODELS/POST82.yaml").get_all()
    conf = copy.deepcopy(model)
    by_name = {v.__name__: v for v in conf.variables.variables}
    g, z = by_name["g"], by_name["z"]

    conf.equations.constraint = {
        "band": Constraint(bind=g(t) < -1, relax=g(t) > 1),
        "over": Constraint(bind=z(t) < 1, relax=z(t) > -1),
    }
    target = next(iter(conf.equations.model))
    conf.equations.regime = {
        combo: {target: sp.Eq(g(t), 0)}
        for combo in (
            frozenset({"band"}),
            frozenset({"over"}),
            frozenset({"band", "over"}),
        )
    }
    return DSGESolver(conf, kalman).compile()


@pytest.fixture(scope="module")
def par(compiled):
    return np.array(
        [
            float(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=np.float64,
    )


def _path(compiled, rows):
    """(T, n_var) level buffer with g and z set per row, every other variable 0."""
    out = np.zeros((len(rows), len(compiled.var_names)), dtype=np.float64)
    for i, (g, z) in enumerate(rows):
        out[i, compiled.idx["g"]] = g
        out[i, compiled.idx["z"]] = z
    return out


def _err(cf, cur, par):
    """The raw bind/relax distances, straight off the cfunc, no driver."""
    fn = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    )(cf.address)
    cur = np.ascontiguousarray(cur, dtype=np.float64)
    par = np.ascontiguousarray(par, dtype=np.float64)
    out = np.zeros(cf.n_cond, dtype=np.float64)
    fn(
        cur.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        par.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out


def _holds(cf, slot, err):
    """A distance decides its condition by sign; zero decides by strictness."""
    return err > 0.0 or (err == 0.0 and bool((cf.inclusive >> slot) & 1))


def _flags(cf, cur, par):
    """The 0/1 conditions the latch reads off those distances."""
    return [int(_holds(cf, k, e)) for k, e in enumerate(_err(cf, cur, par))]


def _oracle(cf, path, par, regime_in):
    """``next = prev ? !relax : bind``, per constraint, per period, in Python.

    A bit moves exactly when the one condition its incoming state asks about
    holds, so that condition's distance is the error the move is worth.
    """
    out = np.empty(len(path), dtype=np.int8)
    worst = 0.0
    for i, row in enumerate(path):
        err = _err(cf, row, par)
        prev, nxt = int(regime_in[i]), 0
        for c in range(cf.n_constraint):
            binding = (prev >> c) & 1
            slot = 2 * c + 1 if binding else 2 * c
            fired = _holds(cf, slot, err[slot])
            nxt |= int(not fired if binding else fired) << c
            if fired:
                worst = max(worst, abs(err[slot]))
        out[i] = nxt
    return out, worst


def test_fixture_conditions_are_not_complements(compiled, par):
    """Guards the premise: a deadband and an overlap both exist as written."""
    cf = compiled.construct_constraint_func()
    assert compiled.constraint_names == ("band", "over")

    dead = _flags(cf, _path(compiled, [(BAND_DEAD, OVER_OFF)])[0], par)
    both = _flags(cf, _path(compiled, [(BAND_OFF, OVER_BOTH)])[0], par)

    assert (dead[0], dead[1]) == (0, 0)
    assert (both[2], both[3]) == (1, 1)


def test_a_strict_condition_misses_its_own_boundary(compiled, par):
    # Every condition here is written strict, so a distance of exactly zero is
    # a miss. Nothing but `inclusive` separates that from a hit.
    cf = compiled.construct_constraint_func()
    assert cf.inclusive == 0
    row = _path(compiled, [(1.0, 1.0)])[0]

    err = _err(cf, row, par)

    # band relaxes on g > 1 and over binds on z < 1, both sitting on 1.0.
    assert (err[1], err[2]) == (0.0, 0.0)
    assert _flags(cf, row, par) == [0, 0, 0, 1]


def test_max_err_is_the_distance_of_the_condition_that_moved(compiled, par):
    # band relaxes on g > 1, so a period entering bound at g = 2.5 moves on a
    # distance of 1.5. over is consulted on its bind condition and misses, so
    # its own distance never enters.
    cf = compiled.construct_constraint_func()
    path = _path(compiled, [(2.5, OVER_OFF)])

    out, changed, max_err = constraint_path(
        cf.address,
        path,
        par,
        np.array([BAND], dtype=np.int8),
        cf.n_constraint,
        cf.inclusive,
    )

    assert out.tolist() == [0b00]
    assert changed == 1
    assert max_err == 1.5


@pytest.mark.parametrize(
    ("regime_in", "expected"),
    [
        (0b00, 0b10),  # band takes bind (0), over takes bind (1)
        (0b01, 0b11),  # band holds on !relax, over takes bind
        (0b10, 0b00),  # band takes bind, over clears on !relax
        (0b11, 0b01),  # band holds, over clears
    ],
)
def test_latch_resolves_each_incoming_bit(compiled, par, regime_in, expected):
    """Both constraints at their awkward cell at once, so a crossed bit shows."""
    cf = compiled.construct_constraint_func()
    path = _path(compiled, [(BAND_DEAD, OVER_BOTH)])

    out, changed, _ = constraint_path(
        cf.address,
        path,
        par,
        np.array([regime_in], dtype=np.int8),
        cf.n_constraint,
        cf.inclusive,
    )

    assert out.tolist() == [expected]
    assert changed == (expected != regime_in)


@pytest.mark.parametrize("regime_in", [0b00, BAND])
def test_deadband_holds_the_incoming_bit(compiled, par, regime_in):
    """Neither condition holds, so band can only come from the incoming regime."""
    cf = compiled.construct_constraint_func()
    path = _path(compiled, [(BAND_DEAD, OVER_OFF)])

    out, changed, _ = constraint_path(
        cf.address,
        path,
        par,
        np.array([regime_in], dtype=np.int8),
        cf.n_constraint,
        cf.inclusive,
    )

    assert out.tolist() == [regime_in]
    assert changed == 0


@pytest.mark.parametrize(("regime_in", "expected"), [(0b00, OVER), (OVER, 0b00)])
def test_overlap_toggles_the_incoming_bit(compiled, par, regime_in, expected):
    """Both conditions hold, so bind enters and !relax leaves on the same period."""
    cf = compiled.construct_constraint_func()
    path = _path(compiled, [(BAND_OFF, OVER_BOTH)])

    out, changed, _ = constraint_path(
        cf.address,
        path,
        par,
        np.array([regime_in], dtype=np.int8),
        cf.n_constraint,
        cf.inclusive,
    )

    assert out.tolist() == [expected]
    assert changed == 1


def test_fixed_point_reports_no_change(compiled, par):
    cf = compiled.construct_constraint_func()
    path = _path(compiled, [(BAND_ON, OVER_OFF)] * 6)
    regime_in = np.full(6, BAND, dtype=np.int8)

    out, changed, max_err = constraint_path(
        cf.address, path, par, regime_in, cf.n_constraint, cf.inclusive
    )

    assert out.tolist() == [BAND] * 6
    assert changed == 0
    # Nothing moved, so no condition contributed a distance.
    assert max_err == 0.0


def test_changed_counts_only_the_periods_that_moved(compiled, par):
    cf = compiled.construct_constraint_func()
    rows = [(BAND_ON, OVER_OFF)] * 5 + [(BAND_OFF, OVER_OFF)] * 2
    regime_in = np.full(len(rows), BAND, dtype=np.int8)

    out, changed, _ = constraint_path(
        cf.address,
        _path(compiled, rows),
        par,
        regime_in,
        cf.n_constraint,
        cf.inclusive,
    )

    assert out.tolist() == [BAND] * 5 + [0b00] * 2
    assert changed == 2


def test_matches_a_python_latch_over_random_paths(compiled, par):
    cf = compiled.construct_constraint_func()
    rng = np.random.default_rng(0)

    for _ in range(50):
        rows = list(zip(rng.uniform(-3, 3, 40), rng.uniform(-3, 3, 40)))
        path = _path(compiled, rows)
        regime_in = rng.integers(0, 4, 40).astype(np.int8)

        out, changed, max_err = constraint_path(
            cf.address, path, par, regime_in, cf.n_constraint, cf.inclusive
        )
        want, want_err = _oracle(cf, path, par, regime_in)

        assert out.tolist() == want.tolist()
        assert changed == int((want != regime_in).sum())
        assert max_err == want_err


def test_latches_in_place_when_out_aliases_regime_in(compiled, par):
    cf = compiled.construct_constraint_func()
    rng = np.random.default_rng(1)
    rows = list(zip(rng.uniform(-3, 3, 32), rng.uniform(-3, 3, 32)))
    path = _path(compiled, rows)
    regime_in = rng.integers(0, 4, 32).astype(np.int8)

    fresh, fresh_changed, fresh_err = constraint_path(
        cf.address, path, par, regime_in, cf.n_constraint, cf.inclusive
    )
    buf = regime_in.copy()
    same, same_changed, same_err = constraint_path(
        cf.address, path, par, buf, cf.n_constraint, cf.inclusive, out=buf
    )

    assert same is buf
    assert same.tolist() == fresh.tolist()
    assert same_changed == fresh_changed
    assert same_err == fresh_err


def test_empty_path_is_a_no_op(compiled, par):
    cf = compiled.construct_constraint_func()
    path = np.zeros((0, len(compiled.var_names)), dtype=np.float64)

    out, changed, max_err = constraint_path(
        cf.address, path, par, np.zeros(0, dtype=np.int8), cf.n_constraint, cf.inclusive
    )

    assert out.shape == (0,)
    assert changed == 0
    assert max_err == 0.0


@pytest.mark.parametrize("n_constraint", [0, -1, MAX_CONSTRAINTS + 1])
def test_rejects_a_constraint_count_the_distance_buffer_cannot_hold(
    compiled, par, n_constraint
):
    cf = compiled.construct_constraint_func()
    path = _path(compiled, [(0.0, 0.0)])

    with pytest.raises(ValueError, match="n_constraint"):
        constraint_path(
            cf.address,
            path,
            par,
            np.zeros(1, dtype=np.int8),
            n_constraint,
            cf.inclusive,
        )


def test_rejects_a_regime_length_that_does_not_match_the_path(compiled, par):
    cf = compiled.construct_constraint_func()
    path = _path(compiled, [(0.0, 0.0)] * 3)

    with pytest.raises(ValueError, match="regime_in has length"):
        constraint_path(
            cf.address,
            path,
            par,
            np.zeros(2, dtype=np.int8),
            cf.n_constraint,
            cf.inclusive,
        )
