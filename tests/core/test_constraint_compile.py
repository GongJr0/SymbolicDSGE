# type: ignore
from __future__ import annotations

import copy
import ctypes

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE.core import DSGESolver
from SymbolicDSGE.core.config import Constraint

t = sp.Symbol("t", integer=True)


def _with_constraints(parsed, constraint):
    """Parsed POST82 carrying `constraint` plus the regimes it requires."""
    model, kalman = parsed
    conf = copy.deepcopy(model)
    conf.equations.constraint = constraint

    target = next(iter(conf.equations.model))
    var = conf.variables.variables[0]
    names = list(constraint)
    combos = [frozenset({n}) for n in names]
    if len(names) == 2:
        combos.append(frozenset(names))
    conf.equations.regime = {c: {target: sp.Eq(var(t), 0)} for c in combos}
    return DSGESolver(conf, kalman)


@pytest.fixture(scope="module")
def compiled_obc(parsed_post82):
    model, _ = parsed_post82
    g, z = (v for v in model.variables.variables[:2])
    beta = model.parameters[0]
    solver = _with_constraints(
        parsed_post82,
        {
            "lo": Constraint(bind=g(t) < 0, relax=g(t) >= 0),
            "hi": Constraint(bind=sp.And(z(t) > beta, g(t) < 1), relax=z(t) <= beta),
        },
    )
    return solver.compile()


def test_conditions_substitute_into_cur_symbols(compiled_obc):
    beta = sp.Symbol("beta")
    cur_g, cur_z = sp.Symbol("cur_g"), sp.Symbol("cur_z")

    assert compiled_obc.constraint_names == ("lo", "hi")
    assert compiled_obc.constraint_exprs == [
        cur_g < 0,
        cur_g >= 0,
        sp.And(cur_z > beta, cur_g < 1),
        cur_z <= beta,
    ]


def test_constraint_func_layout(compiled_obc):
    cf = compiled_obc.construct_constraint_func()

    assert cf.names == ("lo", "hi")
    assert (cf.n_constraint, cf.n_flag) == (2, 4)
    assert (cf.n_var, cf.n_par) == (
        len(compiled_obc.var_names),
        len(compiled_obc.calib_params),
    )
    assert (cf.bind_slot("lo"), cf.relax_slot("lo")) == (0, 1)
    assert (cf.bind_slot("hi"), cf.relax_slot("hi")) == (2, 3)
    assert cf.mask(frozenset({"lo"})) == 0b01
    assert cf.mask(frozenset({"hi"})) == 0b10
    assert cf.mask(frozenset({"lo", "hi"})) == 0b11


def test_constraint_func_is_cached(compiled_obc):
    # The driver holds .address, so the cfunc must not be rebuilt per call.
    assert compiled_obc.construct_constraint_func() is (
        compiled_obc.construct_constraint_func()
    )


def _call(cf, cur, par):
    fn = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int8),
    )(cf.address)
    cur = np.ascontiguousarray(cur, dtype=np.float64)
    par = np.ascontiguousarray(par, dtype=np.float64)
    out = np.zeros(cf.n_flag, dtype=np.int8)
    fn(
        cur.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        par.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )
    return out


@pytest.mark.parametrize(
    ("g", "z", "beta", "expected"),
    [
        # g   z   beta   lo_bind lo_relax hi_bind hi_relax
        (-1.0, 0.0, 0.5, [1, 0, 0, 1]),
        (0.5, 1.0, 0.5, [0, 1, 1, 0]),
        (2.0, 1.0, 0.5, [0, 1, 0, 0]),  # z > beta but g >= 1 fails the And
        (0.0, 0.5, 0.5, [0, 1, 0, 1]),  # exact boundaries: >= and <= both hold
    ],
)
def test_flags_match_the_written_conditions(compiled_obc, g, z, beta, expected):
    cf = compiled_obc.construct_constraint_func()
    cur = np.zeros(cf.n_var)
    cur[0], cur[1] = g, z
    par = np.zeros(cf.n_par)
    par[0] = beta

    out = _call(cf, cur, par)

    assert out.dtype == np.int8
    assert out.tolist() == expected


def test_flags_match_a_lambdify_oracle(compiled_obc):
    cf = compiled_obc.construct_constraint_func()
    oracle = [
        sp.lambdify((compiled_obc.cur_syms, compiled_obc.calib_params), e, "numpy")
        for e in compiled_obc.constraint_exprs
    ]
    rng = np.random.default_rng(0)

    for trial in range(200):
        cur = np.round(rng.normal(0.0, 1.0, cf.n_var), 2)
        par = np.round(rng.normal(0.0, 1.0, cf.n_par), 2)
        if trial % 5 == 0:  # drive the comparisons onto exact boundaries
            cur[0] = 0.0
            par[0] = cur[1]

        want = [int(bool(f(cur, par))) for f in oracle]
        assert _call(cf, cur, par).tolist() == want


def test_model_without_constraints_has_no_constraint_func(compiled_post82):
    assert compiled_post82.constraint_names == ()
    assert compiled_post82.constraint_exprs == []
    assert compiled_post82.construct_constraint_func() is None


@pytest.mark.parametrize("offset", [1, -1])
def test_conditions_reject_leads_and_lags(parsed_post82, offset):
    model, _ = parsed_post82
    g = model.variables.variables[0]
    solver = _with_constraints(
        parsed_post82,
        {"lo": Constraint(bind=g(t + offset) < 0, relax=g(t) >= 0)},
    )

    with pytest.raises(ValueError, match="only reference contemporaneous variables"):
        solver.compile()
