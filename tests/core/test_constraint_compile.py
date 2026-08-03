# type: ignore
from __future__ import annotations

import copy
import ctypes

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.config import Constraint
from SymbolicDSGE._ckernels.core import (
    klein_preprocess,
    residual_eval,
    steady_state_newton,
)

t = sp.Symbol("t", integer=True)


def _with_constraints(parsed, constraint, replacements=None):
    """Parsed POST82 carrying `constraint` plus the regimes it requires.

    `replacements` maps a binding set to the equations that regime replaces;
    unlisted regimes fall back to pinning the first variable at zero.
    """
    model, kalman = parsed
    conf = copy.deepcopy(model)
    conf.equations.constraint = constraint

    target = next(iter(conf.equations.model))
    var = conf.variables.variables[0]
    names = list(constraint)
    combos = [frozenset({n}) for n in names]
    if len(names) == 2:
        combos.append(frozenset(names))
    replacements = replacements or {}
    conf.equations.regime = {
        c: replacements.get(c, {target: sp.Eq(var(t), 0)}) for c in combos
    }
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


@pytest.fixture(scope="module")
def compiled_regimes(parsed_post82):
    """Two constraints whose regimes each replace the third model equation."""
    model, _ = parsed_post82
    g, _, r = model.variables.variables[:3]
    beta = model.parameters[0]
    target = list(model.equations.model)[2]
    solver = _with_constraints(
        parsed_post82,
        {
            "lo": Constraint(bind=r(t) < 0, relax=r(t) >= 0),
            "hi": Constraint(bind=g(t) > beta, relax=g(t) <= beta),
        },
        {
            frozenset({"lo"}): {target: sp.Eq(r(t), 0)},
            frozenset({"hi"}): {target: sp.Eq(r(t), beta)},
            frozenset({"lo", "hi"}): {target: sp.Eq(r(t), 2 * beta)},
        },
    )
    return solver.compile()


def test_regimes_are_keyed_by_binding_bitmask(compiled_regimes):
    cf = compiled_regimes.construct_constraint_func()

    assert sorted(compiled_regimes.regime_eqs) == [1, 2, 3]
    assert cf.mask(frozenset({"lo"})) == 1
    assert cf.mask(frozenset({"hi"})) == 2
    assert cf.mask(frozenset({"lo", "hi"})) == 3


def test_regimes_replace_rows_in_reference_order(compiled_regimes):
    # Every regime pencil must stay row-aligned with the reference, so only the
    # replaced equation's row may differ and n_eq may not change.
    ref = compiled_regimes.objective_eqs

    for mask, eqs in compiled_regimes.regime_eqs.items():
        assert len(eqs) == len(ref)
        differing = [
            i for i, (a, b) in enumerate(zip(ref, eqs)) if sp.simplify(a - b) != 0
        ]
        assert differing == [2], f"mask {mask} changed rows {differing}"


def test_regime_replacements_lower_to_residuals(compiled_regimes):
    cur_r, beta = sp.Symbol("cur_r"), sp.Symbol("beta")

    assert compiled_regimes.regime_eqs[1][2] == cur_r
    assert sp.simplify(compiled_regimes.regime_eqs[2][2] - (cur_r - beta)) == 0
    assert sp.simplify(compiled_regimes.regime_eqs[3][2] - (cur_r - 2 * beta)) == 0


def test_regime_cfuncs_cover_every_regime_and_are_cached(compiled_regimes):
    cfuncs = compiled_regimes.construct_regime_cfuncs()

    assert sorted(cfuncs) == [1, 2, 3]
    assert compiled_regimes.construct_regime_cfuncs() is cfuncs


def test_regime_pencils_swap_only_the_replaced_row(compiled_regimes):
    # The constant is the whole OccBin mechanism: regimes linearize at the
    # reference steady state, where they do not hold, and c_r is that failure.
    par = np.array(
        [
            float(compiled_regimes.config.calibration.parameters[p])
            for p in compiled_regimes.calib_params
        ]
    )
    n_eq = len(compiled_regimes.var_names)
    ref_cfunc = compiled_regimes.construct_objective_cfunc()
    ss_ref, _ = steady_state_newton(ref_cfunc.address, np.zeros(n_eq), par)

    a_ref, b_ref = klein_preprocess(ref_cfunc.address, ss_ref, par, n_eq, False)
    c_ref = residual_eval(ref_cfunc.address, ss_ref, ss_ref, par, n_eq).real
    np.testing.assert_allclose(c_ref, 0.0, atol=1e-12)

    beta = float(compiled_regimes.config.calibration.parameters[sp.Symbol("beta")])
    expected_c = {1: 0.0, 2: -beta, 3: -2 * beta}

    for mask, cfunc in compiled_regimes.construct_regime_cfuncs().items():
        a_r, b_r = klein_preprocess(cfunc.address, ss_ref, par, n_eq, False)
        c_r = residual_eval(cfunc.address, ss_ref, ss_ref, par, n_eq).real

        assert np.flatnonzero(np.abs(a_r - a_ref).max(axis=1)).tolist() == [2]
        assert np.flatnonzero(np.abs(b_r - b_ref).max(axis=1)).tolist() == [2]

        want = np.zeros(n_eq)
        want[2] = expected_c[mask]
        np.testing.assert_allclose(c_r, want, atol=1e-12)


@pytest.fixture(scope="module")
def compiled_rbc_obc(rbc_second_order_test_model_path):
    """Levels RBC where a bad-TFP regime shuts investment off.

    POST82 is a gap model, so its regime constants can be zero purely because
    the reference steady state is. This fixture is in levels, where the
    constant is what actually drives the piecewise solve.
    """
    model, kalman = ModelParser(rbc_second_order_test_model_path).get_all()
    conf = copy.deepcopy(model)
    c, k, z = conf.variables.variables
    delta = sp.Symbol("delta")

    conf.equations.constraint = {"low": Constraint(bind=z(t) < 0, relax=z(t) >= 0)}
    conf.equations.regime = {
        frozenset({"low"}): {"euler": sp.Eq(k(t + 1), (1 - delta) * k(t))}
    }
    return DSGESolver(conf, kalman).compile()


def test_levels_regime_constant_is_the_steady_state_residual(compiled_rbc_obc):
    calib = compiled_rbc_obc.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled_rbc_obc.calib_params])
    n_eq = len(compiled_rbc_obc.var_names)
    seed = np.array(
        [
            float(calib[sp.Symbol(f"{n}_ss")]) if n != "z" else 0.0
            for n in compiled_rbc_obc.var_names
        ]
    )

    ref_cfunc = compiled_rbc_obc.construct_objective_cfunc()
    ss_ref, _ = steady_state_newton(ref_cfunc.address, seed, par)

    # The whole point of a levels fixture: the expansion point is not the origin.
    assert np.abs(ss_ref).max() > 1.0
    c_ref = residual_eval(ref_cfunc.address, ss_ref, ss_ref, par, n_eq).real
    np.testing.assert_allclose(c_ref, 0.0, atol=1e-10)

    # `euler` is row 0; the replacement k(t+1) = (1 - delta) k(t) leaves a
    # residual of delta * k_ss at the reference steady state.
    k_idx = compiled_rbc_obc.var_names.index("k")
    expected = float(calib[sp.Symbol("delta")]) * ss_ref[k_idx]
    assert expected > 0.5

    cfunc = compiled_rbc_obc.construct_regime_cfuncs()[1]
    c_r = residual_eval(cfunc.address, ss_ref, ss_ref, par, n_eq).real

    want = np.zeros(n_eq)
    want[0] = expected
    np.testing.assert_allclose(c_r, want, atol=1e-10)


def test_regime_replacements_zero_shocks_like_the_reference(parsed_post82):
    # Regimes walk the reference residual path, so a shock in a replacement is
    # accepted and then zeroed the way it is in objective_eqs.
    model, _ = parsed_post82
    r = model.variables.variables[2]
    shock = next(iter(model.shock_map))
    rho_r = sp.Symbol("rho_r")
    solver = _with_constraints(
        parsed_post82,
        {"lo": Constraint(bind=r(t) < 0, relax=r(t) >= 0)},
        {frozenset({"lo"}): {"taylor": sp.Eq(r(t), shock + rho_r * r(t))}},
    )

    compiled = solver.compile()

    cur_r = sp.Symbol("cur_r")
    assert sp.simplify(compiled.regime_eqs[1][2] - cur_r * (1 - rho_r)) == 0
    assert shock not in compiled.regime_eqs[1][2].free_symbols
    assert compiled.construct_regime_cfuncs()[1] is not None


def test_model_without_regimes_has_no_regime_eqs(compiled_post82):
    assert compiled_post82.regime_eqs == {}
    assert compiled_post82.construct_regime_cfuncs() == {}


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
