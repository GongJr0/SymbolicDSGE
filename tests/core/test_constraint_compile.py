# type: ignore
from __future__ import annotations

import copy
import ctypes
import dataclasses
import re

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.compiled_model import RegimeBlock
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
    cur[compiled_obc.idx["g"]], cur[compiled_obc.idx["z"]] = g, z
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


def _params(compiled):
    calib = compiled.config.calibration.parameters
    return np.array([float(calib[p]) for p in compiled.calib_params])


def _fold_to_cur(compiled):
    fwd = [sp.Symbol(f"fwd_{name}") for name in compiled.var_names]
    return dict(zip(fwd, compiled.cur_syms))


def _evaluate(compiled, exprs, n_row, point, par):
    """Flat row-major jacobian exprs as an (n_row, n_var) block."""
    args = (compiled.cur_syms, compiled.calib_params)
    vals = [float(sp.lambdify(args, e, "numpy")(point, par)) for e in exprs]
    return np.array(vals).reshape(n_row, len(compiled.var_names))


def _symbolic_pencil(compiled, residuals):
    """(a, b) blocks by the diff-then-fold the regime blocks are emitted with."""
    fold = _fold_to_cur(compiled)
    a = [sp.diff(e, s).subs(fold) for e in residuals for s in fold]
    b = [(-sp.diff(e, s)).subs(fold) for e in residuals for s in compiled.cur_syms]
    return a, b


def _assert_pencil_parity(compiled, ss, scale):
    # Regime blocks are row patches onto this pencil, so the procedure has to
    # reproduce the reference sweep on every row before a patch of it means
    # anything. Checked off the expansion point as well.
    residuals = compiled.objective_eqs
    par = _params(compiled)
    n_var = len(compiled.var_names)
    addr = compiled.construct_objective_cfunc().address
    a_sym, b_sym = _symbolic_pencil(compiled, residuals)

    assert len(a_sym) == len(residuals) * n_var
    assert len(b_sym) == len(residuals) * n_var

    rng = np.random.default_rng(0)
    for trial in range(5):
        point = ss if trial == 0 else ss + rng.normal(0, scale, n_var)
        a_ref, b_ref = klein_preprocess(addr, point, par, n_var, False)

        assert np.abs(a_ref).max() > 0.0
        assert np.abs(b_ref).max() > 0.0
        np.testing.assert_allclose(
            _evaluate(compiled, a_sym, len(residuals), point, par), a_ref, atol=1e-10
        )
        np.testing.assert_allclose(
            _evaluate(compiled, b_sym, len(residuals), point, par), b_ref, atol=1e-10
        )


def test_symbolic_pencil_matches_the_reference_sweep(compiled_post82):
    _assert_pencil_parity(
        compiled_post82, np.zeros(len(compiled_post82.var_names)), 0.3
    )


@pytest.fixture(scope="module")
def compiled_lead_regime(parsed_post82):
    """A replacement with leads, a lead-lead product and a nonlinear cur term.

    The constant replacements above leave ``jac_a`` identically zero, so they
    match a zero reference block no matter what the emission does. This one
    forces both blocks nonzero and leaves fwd symbols inside the fwd derivative,
    which is what the fold onto cur has to clean up.
    """
    model, _ = parsed_post82
    g, z, r = model.variables.variables[:3]
    beta = model.parameters[0]
    target = list(model.equations.model)[2]
    solver = _with_constraints(
        parsed_post82,
        {"lo": Constraint(bind=r(t) < 0, relax=r(t) >= 0)},
        {
            frozenset({"lo"}): {
                target: sp.Eq(
                    r(t), beta * g(t + 1) * z(t + 1) + sp.exp(g(t + 1)) - r(t) ** 2
                )
            }
        },
    )
    return solver.compile()


def test_regime_jacobians_match_the_complex_step_pencil(compiled_lead_regime):
    # The blocks replace a complex-step sweep of the regime, so they have to
    # reproduce it on the replaced rows. Checked away from the steady state too:
    # a gap model sits at ss = 0, where a wrong block can still read as right.
    compiled = compiled_lead_regime
    block = compiled.regimes[1]
    n_var = len(compiled.var_names)
    n_row = len(block.rows)
    par = _params(compiled)
    cfunc = compiled.construct_regime_cfuncs()[1]

    assert len(block.jac_a) == n_row * n_var
    assert len(block.jac_b) == n_row * n_var

    rng = np.random.default_rng(0)
    for trial in range(5):
        point = (
            np.zeros(n_var) if trial == 0 else np.round(rng.normal(0, 0.3, n_var), 3)
        )
        a_r, b_r = klein_preprocess(cfunc.address, point, par, n_var, False)

        assert np.abs(a_r[block.rows, :]).max() > 0.0
        assert np.abs(b_r[block.rows, :]).max() > 0.0
        np.testing.assert_allclose(
            _evaluate(compiled, block.jac_a, n_row, point, par), a_r[block.rows, :]
        )
        np.testing.assert_allclose(
            _evaluate(compiled, block.jac_b, n_row, point, par), b_r[block.rows, :]
        )


def test_regime_jacobians_fold_fwd_symbols_onto_cur(compiled_lead_regime):
    # The blocks are printed against a single `cur` input vector, so a surviving
    # fwd symbol would be unresolvable at emission.
    compiled = compiled_lead_regime
    block = compiled.regimes[1]
    fwd_syms = {sp.Symbol(f"fwd_{name}") for name in compiled.var_names}

    for expr in block.jac_a + block.jac_b:
        assert not expr.free_symbols & fwd_syms


def _call_jac(func, mask, cur, par):
    """The regime jacobian cfunc's output as a (2, n_row, n_var) `[a, b]` view."""
    fn = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    )(func.address(mask))
    cur = np.ascontiguousarray(cur, dtype=np.float64)
    par = np.ascontiguousarray(par, dtype=np.float64)
    out = np.zeros(func.n_out(mask), dtype=np.float64)
    fn(
        cur.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        par.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out.reshape(2, func.n_row(mask), func.n_var)


def test_regime_jacobian_cfunc_writes_both_pencil_blocks(compiled_lead_regime):
    # One call has to reproduce the regime's own complex-step sweep on the
    # replaced rows, a block first then b, so the halves patch a reference
    # pencil copy unchanged.
    compiled = compiled_lead_regime
    func = compiled.construct_regime_jacobian_func()
    par = _params(compiled)
    n_var = len(compiled.var_names)
    rows = func.rows[1]
    cfunc = compiled.construct_regime_cfuncs()[1]

    assert func.masks == (1,)
    assert func.n_out(1) == 2 * len(rows) * n_var

    rng = np.random.default_rng(0)
    for trial in range(5):
        point = (
            np.zeros(n_var) if trial == 0 else np.round(rng.normal(0, 0.3, n_var), 3)
        )
        a_r, b_r = klein_preprocess(cfunc.address, point, par, n_var, False)
        got = _call_jac(func, 1, point, par)

        assert np.abs(got).max() > 0.0
        np.testing.assert_allclose(got[0], a_r[rows, :])
        np.testing.assert_allclose(got[1], b_r[rows, :])


def test_regime_jacobian_func_carries_every_regime_and_is_cached(compiled_regimes):
    func = compiled_regimes.construct_regime_jacobian_func()

    assert func.masks == (1, 2, 3)
    assert (func.n_var, func.n_par) == (
        len(compiled_regimes.var_names),
        len(compiled_regimes.calib_params),
    )
    for mask, block in compiled_regimes.regimes.items():
        # The driver reads these as an i64 buffer next to the address.
        assert func.rows[mask].dtype == np.int64
        assert func.rows[mask].tolist() == block.rows
        assert func.n_row(mask) == len(block.rows)

    assert compiled_regimes.construct_regime_jacobian_func() is func


def test_regime_jacobian_rejects_a_block_that_does_not_match_its_rows(compiled_regimes):
    # Emission is where both the block and n_var are in scope; a short block
    # would otherwise reshape into a wrong pencil rather than fail.
    broken = dataclasses.replace(
        compiled_regimes, regimes={1: RegimeBlock(rows=[0], jac_a=[], jac_b=[])}
    )

    with pytest.raises(ValueError, match="expected"):
        broken.construct_regime_jacobian_func()


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

    assert sorted(compiled_regimes.regimes) == [1, 2, 3]
    assert cf.mask(frozenset({"lo"})) == 1
    assert cf.mask(frozenset({"hi"})) == 2
    assert cf.mask(frozenset({"lo", "hi"})) == 3


def test_regimes_replace_rows_in_reference_order(compiled_regimes):
    # Every regime pencil must stay row-aligned with the reference, so only the
    # replaced equation's row may differ and n_eq may not change.
    ref = compiled_regimes.objective_eqs

    for mask, block in compiled_regimes.regimes.items():
        assert len(block.residuals) == len(ref)
        differing = [
            i
            for i, (a, b) in enumerate(zip(ref, block.residuals))
            if sp.simplify(a - b) != 0
        ]
        assert differing == [2], f"mask {mask} changed rows {differing}"
        assert block.rows == differing


def test_regime_replacements_lower_to_residuals(compiled_regimes):
    cur_r, beta = sp.Symbol("cur_r"), sp.Symbol("beta")

    assert compiled_regimes.regimes[1].residuals[2] == cur_r
    assert sp.simplify(compiled_regimes.regimes[2].residuals[2] - (cur_r - beta)) == 0
    assert (
        sp.simplify(compiled_regimes.regimes[3].residuals[2] - (cur_r - 2 * beta)) == 0
    )


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

        # The taylor rule and its replacements are both contemporaneous, so the
        # swap lands entirely in b; what matters is that no other row moves.
        changed = np.maximum(
            np.abs(a_r - a_ref).max(axis=1), np.abs(b_r - b_ref).max(axis=1)
        )
        assert np.flatnonzero(changed).tolist() == [2]

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


def _rbc_seed(compiled):
    """Newton seed over the compiled layout, generated variables included.

    A lag aux starts where its origin does, so the suffix is stripped before the
    `<name>_ss` lookup; a shock state and an unseeded variable start at zero.
    """
    calib = compiled.config.calibration.parameters
    seeds = []
    for name in compiled.var_names:
        origin = re.sub(r"_lag\d+$", "", name)
        sym = sp.Symbol(f"{origin}_ss")
        seeds.append(float(calib[sym]) if sym in calib else 0.0)
    return np.array(seeds)


def _rbc_steady_state(compiled):
    ss, _ = steady_state_newton(
        compiled.construct_objective_cfunc().address,
        _rbc_seed(compiled),
        _params(compiled),
    )
    return ss


def test_symbolic_pencil_matches_the_reference_sweep_in_levels(compiled_rbc_obc):
    # POST82 sits at ss = 0, where a pencil can read as right for the wrong
    # reason. Here the expansion point is nonzero and the equations are
    # genuinely nonlinear, so the derivatives carry state.
    ss = _rbc_steady_state(compiled_rbc_obc)
    assert np.abs(ss).max() > 1.0
    _assert_pencil_parity(compiled_rbc_obc, ss, 0.05)


def test_levels_regime_constant_is_the_steady_state_residual(compiled_rbc_obc):
    calib = compiled_rbc_obc.config.calibration.parameters
    par = np.array([float(calib[p]) for p in compiled_rbc_obc.calib_params])
    n_eq = len(compiled_rbc_obc.var_names)

    ref_cfunc = compiled_rbc_obc.construct_objective_cfunc()
    ss_ref, _ = steady_state_newton(ref_cfunc.address, _rbc_seed(compiled_rbc_obc), par)

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


def test_regime_replacements_lift_shocks_like_the_reference(parsed_post82):
    # Regimes are desugared alongside the reference, so a shock in a replacement
    # reaches the residual through its shock state rather than as a bare symbol.
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

    cur_r, cur_shock = sp.Symbol("cur_r"), sp.Symbol(f"cur_{shock.name}_st")
    residual = compiled.regimes[1].residuals[2]

    assert sp.simplify(residual - (cur_r * (1 - rho_r) - cur_shock)) == 0
    assert shock not in residual.free_symbols
    assert compiled.construct_regime_cfuncs()[1] is not None


def test_regime_replacing_an_undeclared_equation_is_rejected(parsed_post82):
    # The merge would append rather than overwrite, shifting every later row
    # index off the reference.
    model, _ = parsed_post82
    r = model.variables.variables[2]
    solver = _with_constraints(
        parsed_post82,
        {"lo": Constraint(bind=r(t) < 0, relax=r(t) >= 0)},
        {frozenset({"lo"}): {"not_an_equation": sp.Eq(r(t), 0)}},
    )

    with pytest.raises(ValueError, match=r"does not declare: \['not_an_equation'\]"):
        solver.compile()


def test_model_without_regimes_has_no_regime_blocks(compiled_post82):
    assert compiled_post82.regimes == {}
    assert compiled_post82.construct_regime_cfuncs() == {}
    assert compiled_post82.construct_regime_jacobian_func() is None


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
