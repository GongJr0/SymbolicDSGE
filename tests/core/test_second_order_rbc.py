"""Dynare parity for the single-shock RBC at second order.

The decision rule (ghxx, ghxu, ghuu, ghs2), the ``.solve(order=2)`` public path,
and the pruned simulation and IRF paths. Goldens and the orderings that map them
onto our layout are in :mod:`_oracles.dynare_rbc_second_order`.

Three things this fixture cannot reach, and test_second_order_multishock.py
exists for: cross terms between distinct innovations, the Cholesky branch of the
eta builder, and a risk correction against a full covariance.
"""

from __future__ import annotations

import re

import numpy as np
import sympy as sp

from SymbolicDSGE._ckernels.core._core import bicomplex_hessian, klein_preprocess
from SymbolicDSGE._ckernels.core import second_order, second_order_risk
from SymbolicDSGE._symbolic_printers import ResidualLayout
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solver_backend import klein_solve
from _oracles import dynare_rbc_second_order as golden


def _dynare_pair(dyn_row: list[float]) -> np.ndarray:
    """A Dynare ghxx row [kk, kz, zk, zz] as a 2x2 in state order [k, z]."""
    return np.asarray(dyn_row, dtype=np.float64).reshape(2, 2)


def _our_pair(compiled, tensor_row: np.ndarray) -> np.ndarray:
    """The [k_lag1, z_lag1] sub-block of one row of a state tensor."""
    ix = [compiled.idx[name] for name in golden.STATES]
    return tensor_row[np.ix_(ix, ix)]


def _our_shock_block(compiled, tensor_row: np.ndarray) -> np.ndarray:
    """The e_st column of one tensor row as Dynare orders it: the two ghxu
    entries [d2/dk de, d2/dz de] followed by ghuu [d2/de2]."""
    e = compiled.idx["e_st"]
    k, z = (compiled.idx[n] for n in golden.STATES)
    return np.array([tensor_row[k, e], tensor_row[z, e], tensor_row[e, e]])


def _dynare_shock_block(dr_row: int) -> np.ndarray:
    """One decision-rule row's [ghxu_k, ghxu_z, ghuu], to line up with
    ``_our_shock_block``. The two ghxu columns are ``n_var`` apart."""
    n_var = len(golden.GHUU)
    return np.array(
        [
            golden.GHXU[dr_row],
            golden.GHXU[n_var + dr_row],
            golden.GHUU[dr_row],
        ]
    )


def _levels_steady_state(compiled) -> np.ndarray:
    """The RBC expansion point over the compiled layout, generated vars included."""
    calib = compiled.config.calibration.parameters
    known = {"k": float(calib[sp.Symbol("k_ss")]), "c": float(calib[sp.Symbol("c_ss")])}
    return np.array(
        [known.get(re.sub(r"_lag\d+$", "", n), 0.0) for n in compiled.var_names],
        dtype=np.float64,
    )


def _solved_rbc():
    """The RBC model solved to second order, as (solved, compiled)."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return solver.solve(compiled, order=2), compiled


def _solve_rbc_second_order():
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return solver.solve(compiled, order=2)


def _golden_columns(solved) -> list[int]:
    """Our columns matching the golden's stored [z, k, c].

    The paths were captured against Dynare's own dating, where the reported k is
    the predetermined stock. That is our k_lag1, not our contemporaneous k.
    """
    idx = solved.compiled.idx
    return [idx["z"], idx["k_lag1"], idx["c"]]


def _golden_x0(solved) -> np.ndarray:
    """The Dynare initial condition over our state block [e_st, k_lag1, z_lag1].

    The golden reports z contemporaneously but starts the path one period
    earlier, so the initial lag is one AR step behind the printed value.
    """
    rho = float(solved.compiled.config.calibration.parameters[sp.Symbol("rho")])
    z0, k0 = golden.SIM_X0
    return np.array([0.0, k0, z0 / rho], dtype=np.float64)


def test_rbc_second_order_matches_dynare():
    """The golden: our g_xx/h_xx match Dynare's ghxx on the state pair Dynare and
    we share, and the risk correction g_ss/h_ss matches ghs2. The
    independent-solver check on the actual second-order math."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    compiled = DSGESolver(model, kalman).compile()
    # States are all compiler-minted: the lifted shock and the two lag auxes.
    assert list(compiled.var_names) == ["e_st", "k_lag1", "z_lag1", "c", "k", "z"]

    layout = ResidualLayout.from_compiled(compiled)
    n_eq, n_state = layout.n_eq, compiled.n_state
    calib = compiled.config.calibration.parameters
    ss = _levels_steady_state(compiled)
    par = np.array([float(calib[p]) for p in compiled.calib_params], dtype=np.float64)

    cf = compiled.construct_objective_cfunc()
    cf_bc = compiled.construct_objective_cfunc_bicomplex()
    eq = compiled.equations

    # Steady state actually clears the residual.
    resid = eq(
        ss.astype(np.complex128), ss.astype(np.complex128), par.astype(np.complex128)
    )
    np.testing.assert_allclose(np.real(resid), 0.0, atol=1e-7)

    a, b = klein_preprocess(cf.address, ss, par, n_eq)
    sol = klein_solve(cf, par, ss, n_state)
    assert sol.stab == 0
    gx, hx = np.real(sol.f), np.real(sol.p)
    f_xx = bicomplex_hessian(cf_bc.address, ss, par, n_eq)
    gxx, hxx = second_order(a, b, f_xx, gx, hx, n_state)

    # k_lag1(t+1) = k(t), so Dynare's k' row is ours; z_lag1(t+1) = z(t) is linear.
    ctrl = {n: i for i, n in enumerate(compiled.var_names[n_state:])}
    np.testing.assert_allclose(
        _our_pair(compiled, gxx[ctrl["c"]]),
        _dynare_pair(golden.GHXX_C),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        _our_pair(compiled, hxx[compiled.idx["k_lag1"]]),
        _dynare_pair(golden.GHXX_KPRIME),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(hxx[compiled.idx["z_lag1"]], 0.0, atol=1e-12)

    # Risk correction vs ghs2: eta loads the single shock (std sig) on the lifted
    # shock state; x' = h(x) + eta @ eps.
    sig = float(calib[sp.Symbol("sig")])
    eta = np.zeros((n_state, 1), dtype=np.float64)
    eta[compiled.idx["e_st"], 0] = sig
    gss, hss = second_order_risk(a, b, f_xx, gx, gxx, eta, n_state)
    np.testing.assert_allclose(
        [hss[compiled.idx["k_lag1"]], hss[compiled.idx["z_lag1"]], gss[ctrl["c"]]],
        golden.GHS2,
        rtol=5e-6,
        atol=1e-9,
    )


def test_rbc_shock_state_columns_carry_dynare_ghxu_ghuu():
    """Desugaring lifts the shock into a state, so what Dynare reports as ghxu
    and ghuu lands inside our g_xx/h_xx as the e_st column. That column is a
    third of every tensor row and the ghxx assertions cover none of it.
    """
    solved, compiled = _solved_rbc()
    pol = solved.policy
    ctrl = {n: i for i, n in enumerate(compiled.var_names[compiled.n_state :])}

    np.testing.assert_allclose(
        _our_shock_block(compiled, pol.gxx[ctrl["c"]]),
        _dynare_shock_block(golden.DR_C),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        _our_shock_block(compiled, pol.hxx[compiled.idx["k_lag1"]]),
        _dynare_shock_block(golden.DR_KPRIME),
        rtol=5e-6,
        atol=2e-7,
    )


def test_rbc_second_order_structural_invariants():
    """What the compiler's own layout forces, independent of any golden.

    The lifted shock is exogenous and i.i.d., so it has no transition row and no
    risk correction. z_lag1(t+1) = z(t) is linear, so its row is flat. And
    k_lag1(t+1) = k(t) makes the k_lag1 state row the same object as the k
    control row, which is the identity that lets a ghxx golden be read off
    either tensor.
    """
    solved, compiled = _solved_rbc()
    pol = solved.policy
    ctrl = {n: i for i, n in enumerate(compiled.var_names[compiled.n_state :])}

    np.testing.assert_allclose(pol.hxx[compiled.idx["e_st"]], 0.0, atol=1e-12)
    np.testing.assert_allclose(pol.hxx[compiled.idx["z_lag1"]], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        pol.hxx[compiled.idx["k_lag1"]], pol.gxx[ctrl["k"]], rtol=1e-12, atol=1e-14
    )
    assert pol.hss[compiled.idx["e_st"]] == 0.0


def test_solve_order2_wiring():
    """The .solve(order=2) public path end to end: it resolves + cross-checks the
    nonlinear steady state, builds eta from the shock calibration, and returns a
    PerturbationSolution whose tensors match the Dynare goldens. order=1 is
    unchanged (a plain KleinSolution with no second-order fields)."""
    model, kalman = ModelParser("tests/fixtures/models/rbc_second_order.yaml").get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    solved = solver.solve(compiled, order=2)
    pol = solved.policy
    assert pol.order == 2

    ss = _levels_steady_state(compiled)
    n_state = compiled.n_state
    ctrl = {n: i for i, n in enumerate(compiled.var_names[n_state:])}
    # steady state was solved/validated to the nonlinear point.
    np.testing.assert_allclose(pol.steady_state, ss, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        _our_pair(compiled, pol.gxx[ctrl["c"]]),
        _dynare_pair(golden.GHXX_C),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        _our_pair(compiled, pol.hxx[compiled.idx["k_lag1"]]),
        _dynare_pair(golden.GHXX_KPRIME),
        rtol=5e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        [
            pol.hss[compiled.idx["k_lag1"]],
            pol.hss[compiled.idx["z_lag1"]],
            pol.gss[ctrl["c"]],
        ],
        golden.GHS2,
        rtol=5e-6,
        atol=1e-9,
    )
    # First order path is untouched: KleinSolution, no second-order tensors.
    # (levels model -> the expansion point must be supplied; zeros would fail BK.)
    first = solver.solve(compiled, order=1, ss_seed=ss)
    assert first.policy.order == 1
    assert not hasattr(first.policy, "gxx")


def test_rbc_second_order_deterministic_sim_matches_dynare():
    solved = _solve_rbc_second_order()

    out = solved.sim(
        golden.DETERMINISTIC_SIM.shape[0] - 1,
        x0=_golden_x0(solved),
    )[
        "_X"
    ][:, _golden_columns(solved)]

    np.testing.assert_allclose(
        out,
        golden.DETERMINISTIC_SIM[1:],
        rtol=2e-6,
        atol=2e-6,
    )


def test_rbc_second_order_stochastic_sim_matches_dynare():
    solved = _solve_rbc_second_order()

    out = solved.sim(
        golden.STOCHASTIC_SIM.shape[0] - 1,
        x0=_golden_x0(solved),
        shocks={"e": golden.STOCHASTIC_SHOCKS},
    )["_X"][:, _golden_columns(solved)]

    np.testing.assert_allclose(
        out,
        golden.STOCHASTIC_SIM[1:],
        rtol=2e-6,
        atol=2e-6,
    )


def test_rbc_second_order_irf_matches_dynare():
    solved = _solve_rbc_second_order()

    out = solved.irf(["e"], T=golden.IRF.shape[0] - 1)["_X"][:, _golden_columns(solved)]

    np.testing.assert_allclose(out, golden.IRF[1:], rtol=2e-6, atol=2e-6)
