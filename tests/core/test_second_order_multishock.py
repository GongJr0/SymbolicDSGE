"""Dynare parity for the three-shock correlated RBC at second order.

The decision rule, the risk correction against a full covariance rather than one
variance, and the pruned simulation and IRF paths with all three innovations
live at once. Goldens and the orderings that map them onto our layout are in
:mod:`_oracles.dynare_rbc_multishock_second_order`.

This fixture exists for what the single-shock one in test_second_order_rbc.py
cannot reach: every cross term between distinct innovations, the Cholesky branch
of the eta builder, and a correlated ghs2.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from SymbolicDSGE._ckernels.core._core import bicomplex_hessian, klein_preprocess
from SymbolicDSGE._ckernels.core import second_order_risk
from SymbolicDSGE.core import DSGESolver, ModelParser
from _oracles import dynare_rbc_multishock_second_order as golden

_MULTISHOCK = "tests/fixtures/models/rbc_multishock_second_order.yaml"
_MS_SHOCK_STATES = ("e_z_st", "e_d_st", "e_g_st")
_MS_STDS = ("sig_z", "sig_d", "sig_g")
_MS_CORRS = {(0, 1): "corr_zd", (0, 2): "corr_zg", (1, 2): "corr_dg"}
_MS_SHOCKS = ("e_z", "e_d", "e_g")


def _solved_multishock():
    """The three-shock RBC solved to second order, as (solved, compiled)."""
    model, kalman = ModelParser(_MULTISHOCK).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return solver.solve(compiled, order=2), compiled


def _ms_preproc(solved, compiled):
    """(a, b, f_xx) at the solved steady state, enough to re-run the risk
    correction with an eta of our choosing."""
    par = np.array(
        [
            float(compiled.config.calibration.parameters[p])
            for p in compiled.calib_params
        ],
        dtype=np.float64,
    )
    n_eq = len(compiled.var_names)
    ss = solved.policy.steady_state
    cf = compiled.construct_objective_cfunc()
    cf_bc = compiled.construct_objective_cfunc_bicomplex()
    a, b = klein_preprocess(cf.address, ss, par, n_eq, False)
    return a, b, bicomplex_hessian(cf_bc.address, ss, par, n_eq)


def _ms_covariance(compiled) -> np.ndarray:
    """The 3x3 innovation covariance the yaml calibrates, read straight off the
    parameters rather than off eta."""
    calib = compiled.config.calibration.parameters
    stds = np.array([float(calib[sp.Symbol(s)]) for s in _MS_STDS])
    corr = np.eye(3)
    for (i, j), name in _MS_CORRS.items():
        corr[i, j] = corr[j, i] = float(calib[sp.Symbol(name)])
    return corr * np.outer(stds, stds)


def _ms_dr_row(solved, compiled, dr_row: str):
    """(first order, second order, risk) for one Dynare DR row.

    Dynare's k row is next period's capital, which on our side is the k_lag1
    transition row; every other DR row is one of our controls."""
    pol = solved.policy
    if dr_row == "k":
        i = compiled.idx["k_lag1"]
        return np.real(pol.p)[i], pol.hxx[i], pol.hss[i]
    ctrl = {n: i for i, n in enumerate(compiled.var_names[compiled.n_state :])}
    i = ctrl[dr_row]
    return np.real(pol.f)[i], pol.gxx[i], pol.gss[i]


def test_multishock_first_order_matches_dynare():
    """ghx and ghu with three shocks. The lifted shock states hold ghu, so the
    two Dynare arrays are one slice of our first order rule each."""
    solved, compiled = _solved_multishock()
    si = [compiled.idx[n] for n in golden.DR_STATES]
    ei = [compiled.idx[n] for n in golden.DR_EXO]

    for r, name in enumerate(golden.DR_ROWS):
        first, _, _ = _ms_dr_row(solved, compiled, name)
        np.testing.assert_allclose(
            first[si], golden.GHX[r], rtol=5e-6, atol=2e-7, err_msg=f"ghx {name}"
        )
        np.testing.assert_allclose(
            first[ei], golden.GHU[r], rtol=5e-6, atol=2e-7, err_msg=f"ghu {name}"
        )


def test_multishock_second_order_matches_dynare():
    """ghxx, ghxu and ghuu with three shocks, every DR row.

    ghxu and ghuu are the payload: with one shock they are a rescaling of ghxx
    and constrain nothing new, but here they carry the cross terms between
    distinct innovations, which no single-shock model has.
    """
    solved, compiled = _solved_multishock()
    si = [compiled.idx[n] for n in golden.DR_STATES]
    ei = [compiled.idx[n] for n in golden.DR_EXO]
    ns, ne = len(si), len(ei)

    for r, name in enumerate(golden.DR_ROWS):
        _, tensor, _ = _ms_dr_row(solved, compiled, name)
        for label, ix, expected, shape in (
            ("ghxx", (si, si), golden.GHXX[r], (ns, ns)),
            ("ghxu", (si, ei), golden.GHXU[r], (ns, ne)),
            ("ghuu", (ei, ei), golden.GHUU[r], (ne, ne)),
        ):
            np.testing.assert_allclose(
                tensor[np.ix_(*ix)],
                np.asarray(expected, dtype=np.float64).reshape(shape),
                rtol=5e-6,
                atol=2e-7,
                err_msg=f"{label} {name}",
            )


def test_multishock_risk_correction_matches_dynare():
    """ghs2 against a full covariance rather than one variance. This is the only
    assertion in the file that the shock correlations reach the solution at all,
    since eta enters nothing else."""
    solved, compiled = _solved_multishock()
    ours = [_ms_dr_row(solved, compiled, name)[2] for name in golden.DR_ROWS]

    np.testing.assert_allclose(ours, golden.GHS2, rtol=5e-6, atol=1e-8)


def test_multishock_eta_reproduces_the_calibrated_covariance():
    """eta is a Cholesky factor, so only ``eta @ eta.T`` is meaningful, and that
    has to be the calibrated covariance on the exog-state rows and zero below."""
    _, compiled = _solved_multishock()
    eta = DSGESolver._build_eta(compiled)
    n_exog = compiled.n_exog

    assert eta.shape == (compiled.n_state, n_exog)
    np.testing.assert_allclose(
        eta[:n_exog] @ eta[:n_exog].T, _ms_covariance(compiled), rtol=1e-13, atol=0.0
    )
    np.testing.assert_array_equal(eta[n_exog:], 0.0)


def test_multishock_risk_correction_reads_the_off_diagonals():
    """The correlations have to reach g_ss, and only through ``eta @ eta.T``.

    Zeroing the off-diagonals moves g_ss, so the covariance is not being read as
    a diagonal. Refactoring the same covariance through its symmetric square root
    instead of its Cholesky leaves g_ss alone, so nothing is reading the factor.
    """
    solved, compiled = _solved_multishock()
    a, b, f_xx = _ms_preproc(solved, compiled)
    gxx = solved.policy.gxx
    n_state, n_exog = compiled.n_state, compiled.n_exog

    eta = DSGESolver._build_eta(compiled)
    cov = _ms_covariance(compiled)
    gx = np.real(solved.policy.f)

    diag = np.zeros_like(eta)
    diag[:n_exog, :] = np.diag(np.sqrt(np.diag(cov)))
    w, v = np.linalg.eigh(cov)
    root = np.zeros_like(eta)
    root[:n_exog, :] = v @ np.diag(np.sqrt(w)) @ v.T

    gss = second_order_risk(a, b, f_xx, gx, gxx, eta, n_state)[0]
    gss_diag = second_order_risk(a, b, f_xx, gx, gxx, diag, n_state)[0]
    gss_root = second_order_risk(a, b, f_xx, gx, gxx, root, n_state)[0]

    assert not np.allclose(gss, gss_diag, rtol=1e-3, atol=1e-12)
    np.testing.assert_allclose(gss, gss_root, rtol=1e-10, atol=1e-15)


def test_multishock_second_order_structural_invariants():
    """The same layout facts the single-shock fixture pins, once per shock.

    Every lifted shock is exogenous and i.i.d., so it has no transition row and
    no risk correction; every AR(1) lag row is linear, so it is flat; and
    k_lag1(t+1) = k(t) keeps the k_lag1 state row equal to the k control row.
    """
    solved, compiled = _solved_multishock()
    pol = solved.policy
    ctrl = {n: i for i, n in enumerate(compiled.var_names[compiled.n_state :])}

    np.testing.assert_allclose(pol.gxx, pol.gxx.transpose(0, 2, 1), rtol=0, atol=0)
    np.testing.assert_allclose(pol.hxx, pol.hxx.transpose(0, 2, 1), rtol=0, atol=0)

    linear = _MS_SHOCK_STATES + ("z_lag1", "d_lag1", "g_lag1")
    for name in linear:
        np.testing.assert_allclose(pol.hxx[compiled.idx[name]], 0.0, atol=1e-12)
    for name in _MS_SHOCK_STATES:
        np.testing.assert_allclose(pol.hss[compiled.idx[name]], 0.0, atol=1e-12)

    np.testing.assert_allclose(
        pol.hxx[compiled.idx["k_lag1"]], pol.gxx[ctrl["k"]], rtol=1e-12, atol=1e-14
    )


# --- three correlated shocks, simulation and IRF -----------------------------
# The decision-rule tests above verify ghxu and ghuu as arrays. They do not
# verify that the simulator contracts them: with one innovation the cross terms
# live inside the ghxx column the pruned recursion already touches, so a bug in
# the cross-shock contraction passes everything in the file. These three drive
# the order-2 recursion with all three innovations live at once.
#
# Goldens come from make_rbc_multishock_second_order_sim_goldens.m, which prints
# them ready to paste. Columns are [z, d, g, k, c] with k dated as the
# predetermined stock, matching the single-shock sim goldens above.


def _ms_golden_columns(solved) -> list[int]:
    """Our columns for the golden's [z, d, g, k, c]. As with the single-shock
    goldens the reported k is the predetermined stock, which is our k_lag1."""
    idx = solved.compiled.idx
    return [idx["z"], idx["d"], idx["g"], idx["k_lag1"], idx["c"]]


def _ms_golden_x0(solved) -> np.ndarray:
    """The golden initial condition over our state block
    [e_z_st, e_d_st, e_g_st, k_lag1, z_lag1, d_lag1, g_lag1].

    The golden reports the three processes contemporaneously but starts the path
    one period earlier, so each initial lag is one AR step behind its value.
    """
    calib = solved.compiled.config.calibration.parameters
    rho_z, rho_d, rho_g = (
        float(calib[sp.Symbol(n)]) for n in ("rho_z", "rho_d", "rho_g")
    )
    z0, d0, g0, k0 = golden.SIM_X0
    return np.array(
        [0.0, 0.0, 0.0, k0, z0 / rho_z, d0 / rho_d, g0 / rho_g], dtype=np.float64
    )


def test_multishock_second_order_deterministic_sim_matches_dynare():
    """No innovations, so this is hxx/gxx on a four-state model plus the
    correlated risk correction, which the pruned recursion applies every period
    whether or not a shock lands."""
    solved, _ = _solved_multishock()

    out = solved.sim(
        golden.DETERMINISTIC_SIM.shape[0] - 1,
        x0=_ms_golden_x0(solved),
    )[
        "_X"
    ][:, _ms_golden_columns(solved)]

    np.testing.assert_allclose(
        out,
        golden.DETERMINISTIC_SIM[1:],
        rtol=2e-6,
        atol=2e-6,
    )


def test_multishock_second_order_stochastic_sim_matches_dynare():
    """Every period carries at least two nonzero innovations, so the ghuu
    off-diagonals contribute to every state update. Dynare's ex_ is in levels
    and our B is a plain selector, so the same innovations go into both sides."""
    solved, _ = _solved_multishock()

    out = solved.sim(
        golden.STOCHASTIC_SIM.shape[0] - 1,
        x0=_ms_golden_x0(solved),
        shocks={
            name: golden.STOCHASTIC_SHOCKS[:, i] for i, name in enumerate(_MS_SHOCKS)
        },
    )["_X"][:, _ms_golden_columns(solved)]

    np.testing.assert_allclose(
        out,
        golden.STOCHASTIC_SIM[1:],
        rtol=2e-6,
        atol=2e-6,
    )


def test_multishock_second_order_irf_matches_dynare():
    """All three shocks impulsed together. A one-shock IRF would leave the ghuu
    off-diagonals at zero and reduce to the single-shock fixture."""
    solved, _ = _solved_multishock()

    out = solved.irf(list(_MS_SHOCKS), T=golden.IRF.shape[0] - 1)["_X"][
        :, _ms_golden_columns(solved)
    ]

    np.testing.assert_allclose(out, golden.IRF[1:], rtol=2e-6, atol=2e-6)
