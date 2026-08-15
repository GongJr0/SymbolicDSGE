"""Dynare parity for the three-shock correlated RBC at second order.

The decision rule, the risk correction against a full covariance rather than one
variance, and the pruned simulation and IRF paths with all three innovations
live at once. Goldens and the orderings that map them onto our layout are in
:mod:`_oracles.dynare_rbc_multishock_second_order`.

This fixture exists for what the single-shock one in test_second_order_rbc.py
cannot reach: every cross term between distinct innovations, a full shock
covariance, and a correlated ghs2.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from SymbolicDSGE._ckernels.core._core import bicomplex_hessian, klein_preprocess
from SymbolicDSGE._ckernels.core import second_order
from SymbolicDSGE.core import DSGESolver, ModelParser
from _oracles import dynare_rbc_multishock_second_order as golden

_MULTISHOCK = "tests/fixtures/models/rbc_multishock_second_order.yaml"
_MS_STDS = ("sig_z", "sig_d", "sig_g")
_MS_CORRS = {(0, 1): "corr_zd", (0, 2): "corr_zg", (1, 2): "corr_dg"}
_MS_SHOCKS = ("e_z", "e_d", "e_g")
_MS_AR_STATES = ("z", "d", "g")


def _solved_multishock():
    """The three-shock RBC solved to second order, as (solved, compiled)."""
    model, kalman = ModelParser(_MULTISHOCK).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return solver.solve(compiled, order=2), compiled


def _ms_preproc(solved, compiled):
    """(a, b, f_xx) at the solved steady state, enough to re-run the solve with a
    covariance of our choosing."""
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
    a, b, _, _ = klein_preprocess(cf.address, ss, par, n_eq, compiled.n_exog)
    return a, b, bicomplex_hessian(cf_bc.address, ss, par, compiled.n_exog, n_eq)


def _ms_covariance(compiled) -> np.ndarray:
    """The 3x3 innovation covariance the yaml calibrates, read straight off the
    parameters."""
    calib = compiled.config.calibration.parameters
    stds = np.array([float(calib[sp.Symbol(s)]) for s in _MS_STDS])
    corr = np.eye(3)
    for (i, j), name in _MS_CORRS.items():
        corr[i, j] = corr[j, i] = float(calib[sp.Symbol(name)])
    return corr * np.outer(stds, stds)


def _stack(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The state and control blocks of one tensor as Dynare's stacked rule. Our
    canonical order is states then controls, so ``compiled.idx`` indexes it."""
    return np.concatenate([h, g])


def _ms_axes(compiled) -> tuple[list[int], list[int], list[int]]:
    """(row, state column, shock column) permutations from our order to the
    golden's. Dynare's k row is next period's capital, which is our k."""
    ri = [compiled.idx[n] for n in golden.DR_ROWS]
    si = [compiled.idx[n] for n in golden.DR_STATES]
    ei = [compiled.shock_names.index(n) for n in golden.DR_EXO]
    return ri, si, ei


def test_multishock_first_order_matches_dynare():
    """ghx and ghu with three shocks."""
    solved, compiled = _solved_multishock()
    pol = solved.policy
    ri, si, ei = _ms_axes(compiled)

    ghx = _stack(np.real(pol.p), np.real(pol.f))[np.ix_(ri, si)]
    ghu = np.real(pol.B)[np.ix_(ri, ei)]

    np.testing.assert_allclose(ghx, golden.GHX, rtol=5e-6, atol=2e-7)
    np.testing.assert_allclose(ghu, golden.GHU, rtol=5e-6, atol=2e-7)


def test_multishock_second_order_matches_dynare():
    """ghxx, ghxu and ghuu with three shocks, every DR row.

    ghxu and ghuu are the payload: with one shock they are a rescaling of ghxx
    and constrain nothing new, but here they carry the cross terms between
    distinct innovations, which no single-shock model has.
    """
    solved, compiled = _solved_multishock()
    pol = solved.policy
    ri, si, ei = _ms_axes(compiled)
    ns, ne = len(si), len(ei)

    for label, tensor, ix, expected, shape in (
        ("ghxx", _stack(pol.hxx, pol.gxx), (ri, si, si), golden.GHXX, (ns, ns)),
        ("ghxu", _stack(pol.hxu, pol.gxu), (ri, si, ei), golden.GHXU, (ns, ne)),
        ("ghuu", _stack(pol.huu, pol.guu), (ri, ei, ei), golden.GHUU, (ne, ne)),
    ):
        got = tensor[np.ix_(*ix)]
        for r, name in enumerate(golden.DR_ROWS):
            np.testing.assert_allclose(
                got[r],
                np.asarray(expected[r], dtype=np.float64).reshape(shape),
                rtol=5e-6,
                atol=2e-7,
                err_msg=f"{label} {name}",
            )


def test_multishock_risk_correction_matches_dynare():
    """ghs2 against a full covariance rather than one variance. This is the only
    assertion in the file that the shock correlations reach the solution at all,
    since the covariance enters nothing else."""
    solved, compiled = _solved_multishock()
    ri, _, _ = _ms_axes(compiled)
    ours = _stack(solved.policy.hss, solved.policy.gss)[ri]

    np.testing.assert_allclose(ours, golden.GHS2, rtol=5e-6, atol=1e-8)


def test_multishock_Q_reproduces_the_calibrated_covariance():
    """The stds scale it and the correlations fill it."""
    _, compiled = _solved_multishock()
    Q = DSGESolver._build_Q(compiled)

    assert Q.shape == (compiled.n_exog, compiled.n_exog)
    np.testing.assert_allclose(Q, _ms_covariance(compiled), rtol=1e-13, atol=0.0)


def test_multishock_risk_correction_reads_the_off_diagonals():
    """The correlations have to reach g_ss. Zeroing the off-diagonals moves it,
    so the covariance is not being read as a diagonal."""
    solved, compiled = _solved_multishock()
    a, b, f_xx = _ms_preproc(solved, compiled)
    n_state = compiled.n_state
    pol = solved.policy
    bu = np.real(pol.B)

    cov = _ms_covariance(compiled)
    gss = second_order(a, b, f_xx, pol.f, pol.p, bu, cov, n_state)[6]
    gss_diag = second_order(
        a, b, f_xx, pol.f, pol.p, bu, np.diag(np.diag(cov)), n_state
    )[6]

    np.testing.assert_allclose(gss, pol.gss, rtol=1e-10, atol=1e-15)
    assert not np.allclose(gss, gss_diag, rtol=1e-3, atol=1e-12)


def test_multishock_second_order_structural_invariants():
    """The same layout facts the single-shock fixture pins, once per process.

    Every AR(1) is linear, so its second-order rows are flat and it takes no risk
    correction. The tensors are symmetric in the pair they weigh.
    """
    solved, compiled = _solved_multishock()
    pol = solved.policy

    # Symmetry falls out of the chain rule rather than being imposed by a
    # symmetry-reduced system, so it holds to roundoff and not bitwise.
    for block in (pol.gxx, pol.hxx, pol.guu, pol.huu):
        np.testing.assert_allclose(
            block, block.transpose(0, 2, 1), rtol=1e-12, atol=1e-14
        )

    for name in _MS_AR_STATES:
        i = compiled.idx[name]
        np.testing.assert_allclose(pol.hxx[i], 0.0, atol=1e-12)
        np.testing.assert_allclose(pol.hxu[i], 0.0, atol=1e-12)
        np.testing.assert_allclose(pol.huu[i], 0.0, atol=1e-12)
        np.testing.assert_allclose(pol.hss[i], 0.0, atol=1e-12)


# --- three correlated shocks, simulation and IRF -----------------------------
# The decision-rule tests above verify ghxu and ghuu as arrays. They do not
# verify that the simulator contracts them: with one innovation the cross terms
# are a rescaling of terms the pruned recursion already touches, so a bug in the
# cross-shock contraction passes everything in the file. These three drive the
# order-2 recursion with all three innovations live at once.
#
# Goldens come from make_rbc_multishock_second_order_sim_goldens.m, which prints
# them ready to paste. Columns are [z, d, g, k, c] with k dated as the
# predetermined stock, matching the single-shock sim goldens above.


def _ms_golden_columns(solved) -> list[int]:
    """Our columns for the golden's [z, d, g, k, c]."""
    idx = solved.compiled.idx
    return [idx["z"], idx["d"], idx["g"], idx["k"], idx["c"]]


def _ms_golden_x0(solved) -> dict[str, float]:
    """The golden initial condition, by name.

    A dense x0 would be read in declaration order, which is not our canonical
    order, so the states are named instead.

    The generator seeds Dynare's own y0 with each process one AR step back so
    that its first simulated value is the printed one; our x0 is that y0 state.
    """
    calib = solved.compiled.config.calibration.parameters
    rho_z, rho_d, rho_g = (
        float(calib[sp.Symbol(n)]) for n in ("rho_z", "rho_d", "rho_g")
    )
    z0, d0, g0, k0 = golden.SIM_X0
    return {
        "k": float(k0),
        "z": float(z0) / rho_z,
        "d": float(d0) / rho_d,
        "g": float(g0) / rho_g,
    }


_K_COL = 3


def _golden_rows(block: np.ndarray, lead: int = 0) -> np.ndarray:
    """The golden rows our path lines up with, in the golden's column order.

    Two offsets, both from the generator rather than from the model:

    * k is stored one period behind the rest, as the predetermined stock. Under
      the old layout the lag aux ``k_lag1`` supplied that series directly; our k
      is the contemporaneous stock, so its column is taken one row later.
    * ``lead`` drops rows the generator simulates before ours begin. The shock
      vector handed to Dynare starts with a zero row, so its first simulated
      period is shock free.
    """
    n = len(block) - 1 - lead
    rows = block[lead : lead + n].copy()
    rows[:, _K_COL] = block[lead + 1 : lead + 1 + n, _K_COL]
    return rows


def test_multishock_second_order_deterministic_sim_matches_dynare():
    """No innovations, so this is hxx/gxx on a four-state model plus the
    correlated risk correction, which the pruned recursion applies every period
    whether or not a shock lands."""
    solved, _ = _solved_multishock()

    out = solved.sim(
        golden.DETERMINISTIC_SIM.shape[0] - 1,
        x0=_ms_golden_x0(solved),
    ).X[:, _ms_golden_columns(solved)]

    np.testing.assert_allclose(
        out,
        _golden_rows(golden.DETERMINISTIC_SIM),
        rtol=2e-6,
        atol=2e-6,
    )


def test_multishock_second_order_stochastic_sim_matches_dynare():
    """Every period carries at least two nonzero innovations, so the ghuu
    off-diagonals contribute to every state update. Dynare's ex_ is in levels
    and our B is a plain selector, so the same innovations go into both sides."""
    solved, _ = _solved_multishock()

    # The generator hands Dynare a leading zero shock row, so the first period
    # it simulates is shock free. Ours has to be too, or every row is offset.
    shocks = np.vstack([np.zeros((1, len(_MS_SHOCKS))), golden.STOCHASTIC_SHOCKS])
    out = solved.sim(
        len(shocks),
        x0=_ms_golden_x0(solved),
        shocks={name: shocks[:, i] for i, name in enumerate(_MS_SHOCKS)},
    ).X[:, _ms_golden_columns(solved)]

    expected = _golden_rows(golden.STOCHASTIC_SIM)
    np.testing.assert_allclose(out[: len(expected)], expected, rtol=2e-6, atol=2e-6)


def test_multishock_second_order_irf_matches_dynare():
    """All three shocks impulsed together. A one-shock IRF would leave the ghuu
    off-diagonals at zero and reduce to the single-shock fixture."""
    solved, _ = _solved_multishock()

    # The generator impulses ex_(2), one period after its lead-in zero, while
    # irf() impulses the first period it simulates.
    out = solved.irf(list(_MS_SHOCKS), T=len(golden.IRF)).X[
        :, _ms_golden_columns(solved)
    ]

    expected = _golden_rows(golden.IRF, lead=1)
    np.testing.assert_allclose(out[: len(expected)], expected, rtol=2e-6, atol=2e-6)
