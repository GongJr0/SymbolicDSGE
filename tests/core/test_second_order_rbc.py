"""Dynare parity for the single-shock RBC at second order.

The decision rule (ghxx, ghxu, ghuu, ghs2), the ``.solve(order=2)`` public path,
and the pruned simulation and IRF paths. Goldens and the orderings that map them
onto our layout are in :mod:`_oracles.dynare_rbc_second_order`.

Three things this fixture cannot reach, and test_second_order_multishock.py
exists for: cross terms between distinct innovations, the Cholesky branch of the
shock covariance, and a risk correction against a full covariance.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from _oracles import dynare_rbc_second_order as golden

# Dynare's DR order [k', z, c] is our canonical order [k, z, c] row for row: k'
# is k_t, and both layouts put the states first in declaration order. Every
# golden below is therefore read straight off the stacked tensors, with the
# states taken from the h block and the controls from the g block.
N_VAR, N_STATE = 3, 2


def _stack(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The state and control blocks of one tensor as Dynare's stacked rule."""
    return np.concatenate([h, g])


def _dynare_ghxx() -> np.ndarray:
    """The (n_var, nx, nx) golden. Only k' and c are nonlinear; z is an AR(1)."""
    out = np.zeros((N_VAR, N_STATE, N_STATE))
    out[golden.DR_KPRIME] = np.reshape(golden.GHXX_KPRIME, (N_STATE, N_STATE))
    out[golden.DR_C] = np.reshape(golden.GHXX_C, (N_STATE, N_STATE))
    return out


def _dynare_ghxu() -> np.ndarray:
    """The (n_var, nx, ne) golden. Dynare flattens (n_var, nx*ne) column major,
    so the stored order runs down the rows of each state column."""
    return np.reshape(golden.GHXU, (N_STATE, N_VAR)).T[:, :, None]


def _dynare_ghuu() -> np.ndarray:
    """The (n_var, ne, ne) golden."""
    return np.reshape(golden.GHUU, (N_VAR, 1, 1))


def _levels_steady_state(config, compiled) -> np.ndarray:
    """The RBC expansion point over the compiled layout."""
    calib = config.calibration.parameters
    known = {"k": calib["k_ss"], "c": calib["c_ss"]}
    return np.array([known.get(n, 0.0) for n in compiled.var_names], dtype=np.float64)


def _golden_columns(solved) -> list[int]:
    """Our columns matching the golden's stored [z, k, c]."""
    idx = solved.compiled.idx
    return [idx["z"], idx["k"], idx["c"]]


def _golden_x0(solved) -> dict[str, float]:
    """The Dynare initial condition, by name.

    A dense x0 would be read in declaration order, which is not our canonical
    order, so the states are named instead.

    The generator seeds Dynare's own y0 with ``z0 / rho`` so that its first
    simulated z is the printed ``z0``; our x0 is that same y0 state.
    """
    rho = float(solved.config.calibration.parameters[sp.Symbol("rho")])
    z0, k0 = golden.SIM_X0
    return {"k": float(k0), "z": float(z0) / rho}


_K_COL = 1


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


def test_rbc_second_order_structural_invariants(rbc_so_bundle):
    """What the model's own structure forces, independent of any golden.

    z is an AR(1), so every one of its second-order rows is flat, and its risk
    correction is zero. The tensors are symmetric in the pair they weigh, which
    the solve gets from the chain rule rather than by imposing it.
    """
    solved = rbc_so_bundle["solved"]
    compiled = rbc_so_bundle["compiled"]
    pol = solved.policy
    z = compiled.idx["z"]

    np.testing.assert_allclose(pol.hxx[z], 0.0, atol=1e-12)
    np.testing.assert_allclose(pol.hxu[z], 0.0, atol=1e-12)
    np.testing.assert_allclose(pol.huu[z], 0.0, atol=1e-12)
    assert pol.hss[z] == 0.0

    np.testing.assert_allclose(
        pol.hxx, np.transpose(pol.hxx, (0, 2, 1)), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        pol.gxx, np.transpose(pol.gxx, (0, 2, 1)), rtol=1e-12, atol=1e-14
    )


def test_solve_order2_wiring(rbc_so_bundle):
    """The .solve(order=2) public path end to end: it resolves + cross-checks the
    nonlinear steady state and returns a SecondOrderSolution whose tensors match
    the Dynare goldens. order=1 is unchanged (no second-order fields)."""
    solved = rbc_so_bundle["solved"]
    compiled = rbc_so_bundle["compiled"]
    solver = rbc_so_bundle["solver"]

    pol = solved.policy
    assert pol.order == 2

    ss = _levels_steady_state(solved.config, compiled)
    np.testing.assert_allclose(pol.steady_state, ss, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        _stack(pol.hxx, pol.gxx), _dynare_ghxx(), rtol=5e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        _stack(pol.hxu, pol.gxu), _dynare_ghxu(), rtol=5e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        _stack(pol.huu, pol.guu), _dynare_ghuu(), rtol=5e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        _stack(pol.hss, pol.gss), golden.GHS2, rtol=5e-6, atol=1e-9
    )

    # First order path is untouched: no second-order tensors.
    # (levels model -> the expansion point must be supplied; zeros would fail BK.)
    first = solver.solve(compiled, order=1)
    assert first.policy.order == 1
    assert not hasattr(first.policy, "gxx")


def test_rbc_second_order_deterministic_sim_matches_dynare(solved_rbc_so):
    solved = solved_rbc_so

    out = solved.sim(
        golden.DETERMINISTIC_SIM.shape[0] - 1,
        x0=_golden_x0(solved),
    ).X[:, _golden_columns(solved)]

    np.testing.assert_allclose(
        out,
        _golden_rows(golden.DETERMINISTIC_SIM),
        rtol=2e-6,
        atol=2e-6,
    )


def test_rbc_second_order_stochastic_sim_matches_dynare(solved_rbc_so):
    solved = solved_rbc_so

    # The generator hands Dynare a leading zero shock row, so the first period
    # it simulates is shock free. Ours has to be too, or every row is offset.
    shocks = np.concatenate([[0.0], golden.STOCHASTIC_SHOCKS])
    out = solved.sim(
        len(shocks),
        x0=_golden_x0(solved),
        shocks={"e": shocks},
    ).X[:, _golden_columns(solved)]

    expected = _golden_rows(golden.STOCHASTIC_SIM)
    np.testing.assert_allclose(out[: len(expected)], expected, rtol=2e-6, atol=2e-6)


def test_rbc_second_order_irf_matches_dynare(solved_rbc_so):
    solved = solved_rbc_so

    # The generator impulses ex_(2), one period after its lead-in zero, while
    # irf() impulses the first period it simulates.
    out = solved.irf(["e"], T=len(golden.IRF)).X[:, _golden_columns(solved)]

    expected = _golden_rows(golden.IRF, lead=1)
    np.testing.assert_allclose(out[: len(expected)], expected, rtol=2e-6, atol=2e-6)
