from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.core.resolvers import shock_covariance

from _oracles import dynare_post82_first_order as dyn

# Dynare solves to 1e-16 and these are transcribed doubles, so the only slack
# needed is accumulated floating point.
TOL = 1e-10

# The variables Dynare carries as states, named.
DYNARE_STATES = tuple(dyn.DECL_COLUMNS[i - 1] for i in dyn.STATE_VAR)


@pytest.fixture(scope="module")
def blocks(solved_post82):
    """Our decision rule: the state rows stacked over the controls, and the impact."""
    sol = solved_post82
    ghx = np.vstack([np.real(sol.policy.p), np.real(sol.policy.f)])
    return ghx, np.real(sol.policy.B)


def _declared(compiled) -> list[int]:
    return [compiled.idx[v] for v in dyn.DECL_COLUMNS]


def _x0() -> dict[str, float]:
    """Dynare's ``y0``, which is our ``x0`` outright now that the states agree."""
    return dict(zip(DYNARE_STATES, dyn.X0_STATES))


def _path(compiled, out) -> np.ndarray:
    return np.column_stack([out.states[name] for name in dyn.DECL_COLUMNS])


def _shocks(block: np.ndarray) -> dict[str, np.ndarray]:
    return {name: block[:, j] for j, name in enumerate(dyn.EXO_COLUMNS)}


# --- ordering ---------------------------------------------------------------
# Nothing below means anything if the columns are not the same columns.


def test_compiled_order_is_dynares_declaration_order(compiled_post82):
    compiled = compiled_post82
    assert compiled.layout.canonical_names == dyn.DECL_COLUMNS
    assert compiled.layout.state_names == DYNARE_STATES


def test_shock_columns_match_dynares_exogenous_order(compiled_post82):
    assert compiled_post82.shock_names == dyn.EXO_COLUMNS


# --- the decision rule ------------------------------------------------------


def test_policy_lag_block_matches_ghx(blocks):
    ghx, _ = blocks
    assert np.abs(ghx - dyn.GHX_DECL).max() < TOL


def test_policy_shock_block_matches_ghu(blocks):
    _, ghu = blocks
    assert np.abs(ghu - dyn.GHU_DECL).max() < TOL


def test_impact_on_declared_variables_matches_ghu(solved_post82):
    # The headline. The r row is the fixed point issue #390 was opened over.
    compiled = solved_post82.compiled
    solution = solved_post82
    B = np.real(solution.policy.B)

    assert np.abs(B[_declared(compiled)] - dyn.GHU_DECL).max() < TOL
    r_row = B[compiled.idx["r"]]
    assert r_row[dyn.EXO_COLUMNS.index("e_r")] == pytest.approx(0.33355298, abs=1e-8)
    assert r_row[dyn.EXO_COLUMNS.index("e_g")] == pytest.approx(0.34702357, abs=1e-8)


def test_shock_covariance_matches_dynare(parsed_post82):
    Q = shock_covariance(parsed_post82, parsed_post82.calibration.parameters)

    assert np.abs(Q - dyn.SIGMA_E).max() < TOL


# --- the transition ---------------------------------------------------------
# The simulations below run A and B with no f in the loop, so they are what
# exercises A. These state what A holds, which a path only implies.


def test_transition_matches_dynares_A(solved_post82):
    # A maps y_{t-1} to y_t, so its state columns are the decision rule itself
    # and a control at t-1 reaches nothing.
    solution = solved_post82
    A = np.real(solution.policy.A)
    n_state = solution.compiled.n_state

    assert np.abs(A - dyn.A_DECL).max() < TOL
    assert np.abs(A[:, :n_state] - dyn.GHX_DECL).max() < TOL
    assert np.abs(A[:, n_state:]).max() < TOL


# --- simulation -------------------------------------------------------------


def test_deterministic_path_matches_dynare(solved_post82):
    # Driven by the transition alone, so it also pins the row convention: row 0
    # is the first simulated period, not the initial condition.
    compiled = solved_post82.compiled
    solution = solved_post82
    out = solution.sim(T=len(dyn.DET), x0=_x0())

    assert np.abs(_path(compiled, out) - dyn.DET).max() < TOL


def test_stochastic_path_matches_dynare(solved_post82):
    solution = solved_post82
    out = solution.sim(
        T=len(dyn.STOCH),
        shocks=_shocks(dyn.SHOCK_BLOCK),
        x0=_x0(),
    )

    assert np.abs(_path(solution.compiled, out) - dyn.STOCH).max() < TOL


@pytest.mark.parametrize("shock", dyn.EXO_COLUMNS)
def test_unit_innovation_response_matches_dynare(solved_post82, shock):
    # From the steady state, one unit innovation on the first simulated period,
    # which is the experiment the golden ran.
    solution = solved_post82
    horizon = len(dyn.IRF[shock])
    block = np.zeros((horizon, len(dyn.EXO_COLUMNS)))
    block[0, dyn.EXO_COLUMNS.index(shock)] = 1.0

    out = solution.sim(T=horizon, shocks=_shocks(block))

    assert np.abs(_path(solution.compiled, out) - dyn.IRF[shock]).max() < TOL
