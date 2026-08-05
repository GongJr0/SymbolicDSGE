"""Exact parity against Dynare's first-order solution of POST82.

``tests/fixtures/models/POST82.yaml`` and
``tests/fixtures/models/post82_first_order.mod`` are the same model written
twice, both at natural dating. Goldens are transcribed in
``tests/_oracles/dynare_post82_first_order``, raw: nothing was re-dated,
rescaled or reordered on the way out of Dynare.

Dynare's state is the model's own lagged variables and ours is the block the
compiler mints, so the two decision rules are related by a selection rather than
a transform. ``ghx`` and ``ghu`` are our policy function ``f``, read on the
declared rows against the lag and shock state columns, and ``B``'s declared rows
are ``ghu`` outright.

Two objects are stated in Dynare's coordinates and have to be carried into ours.
Both are exact, and both are set up from our own ``f`` rather than from a golden:

* an initial condition. Dynare's ``y0`` names ``(g, z, r)`` at date 0, which our
  state reaches only through ``f``, so the lag values behind it are a 3x3 solve.
* ``P0``. The unconditional covariance is the same object in either state space,
  so that case is parity outright. ``P0 = I`` is not: Dynare's own ``A``
  annihilates its ``x`` and ``Pi`` block, which leaves ``I`` on ``(g, z, r)``,
  and the lag covariance carrying that through ``f`` is ``inv(M) inv(M).T``.

The filter goldens are Dynare's own, not a second recursion of ours: the
likelihoods come from ``kalman_filter.m`` called directly, the state paths from
Dynare's smoother. Two conventions differ, and both are asserted rather than
absorbed.

The first is what a row index means. Dynare dates a prediction by the information
set behind it, so ``oo_.FilteredVariables[t]`` is the forecast made standing at
``t``, which is ``a_{t+1|t}``. We date it by the period being estimated, so
``x_pred[t]`` is ``a_{t|t-1}``. Same quantity, one row apart, and each array is
complete under its own rule: ours runs ``a_{1|0}`` to ``a_{T|T-1}`` and Dynare's
``a_{2|1}`` to ``a_{T+1|T}``, the last of those a step past the sample.
``oo_.UpdatedVariables`` is ``a_{t|t}`` and lines up with ``x_filt`` directly.

The second is what ``P0`` covers. The two filters run the same recursion with
mirrored loop invariants: ``kalman_filter.m`` carries the prediction and folds
the update into the transition, ours carries the update and predicts at the top.
``P0`` is the covariance of whichever invariant the loop is entered on, so the
argument names covariances one period apart. The unconditional covariance is a
fixed point of the map between them, which is why that case alone needs no
translation.

This model is why the goldens are worth having. ``e_r`` enters a Taylor rule
carrying ``Pi(t)`` and ``x(t)``, so the impact of ``e_r`` on ``r`` is a fixed
point, 0.3336, and ``e_g`` moves ``r`` by 0.3470 through the same rule. Issue
#390 recorded a hand-built impact block writing those two entries as 1.0 and
0.0. Lifting every shock into a state of its own puts ``r`` in the control
block, where ``f`` carries the feedback.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.estimation import backend
from SymbolicDSGE.kalman.filter import KalmanFilter

from _oracles import dynare_post82_first_order as dyn

YAML = "tests/fixtures/models/POST82.yaml"

# Dynare solves to 1e-16 and these are transcribed doubles, so the only slack
# needed is accumulated floating point.
TOL = 1e-10

# The variables Dynare carries as states, named.
DYNARE_STATES = tuple(dyn.DECL_COLUMNS[i - 1] for i in dyn.STATE_VAR)


@pytest.fixture(scope="module")
def solved():
    model, kalman = ModelParser(YAML).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return compiled, solver.solve(compiled=compiled, order=1)


@pytest.fixture(scope="module")
def blocks(solved):
    """Our policy function on the rows and columns Dynare's decision rule spans."""
    compiled, solution = solved
    f = np.real(solution.policy.f)
    rows = [compiled.layout.control_names.index(v) for v in dyn.DECL_COLUMNS]
    return (
        f[np.ix_(rows, _lag_index(compiled))],
        f[np.ix_(rows, _shock_index(compiled))],
    )


def _declared(compiled) -> list[int]:
    return [compiled.idx[v] for v in dyn.DECL_COLUMNS]


def _lag_index(compiled) -> list[int]:
    """Canonical positions of the lag states behind Dynare's state variables.

    ``A`` is square over the canonical names, so these index rows and columns
    alike.
    """
    canonical = list(compiled.layout.canonical_names)
    return [canonical.index(f"{v}_lag1") for v in DYNARE_STATES]


def _shock_index(compiled) -> list[int]:
    canonical = list(compiled.layout.canonical_names)
    return [canonical.index(f"{s}_st") for s in dyn.EXO_COLUMNS]


def _state_rows() -> list[int]:
    """Rows of Dynare's decision rule belonging to its own state variables."""
    return [dyn.DECL_COLUMNS.index(v) for v in DYNARE_STATES]


def _x0(compiled, blocks) -> np.ndarray:
    """Dynare's ``y0`` carried into our state block."""
    ghx, _ = blocks

    x0 = np.zeros(len(compiled.var_names))
    x0[_lag_index(compiled)] = np.linalg.solve(ghx[: len(DYNARE_STATES)], dyn.X0_STATES)
    return x0


def _path(compiled, out) -> np.ndarray:
    return np.column_stack([out[name] for name in dyn.DECL_COLUMNS])


def _shocks(block: np.ndarray) -> dict[str, np.ndarray]:
    return {name: block[:, j] for j, name in enumerate(dyn.EXO_COLUMNS)}


def _measurement(compiled):
    params = backend.extract_base_params(compiled)
    C, d = compiled.build_affine_measurement_matrices(
        params, list(compiled.observable_names), np.zeros(len(compiled.var_names))
    )
    return backend.build_Q(compiled, params), C, d


# --- ordering ---------------------------------------------------------------
# Nothing below means anything if the columns are not the same columns.


def test_control_block_is_dynares_declaration_order(solved):
    compiled, _ = solved
    assert compiled.layout.control_names == dyn.DECL_COLUMNS


def test_shock_columns_match_dynares_exogenous_order(solved):
    compiled, _ = solved
    assert compiled.shock_names == dyn.EXO_COLUMNS


# --- the decision rule ------------------------------------------------------


def test_policy_lag_block_matches_ghx(blocks):
    ghx, _ = blocks
    assert np.abs(ghx - dyn.GHX_DECL).max() < TOL


def test_policy_shock_block_matches_ghu(blocks):
    _, ghu = blocks
    assert np.abs(ghu - dyn.GHU_DECL).max() < TOL


def test_impact_on_declared_variables_matches_ghu(solved):
    # The headline. The r row is the fixed point issue #390 was opened over.
    compiled, solution = solved
    B = np.real(solution.B)

    assert np.abs(B[_declared(compiled)] - dyn.GHU_DECL).max() < TOL
    r_row = B[compiled.idx["r"]]
    assert r_row[dyn.EXO_COLUMNS.index("e_r")] == pytest.approx(0.33355298, abs=1e-8)
    assert r_row[dyn.EXO_COLUMNS.index("e_g")] == pytest.approx(0.34702357, abs=1e-8)


def test_generated_states_take_no_shock_beyond_their_own(solved):
    compiled, solution = solved
    B = np.real(solution.B)
    n_exog = compiled.n_exog

    np.testing.assert_array_equal(B[:n_exog], np.eye(n_exog))
    np.testing.assert_array_equal(B[n_exog : compiled.layout.n_state], 0.0)


def test_shock_covariance_matches_dynare(solved):
    compiled, _ = solved
    Q, _, _ = _measurement(compiled)

    assert np.abs(Q - dyn.SIGMA_E).max() < TOL


# --- the transition ---------------------------------------------------------
# The simulations below run A and B with no f in the loop, so they are what
# exercises A. These state what A holds, which a path only implies.


def test_lag_state_rows_of_A_copy_the_policy_rows_they_lag(solved):
    # v_lag1(t+1) = v(t), so a lag row of A is that variable's policy row.
    compiled, solution = solved
    A, f = np.real(solution.A), np.real(solution.policy.f)
    n_state = compiled.layout.n_state
    lag = _lag_index(compiled)
    rows = [compiled.layout.control_names.index(v) for v in DYNARE_STATES]

    assert np.abs(A[np.ix_(lag, range(n_state))] - f[rows]).max() < TOL
    assert np.abs(A[lag, n_state:]).max() < TOL


def test_control_rows_of_A_are_f_composed_with_the_state_block(solved):
    # y_{t+1} = f x_{t+1}, so the controls carry no transition of their own and
    # A holds nothing beyond f and the state block.
    compiled, solution = solved
    A, f = np.real(solution.A), np.real(solution.policy.f)
    n_state = compiled.layout.n_state

    assert np.abs(A[n_state:] - f @ A[:n_state]).max() < TOL


def test_state_transition_matches_ghx(solved):
    compiled, solution = solved
    A = np.real(solution.A)
    lag = _lag_index(compiled)

    assert np.abs(A[np.ix_(lag, lag)] - dyn.GHX_DECL[_state_rows()]).max() < TOL


def test_A_on_declared_rows_is_one_transition_past_ghx(solved):
    # Our lag state at t-1 is g(t-2), so this is ghx @ ghx. It is why ghx is
    # compared against f above and never against A.
    compiled, solution = solved
    A = np.real(solution.A)
    block = A[np.ix_(_declared(compiled), _lag_index(compiled))]

    assert np.abs(block - dyn.GHX_DECL @ dyn.GHX_DECL[_state_rows()]).max() < TOL
    assert np.abs(block - dyn.GHX_DECL).max() > 1.0


def test_A_from_shock_states_to_declared_rows_is_ghx_ghu(solved):
    # A shock state holds its innovation for one period, so reaching a control
    # through it costs the same transition.
    compiled, solution = solved
    A = np.real(solution.A)
    block = A[np.ix_(_declared(compiled), _shock_index(compiled))]

    assert np.abs(block - dyn.GHX_DECL @ dyn.GHU_DECL[_state_rows()]).max() < TOL


def test_shock_state_rows_of_A_are_empty(solved):
    # A shock state is reached only by its own innovation, never by the state.
    compiled, solution = solved

    assert np.abs(np.real(solution.A)[: compiled.n_exog]).max() < TOL


# --- simulation -------------------------------------------------------------


def test_deterministic_path_matches_dynare(solved, blocks):
    # Driven by the transition alone, so it also pins the row convention: row 0
    # is the first simulated period, not the initial condition.
    compiled, solution = solved
    out = solution.sim(T=len(dyn.DET), x0=_x0(compiled, blocks))

    assert np.abs(_path(compiled, out) - dyn.DET).max() < TOL


def test_stochastic_path_matches_dynare(solved, blocks):
    compiled, solution = solved
    out = solution.sim(
        T=len(dyn.STOCH),
        shocks=_shocks(dyn.SHOCK_BLOCK),
        x0=_x0(compiled, blocks),
    )

    assert np.abs(_path(compiled, out) - dyn.STOCH).max() < TOL


@pytest.mark.parametrize("shock", dyn.EXO_COLUMNS)
def test_unit_innovation_response_matches_dynare(solved, shock):
    # From the steady state, one unit innovation on the first simulated period,
    # which is the experiment the golden ran.
    compiled, solution = solved
    horizon = len(dyn.IRF[shock])
    block = np.zeros((horizon, len(dyn.EXO_COLUMNS)))
    block[0, dyn.EXO_COLUMNS.index(shock)] = 1.0

    out = solution.sim(T=horizon, shocks=_shocks(block))

    assert np.abs(_path(compiled, out) - dyn.IRF[shock]).max() < TOL


# --- measurement and filter -------------------------------------------------


def test_measurement_matrices_match_dynare(solved):
    compiled, _ = solved
    _, C, d = _measurement(compiled)

    assert np.abs(C[:, _declared(compiled)] - dyn.KF_Z).max() < TOL
    # The generated block is unobserved, which is what makes the two filters
    # comparable despite the state vectors differing.
    assert np.abs(np.delete(C, _declared(compiled), axis=1)).max() < TOL
    assert np.abs(d - dyn.KF_D).max() < TOL


def _filter(compiled, solution, P0):
    Q, C, d = _measurement(compiled)
    return KalmanFilter.run(
        np.real(solution.A),
        np.real(solution.B),
        C,
        d,
        Q,
        dyn.KF_H,
        dyn.KF_DATA,
        x0=np.zeros(len(compiled.var_names)),
        P0=P0,
    )


def _unconditional_P0(solution, compiled):
    A, B = np.real(solution.A), np.real(solution.B)
    Q, _, _ = _measurement(compiled)
    P0 = solve_discrete_lyapunov(A, B @ Q @ B.T)
    return 0.5 * (P0 + P0.T)


def test_kalman_loglik_matches_dynare_from_the_unconditional_P0(solved):
    # Both filters start the model at its own stationary distribution, so this
    # comparison needs no coordinate change in either direction.
    compiled, solution = solved
    out = _filter(compiled, solution, _unconditional_P0(solution, compiled))

    assert float(out.loglik) == pytest.approx(dyn.LOGLIK_UNCOND, abs=1e-9)


def test_filtered_states_match_dynares_updated_variables(solved):
    # a_{t|t} on both sides, dated the same way, so nothing to align.
    compiled, solution = solved
    out = _filter(compiled, solution, _unconditional_P0(solution, compiled))

    x_filt = np.asarray(out.x_filt)[:, _declared(compiled)]
    assert np.abs(x_filt - dyn.UPDATED).max() < TOL


def test_predicted_states_match_dynares_filtered_variables_one_period_on(solved):
    # Dynare indexes a prediction by the information set behind it and we index
    # it by the period estimated, so the same quantity sits one row apart. The
    # overlap is every row but an endpoint each. Aligning the two without the
    # shift disagrees by 0.37 here, which the second assert holds the line on.
    compiled, solution = solved
    out = _filter(compiled, solution, _unconditional_P0(solution, compiled))

    x_pred = np.asarray(out.x_pred)[:, _declared(compiled)]
    assert np.abs(x_pred[1:] - dyn.FILTERED[: len(x_pred) - 1]).max() < TOL
    assert np.abs(x_pred - dyn.FILTERED[: len(x_pred)]).max() > 0.3


def test_prediction_is_the_transition_applied_to_the_previous_update(solved):
    # x_pred[t] = A x_filt[t-1]: the shocks are mean zero, so a prediction is the
    # update behind it carried forward. The second assert reads that same
    # identity off Dynare's array, where the index is the information set date,
    # so it needs no shift and reaches the row past the end of the sample.
    compiled, solution = solved
    out = _filter(compiled, solution, _unconditional_P0(solution, compiled))
    A = np.real(solution.A)

    x_filt = np.asarray(out.x_filt)
    x_pred = np.asarray(out.x_pred)
    assert np.abs(x_pred[1:] - x_filt[:-1] @ A.T).max() < TOL

    assert np.abs((x_filt @ A.T)[:, _declared(compiled)] - dyn.FILTERED).max() < TOL


def test_kalman_loglik_matches_dynare_from_a_unit_prior_on_dynares_states(
    solved, blocks
):
    # Dynare's P0 is the identity on five variables, but its transition reads
    # only three columns, so it is a unit prior on (g, z, r) at date 0. Ours
    # carries the same prior as the lag covariance that produces it through f.
    #
    # LOGLIK_FIXED_OURS, not LOGLIK_FIXED_DYNARE: the same P0 argument means the
    # covariance one period apart on the two sides, and only the unconditional
    # covariance is a fixed point of the map between them. Dynare's reading of
    # this same prior is 0.558 away.
    compiled, solution = solved
    ghx, _ = blocks
    lag = _lag_index(compiled)

    inverse = np.linalg.inv(ghx[: len(DYNARE_STATES)])
    P0 = np.zeros((len(compiled.var_names),) * 2)
    P0[np.ix_(lag, lag)] = inverse @ inverse.T

    out = _filter(compiled, solution, P0)

    assert float(out.loglik) == pytest.approx(dyn.LOGLIK_FIXED_OURS, abs=1e-9)
    assert abs(float(out.loglik) - dyn.LOGLIK_FIXED_DYNARE) > 0.5
