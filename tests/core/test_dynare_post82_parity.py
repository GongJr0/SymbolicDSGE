"""Exact parity against Dynare's first-order solution of POST82.

``tests/fixtures/models/POST82.yaml`` and
``tests/fixtures/models/post82_first_order.mod`` are the same model written
twice, both at natural dating. Goldens are transcribed in
``tests/_oracles/dynare_post82_first_order``, raw: nothing was re-dated,
rescaled or reordered on the way out of Dynare.

Both sides carry the same variables in the same order: Dynare's state is the
model's own lagged variables and so is ours, so the two decision rules are
compared outright, with no selection and no transform. ``ghx`` is ``[p; f]``
stacked, ``ghu`` is ``B``, and ``A`` is Dynare's ``A``.

One quantity still does not carry, and it is a filter convention rather than a
coordinate one. Our filter predicts before its first update, so a ``P0`` argument
is the covariance one period behind the same argument to Dynare's. The
unconditional covariance is a fixed point of that map, which is why it is parity
outright while a unit prior is not.

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
    """Our decision rule: the state rows stacked over the controls, and the impact."""
    _, solution = solved
    ghx = np.vstack([np.real(solution.policy.p), np.real(solution.policy.f)])
    return ghx, np.real(solution.policy.B)


def _declared(compiled) -> list[int]:
    return [compiled.idx[v] for v in dyn.DECL_COLUMNS]


def _x0() -> dict[str, float]:
    """Dynare's ``y0``, which is our ``x0`` outright now that the states agree."""
    return dict(zip(DYNARE_STATES, dyn.X0_STATES))


def _path(compiled, out) -> np.ndarray:
    return np.column_stack([out.states[name] for name in dyn.DECL_COLUMNS])


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


def test_compiled_order_is_dynares_declaration_order(solved):
    compiled, _ = solved
    assert compiled.layout.canonical_names == dyn.DECL_COLUMNS
    assert compiled.layout.state_names == DYNARE_STATES


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
    B = np.real(solution.policy.B)

    assert np.abs(B[_declared(compiled)] - dyn.GHU_DECL).max() < TOL
    r_row = B[compiled.idx["r"]]
    assert r_row[dyn.EXO_COLUMNS.index("e_r")] == pytest.approx(0.33355298, abs=1e-8)
    assert r_row[dyn.EXO_COLUMNS.index("e_g")] == pytest.approx(0.34702357, abs=1e-8)


def test_shock_covariance_matches_dynare(solved):
    compiled, _ = solved
    Q, _, _ = _measurement(compiled)

    assert np.abs(Q - dyn.SIGMA_E).max() < TOL


# --- the transition ---------------------------------------------------------
# The simulations below run A and B with no f in the loop, so they are what
# exercises A. These state what A holds, which a path only implies.


def test_transition_matches_dynares_A(solved):
    # A maps y_{t-1} to y_t, so its state columns are the decision rule itself
    # and a control at t-1 reaches nothing.
    compiled, solution = solved
    A = np.real(solution.policy.A)
    n_state = compiled.layout.n_state

    assert np.abs(A - dyn.A_DECL).max() < TOL
    assert np.abs(A[:, :n_state] - dyn.GHX_DECL).max() < TOL
    assert np.abs(A[:, n_state:]).max() < TOL


# --- simulation -------------------------------------------------------------


def test_deterministic_path_matches_dynare(solved):
    # Driven by the transition alone, so it also pins the row convention: row 0
    # is the first simulated period, not the initial condition.
    compiled, solution = solved
    out = solution.sim(T=len(dyn.DET), x0=_x0())

    assert np.abs(_path(compiled, out) - dyn.DET).max() < TOL


def test_stochastic_path_matches_dynare(solved):
    compiled, solution = solved
    out = solution.sim(
        T=len(dyn.STOCH),
        shocks=_shocks(dyn.SHOCK_BLOCK),
        x0=_x0(),
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
    assert np.abs(d - dyn.KF_D).max() < TOL


def _filter(compiled, solution, P0):
    Q, C, d = _measurement(compiled)
    return KalmanFilter.run(
        np.real(solution.policy.A),
        np.real(solution.policy.B),
        C,
        d,
        Q,
        dyn.KF_H,
        dyn.KF_DATA,
        x0=np.zeros((compiled.n_var,), dtype=np.float64),
        P0=P0,
    )


def _unconditional_P0(solution, compiled):
    A, B = np.real(solution.policy.A), np.real(solution.policy.B)
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
    A = np.real(solution.policy.A)

    x_filt = np.asarray(out.x_filt)
    x_pred = np.asarray(out.x_pred)
    assert np.abs(x_pred[1:] - x_filt[:-1] @ A.T).max() < TOL

    assert np.abs((x_filt @ A.T)[:, _declared(compiled)] - dyn.FILTERED).max() < TOL


def test_kalman_loglik_matches_dynare_from_a_unit_prior_on_dynares_states(solved):
    # Dynare's P0 is the identity on five variables, which is the same matrix in
    # our coordinates now, so nothing is carried across.
    #
    # An arbitrary prior, not the unconditional one: both loops now open on the
    # update, so P0 is the covariance of the same state on both sides and any
    # prior agrees. Under a predict-first loop only the unconditional covariance
    # did, being the one fixed point of the prediction map.
    compiled, solution = solved

    out = _filter(compiled, solution, np.eye(len(compiled.var_names)))

    assert float(out.loglik) == pytest.approx(dyn.LOGLIK_FIXED_DYNARE, abs=1e-9)
