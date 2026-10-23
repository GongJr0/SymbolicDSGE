"""The two paths to a first order solution, solved and compared.

Path A solves the authored levels model, so the solver linearizes numerically
at the steady state it resolves. Path B hands ``PointLinearView.to_config()``
back to the solver, which finds a model already linear about the origin. Both
describe the same economy at first order, so the policy has to come out the
same, and the measurement equation has to keep mapping to the same observed
levels.

These solve, which the rest of ``tests/linearization`` does not. The expansion
math is pinned in ``test_expansion.py`` and the view's own surface in
``test_linearizer.py``; what is pinned here is that the two pipelines agree,
and that a constrained model has no second path to agree on.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import numpy as np
import pytest
import sympy as sp

from SymbolicDSGE import DSGESolver, ModelParser
from SymbolicDSGE.linearization import Linearizer

#: A single impulse, then nothing. One shock is enough to drive every state,
#: and a fixed path keeps both simulations on identical innovations.
SHOCK_PATH = np.array([0.01, 0.0, 0.0, 0.0])

#: The round trip rearranges each equation before the Jacobian is taken, so the
#: two pencils agree to rounding rather than bitwise. Coefficients here are
#: order one.
TOL = 1e-10

#: A minted lag is the one shape no fixture model carries. ``u(t-2)`` mints
#: ``u_lag1``, which the round trip cannot hand across: the linear config
#: declares ``u`` alone, and the second compile has to mint the aux again off
#: the linearized equation.
AUX_MODEL = """
name: AUX_LEVELS
variables: [u]
ss_seed:
  u: ubar
shocks: [e]
equations:
  model:
    u_eq: "u(t) = (1 - rho)*ubar + rho*u(t-2) + e"
  observables:
    u_obs: "u(t)"
calibration:
  parameters:
    rho: 0.5
    ubar: 2.0
    sig: 0.1
  shocks:
    std:
      e: sig
    corr: {}
"""


class _RoundTrip(NamedTuple):
    direct: object
    linear: object
    point: Mapping[str, float]


def _round_trip(
    parser: ModelParser, methods: Mapping[str, str] | None = None
) -> _RoundTrip:
    model, kalman = parser.get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    view = Linearizer(compiled, methods).at()
    linear_solver = DSGESolver(view.to_config(), kalman)

    return _RoundTrip(
        direct=solver.solve(compiled),
        linear=linear_solver.solve(linear_solver.compile()),
        point={str(name): float(value) for name, value in view.steady_state.items()},
    )


@pytest.fixture(scope="module")
def rbc(rbc_second_order_test_model_path) -> _RoundTrip:
    """Levels RBC, every variable left in levels."""
    return _round_trip(ModelParser(rbc_second_order_test_model_path))


@pytest.fixture(scope="module")
def rbc_log(rbc_second_order_test_model_path) -> _RoundTrip:
    """The same model, with the two positive variables in logs."""
    return _round_trip(
        ModelParser(rbc_second_order_test_model_path), {"c": "log", "k": "log"}
    )


@pytest.fixture(scope="module")
def gap(test_model_path) -> _RoundTrip:
    """A model already authored in gaps, so the round trip changes nothing."""
    return _round_trip(ModelParser(test_model_path))


@pytest.fixture(scope="module")
def aux() -> _RoundTrip:
    return _round_trip(ModelParser.from_string(AUX_MODEL), {"u": "log"})


# ---------------- the policy ----------------


def test_the_round_trip_keeps_the_variable_order(rbc):
    # Everything below compares matrices position by position, which only means
    # anything while both compiles agree on what each position holds.
    assert rbc.direct.compiled.var_names == rbc.linear.compiled.var_names


def test_a_levels_model_round_trips_to_the_same_policy(rbc):
    np.testing.assert_allclose(
        rbc.linear.policy.A, rbc.direct.policy.A, rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        rbc.linear.policy.B, rbc.direct.policy.B, rtol=0, atol=TOL
    )


def test_a_gap_model_round_trips_to_the_same_policy(gap):
    np.testing.assert_allclose(
        gap.linear.policy.A, gap.direct.policy.A, rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        gap.linear.policy.B, gap.direct.policy.B, rtol=0, atol=TOL
    )


def test_a_log_round_trip_rescales_the_policy_by_the_point(rbc_log):
    # A log deviation is a level deviation over the steady state, so the two
    # policies are the same map in different units: A_log = D^-1 A_level D for
    # D the diagonal of the expansion point. z stays in levels and takes 1,
    # which is also the only scale available to it at a zero steady state.
    scale = np.array(
        [
            rbc_log.point[name] if name in ("c", "k") else 1.0
            for name in rbc_log.direct.compiled.var_names
        ]
    )
    D = np.diag(scale)
    D_inv = np.diag(1.0 / scale)

    np.testing.assert_allclose(
        rbc_log.linear.policy.A, D_inv @ rbc_log.direct.policy.A @ D, rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        rbc_log.linear.policy.B, D_inv @ rbc_log.direct.policy.B, rtol=0, atol=TOL
    )


# ---------------- the expansion point ----------------


def test_the_linear_config_solves_to_the_origin(rbc):
    # The deviation system clears at zero. Path A's own point is the levels
    # steady state, which is what the deviations are measured from.
    np.testing.assert_allclose(
        rbc.linear.policy.steady_state, np.zeros(3), rtol=0, atol=1e-12
    )
    assert rbc.direct.policy.steady_state[0] == pytest.approx(rbc.point["k"], rel=1e-9)
    assert rbc.direct.policy.steady_state[2] == pytest.approx(rbc.point["c"], rel=1e-9)


# ---------------- the measurement equation ----------------


def test_both_paths_simulate_the_same_observed_levels(rbc):
    # The intercept is what makes this hold. Observables come out of the round
    # trip affine rather than in deviations, so a measurement equation written
    # against observed levels still reads in levels after linearization. The
    # state paths are the same economy measured from different origins.
    direct = rbc.direct.sim(4, shocks={"e": SHOCK_PATH}, observables=True)
    linear = rbc.linear.sim(4, shocks={"e": SHOCK_PATH}, observables=True)

    np.testing.assert_allclose(
        linear.observables["c_obs"], direct.observables["c_obs"], rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        linear.states["k"], direct.states["k"] - rbc.point["k"], rtol=0, atol=TOL
    )


# ---------------- a minted lag ----------------


def test_a_minted_lag_is_minted_again_from_the_linear_config(aux):
    # The linear config declares u alone, so the aux cannot be carried across.
    # Both compiles landing on the same layout is what makes the policies
    # comparable at all.
    assert aux.direct.compiled.var_names == ["u", "u_lag1"]
    assert aux.linear.compiled.var_names == ["u", "u_lag1"]
    assert aux.linear.compiled.layout.aux_origin == {"u_lag1": "u"}
    assert set(aux.linear.config.ss_seed) == {sp.Function("u")}


def test_an_aux_round_trips_on_its_origins_scale(aux):
    # u_lag1 is u one date back, so it expands about the same point and takes
    # the same log scale. A bare 1.0 here would mean the aux had been left in
    # levels while its origin went to logs, and the identity tying them would
    # no longer hold.
    scale = np.array(
        [
            aux.point[aux.direct.compiled.layout.aux_origin.get(name, name)]
            for name in aux.direct.compiled.var_names
        ]
    )

    assert scale.tolist() == [2.0, 2.0]
    np.testing.assert_allclose(
        aux.linear.policy.A,
        np.diag(1.0 / scale) @ aux.direct.policy.A @ np.diag(scale),
        rtol=0,
        atol=TOL,
    )


def test_an_aux_model_simulates_the_same_observed_levels(aux):
    direct = aux.direct.sim(4, shocks={"e": SHOCK_PATH}, observables=True)
    linear = aux.linear.sim(4, shocks={"e": SHOCK_PATH}, observables=True)

    np.testing.assert_allclose(
        linear.observables["u_obs"], direct.observables["u_obs"], rtol=0, atol=TOL
    )


# ---------------- models with no second path ----------------


def test_a_constrained_model_refuses_to_become_a_config(rbc_occbin_test_model_path):
    # The regimes and the constraint conditions stay in nonlinear levels through
    # to_config's copy, so the config would pair deviation residuals with level
    # thresholds. The view itself is a sound diagnostic, which is why the guard
    # sits on the conversion rather than on at().
    model, kalman = ModelParser(rbc_occbin_test_model_path).get_all()
    compiled = DSGESolver(model, kalman).compile()

    view = Linearizer(compiled).at()
    assert view.model
    assert compiled.constraint_names == ("IRR", "INEG")

    with pytest.raises(
        NotImplementedError, match="constrained models are not convertible"
    ):
        view.to_config()
