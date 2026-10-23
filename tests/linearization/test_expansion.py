"""Expansion math, driven through the helpers at a hand-chosen point.

``Linearizer.at`` solves its expansion point with Newton, so every coefficient it
emits is a float that came out of a compiled model. These tests supply the point
directly instead, which is what keeps the expected values exact and the
assertions readable. ``test_linearizer.py`` covers the same math through
``.at()``, and ``test_log_expansion_agrees_with_the_helper`` there ties the two
together so these cannot keep passing if ``at`` stops routing through them.

Every point used with an equation below solves that equation. A single
expression has no such constraint, but an equation does: ``.at()`` only ever
hands the helpers a point Newton found, so a point that leaves the two sides
unequal is a state the library cannot reach.
"""

from __future__ import annotations

import pytest
import sympy as sp

from SymbolicDSGE.linearization.linearizer import (
    LinearizationMethod,
    _collect_calls,
    _expand,
    _linearize_equation,
)

T = sp.Symbol("t", integer=True)
C = sp.Function("c")
K = sp.Function("k")
E = sp.Symbol("e")
SHOCKS = frozenset({E})

LOG = LinearizationMethod.LOG
LEVEL = LinearizationMethod.LEVEL

#: c log-linearized, k in levels.
MIXED = {"c": LOG, "k": LEVEL}
LEVELS = {"c": LEVEL, "k": LEVEL}

#: For the single-expression cases, where nothing has to balance.
POINT = {"c": 2.0, "k": 5.0}


def _residual(equation: sp.Eq) -> sp.Expr:
    return equation.lhs - equation.rhs


def test_log_slope_is_the_elasticity_at_the_point():
    # d/dc_hat of (cbar*exp(c_hat))**0.5 at zero is 0.5*cbar**0.5.
    expansion = _expand(C(T) ** 0.5, MIXED, POINT, SHOCKS)

    assert float(expansion.constant) == pytest.approx(2.0**0.5)
    assert float(expansion.affine.coeff(C(T))) == pytest.approx(0.5 * 2.0**0.5)


@pytest.mark.parametrize(
    "kbar, slope, constant",
    [(5.0, 10.0, 25.0), (0.0, 0.0, 0.0)],
)
def test_level_slope_is_the_derivative_at_the_point(kbar, slope, constant):
    # A quadratic term carries its coefficient at every point but the origin,
    # where 2*kbar is itself zero.
    expansion = _expand(K(T) ** 2, {"k": LEVEL}, {"k": kbar}, SHOCKS)

    assert float(expansion.constant) == pytest.approx(constant)
    assert float(expansion.affine.coeff(K(T))) == pytest.approx(slope)


def test_the_value_at_the_expansion_point_is_subtracted_from_each_side():
    # cbar = 20.25 is what solves this equation at kbar = 5.0: both sides are
    # worth 4.5 there, and subtracting each side's own value is equation
    # preserving precisely because the two values agree.
    point = {"c": 20.25, "k": 5.0}
    equation = sp.Eq(C(T) ** 0.5, 0.9 * K(T - 1) + E)

    assert float(_expand(equation.lhs, MIXED, point, SHOCKS).constant) == pytest.approx(
        4.5
    )
    assert float(_expand(equation.rhs, MIXED, point, SHOCKS).constant) == pytest.approx(
        4.5
    )

    linear = _linearize_equation(equation, MIXED, point, SHOCKS)

    assert float(linear.lhs.coeff(C(T))) == pytest.approx(2.25)
    assert linear.rhs == E + 0.9 * K(T - 1)
    # Nothing is left over: the deviation system clears at the origin.
    assert _residual(linear).xreplace({C(T): 0, K(T - 1): 0, E: 0}) == 0


def test_a_whole_constant_comes_off_not_a_designated_intercept():
    # Both sides are worth 16.0 at this point, one as 2*ybar and the other as
    # kbar plus a literal. Neither survives.
    equation = sp.Eq(2 * C(T), K(T - 1) + 11.0)

    linear = _linearize_equation(equation, LEVELS, {"c": 8.0, "k": 5.0}, SHOCKS)

    assert linear == sp.Eq(2 * C(T), K(T - 1))


def test_a_second_pass_changes_nothing():
    # The output is a deviation system, so expanding it again in levels about the
    # origin is the identity. A to_config() round trip rests on this.
    point = {"c": 20.25, "k": 5.0}
    once = _linearize_equation(
        sp.Eq(C(T) ** 0.5, 0.9 * K(T - 1) + E), MIXED, point, SHOCKS
    )
    twice = _linearize_equation(once, LEVELS, {"c": 0.0, "k": 0.0}, SHOCKS)

    assert sp.simplify(_residual(once) - _residual(twice)) == 0


def test_a_shock_survives_as_a_linear_term():
    linear = _linearize_equation(
        sp.Eq(C(T), E + K(T - 1)), MIXED, {"c": 5.0, "k": 5.0}, SHOCKS
    )

    assert float(linear.rhs.coeff(E)) == pytest.approx(1.0)


def test_a_nonlinear_shock_is_linearized_too():
    # exp(e)*k becomes kbar*e + k_hat: first order in the innovation as well as
    # in the state.
    linear = _linearize_equation(
        sp.Eq(C(T), sp.exp(E) * K(T - 1)), MIXED, {"c": 5.0, "k": 5.0}, SHOCKS
    )

    assert float(linear.rhs.coeff(E)) == pytest.approx(5.0)
    assert float(linear.rhs.coeff(K(T - 1))) == pytest.approx(1.0)


def test_cross_terms_between_a_shock_and_a_variable_are_dropped():
    # e*k is second order at the point, so kbar*e is the whole of it and no
    # e*k_hat term survives.
    linear = _linearize_equation(
        sp.Eq(C(T), E * K(T - 1)), LEVELS, {"c": 0.0, "k": 5.0}, SHOCKS
    )

    assert linear == sp.Eq(C(T), 5.0 * E)


def test_calls_are_collected_in_a_deterministic_order():
    # atoms() returns a set, so the emitted sum would otherwise follow hash
    # order. The key is (name, srepr), which puts a dated call before a bare one.
    calls = _collect_calls(C(T) + C(T + 1) + K(T - 1) + K(T), MIXED)

    assert [str(call) for call in calls] == ["c(t + 1)", "c(t)", "k(t - 1)", "k(t)"]


def test_an_observable_keeps_its_value_at_the_point():
    # An equation subtracts the constant per side; an observable does not,
    # because its intercept is what maps a deviation back to an observed level.
    expansion = _expand(C(T) + K(T), MIXED, POINT, SHOCKS)

    assert float(expansion.constant) == pytest.approx(7.0)
    assert float(expansion.affine.coeff(C(T))) == pytest.approx(2.0)
    assert float(expansion.affine.coeff(K(T))) == pytest.approx(1.0)
