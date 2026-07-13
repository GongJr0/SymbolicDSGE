"""Direct coverage for the numba bicomplex primitive ops.

These `bc_*` helpers are normally only composed into the residual cfunc, so
coverage never sees their bodies. Calling them directly and checking against the
bicomplex algebra (value ``a + b*j``, ``j**2 = -1``) exercises each one.
"""

from __future__ import annotations

from SymbolicDSGE.core import bicomplex as B

X = (1.0 + 2.0j, 0.5 - 1.0j)
Y = (2.0 + 0.0j, 1.0 + 0.5j)


def _close(u, v, tol: float = 1e-9) -> bool:
    return abs(u[0] - v[0]) < tol and abs(u[1] - v[1]) < tol


def test_add_sub_neg_scale():
    assert _close(B.bc_add(X, Y), (X[0] + Y[0], X[1] + Y[1]))
    assert _close(B.bc_sub(X, Y), (X[0] - Y[0], X[1] - Y[1]))
    assert _close(B.bc_neg(X), (-X[0], -X[1]))
    assert _close(B.bc_real_scale(X, 3.0), (X[0] * 3.0, X[1] * 3.0))


def test_mul_matches_definition():
    expected = (X[0] * Y[0] - X[1] * Y[1], X[0] * Y[1] + X[1] * Y[0])
    assert _close(B.bc_mul(X, Y), expected)


def test_div_is_inverse_of_mul():
    q = B.bc_div(X, Y)
    # (X / Y) * Y == X
    assert _close(B.bc_mul(q, Y), X)


def test_exp_log_roundtrip_and_identities():
    assert _close(B.bc_exp((0.0 + 0.0j, 0.0 + 0.0j)), (1.0 + 0.0j, 0.0 + 0.0j))
    assert _close(B.bc_log((1.0 + 0.0j, 0.0 + 0.0j)), (0.0 + 0.0j, 0.0 + 0.0j))
    # exp(log(x)) == x
    assert _close(B.bc_exp(B.bc_log(X)), X, tol=1e-8)


def test_sqrt_squared_recovers_value():
    assert _close(B.bc_sqrt((4.0 + 0.0j, 0.0 + 0.0j)), (2.0 + 0.0j, 0.0 + 0.0j))
    r = B.bc_sqrt(X)
    assert _close(B.bc_mul(r, r), X, tol=1e-8)
