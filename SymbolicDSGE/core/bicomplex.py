"""numba bicomplex value-type ops for the second-order (bicomplex-step) residual.

A bicomplex is the 2-tuple ``(a, b)`` of ``complex128`` -- value ``a + b*j``,
``j^2 = -1`` -- where ``a`` carries the ``(1, i)`` components and ``b`` the
``(j, ij)`` components. These are the arithmetic substrate the residual @cfunc
runs on (numba has no native bicomplex), exactly mirroring the C
``sdsge_bicomplex.h``, which is their parity oracle. Register-resident (4 f64,
no heap), ``inline="always"`` so they fold into the residual.

Only the primitive set is here; the printer composes ``ipow``/``spow``/``cpow``/
half-integer powers from these (repeated ``bc_mul``, ``bc_exp(bc_log)``, ...).

Transcendentals use the polar form, not the idempotent projection. The
projection evaluates ``f(a - i*b)`` and ``f(a + i*b)`` and reconstructs from
their half-difference, and those two agree to O(step), so the ``ij`` slot the
Hessian reads is what the subtraction cancels away. That is what put a
``1/step^2`` roundoff floor under the second-order tensors, on top of the
O(step^2) truncation the bicomplex step carries inherently.
"""

from __future__ import annotations

import cmath

from numba import njit

_INLINE = {"inline": "always"}


@njit(**_INLINE)
def bc_add(x, y):  # type: ignore[no-untyped-def]
    return (x[0] + y[0], x[1] + y[1])


@njit(**_INLINE)
def bc_sub(x, y):  # type: ignore[no-untyped-def]
    return (x[0] - y[0], x[1] - y[1])


@njit(**_INLINE)
def bc_neg(x):  # type: ignore[no-untyped-def]
    return (-x[0], -x[1])


@njit(**_INLINE)
def bc_mul(x, y):  # type: ignore[no-untyped-def]
    return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0])


@njit(**_INLINE)
def bc_div(x, y):  # type: ignore[no-untyped-def]
    denom = y[0] * y[0] + y[1] * y[1]
    return (
        (x[0] * y[0] + x[1] * y[1]) / denom,
        (x[1] * y[0] - x[0] * y[1]) / denom,
    )


@njit(**_INLINE)
def bc_real_scale(x, s):  # type: ignore[no-untyped-def]
    return (x[0] * s, x[1] * s)


@njit(**_INLINE)
def bc_exp(x):  # type: ignore[no-untyped-def]
    # exp(a + b*j) = exp(a) * (cos b + j sin b), the j-analogue of the complex
    # polar form. b is O(step), so both slots are products, never differences.
    e = cmath.exp(x[0])
    return (e * cmath.cos(x[1]), e * cmath.sin(x[1]))


@njit(**_INLINE)
def bc_log(x):  # type: ignore[no-untyped-def]
    # Same polar form inverted: log(a + b*j) = log(a) + log(1 + q^2)/2 + j*atan(q)
    # for q = b/a. Requires a real-dominant a, as bc_sqrt does.
    q = x[1] / x[0]
    return (cmath.log(x[0]) + 0.5 * cmath.log(1.0 + q * q), cmath.atan(q))


@njit(**_INLINE)
def bc_sqrt(x):  # type: ignore[no-untyped-def]
    # Direct in-slot solve of w^2 = x (cancellation-free, unlike idempotent).
    s = cmath.sqrt(x[0] * x[0] + x[1] * x[1])
    w1 = cmath.sqrt((x[0] + s) * 0.5)
    return (w1, x[1] / (2.0 * w1))
