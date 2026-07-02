"""numba bicomplex value-type ops for the second-order (bicomplex-step) residual.

A bicomplex is the 2-tuple ``(a, b)`` of ``complex128`` -- value ``a + b*j``,
``j^2 = -1`` -- where ``a`` carries the ``(1, i)`` components and ``b`` the
``(j, ij)`` components. These are the arithmetic substrate the residual @cfunc
runs on (numba has no native bicomplex), exactly mirroring the C
``sdsge_bicomplex.h``, which is their parity oracle. Register-resident (4 f64,
no heap), ``inline="always"`` so they fold into the residual.

Only the primitive set is here; the printer composes ``ipow``/``spow``/``cpow``/
half-integer powers from these (repeated ``bc_mul``, ``bc_exp(bc_log)``, ...).
Transcendentals use the idempotent projection, matching the C reconst sign.
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
    # Idempotent projection: p1 = a - i*b, p2 = a + i*b; f componentwise;
    # reconstruct a' = (w1+w2)/2, b' = i*(w1-w2)/2.
    iz2 = 1j * x[1]
    w1 = cmath.exp(x[0] - iz2)
    w2 = cmath.exp(x[0] + iz2)
    return ((w1 + w2) * 0.5, 1j * (w1 - w2) * 0.5)


@njit(**_INLINE)
def bc_log(x):  # type: ignore[no-untyped-def]
    iz2 = 1j * x[1]
    w1 = cmath.log(x[0] - iz2)
    w2 = cmath.log(x[0] + iz2)
    return ((w1 + w2) * 0.5, 1j * (w1 - w2) * 0.5)


@njit(**_INLINE)
def bc_sqrt(x):  # type: ignore[no-untyped-def]
    # Direct in-slot solve of w^2 = x (cancellation-free, unlike idempotent).
    s = cmath.sqrt(x[0] * x[0] + x[1] * x[1])
    w1 = cmath.sqrt((x[0] + s) * 0.5)
    return (w1, x[1] / (2.0 * w1))
