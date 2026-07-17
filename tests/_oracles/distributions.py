"""Numba reference implementations for the native ``_ckernels.distributions``
kernels, retained as parity oracles.

These mirror the njit kernels that used to live in
``SymbolicDSGE.bayesian.distributions``; the native C is the production path and
these independent implementations exist only so the parity tests can pin it. The
AS 241 coefficients are duplicated here on purpose: an oracle that shared the
production constants would not be an independent check.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy import float64
from numpy.typing import NDArray

NDF = NDArray[float64]

# --- distributions._as241 (Wichura inverse-normal) ---------------------------
_A = np.array(
    (
        3.3871328727963666080,
        1.3314166789178437745e2,
        1.9715909503065514427e3,
        1.3731693765509461125e4,
        4.5921953931549871457e4,
        6.7265770927008700853e4,
        3.3430575583588128105e4,
        2.5090809287301226727e3,
    ),
    dtype=float64,
)
_B = np.array(
    (
        1.0,
        4.2313330701600911252e1,
        6.8718700749205790830e2,
        5.3941960214247511077e3,
        2.1213794301586595867e4,
        3.9307895800092710610e4,
        2.8729085735721942674e4,
        5.2264952788528545610e3,
    ),
    dtype=float64,
)
_C = np.array(
    (
        1.42343711074968357734,
        4.63033784615654529590,
        5.76949722146069140550,
        3.64784832476320460504,
        1.27045825245236838258,
        2.41780725177450611770e-1,
        2.27238449892691845833e-2,
        7.74545014278341407640e-4,
    ),
    dtype=float64,
)
_D = np.array(
    (
        1.0,
        2.05319162663775882187,
        1.67638483018380384940,
        6.89767334985100004550e-1,
        1.48103976427480074590e-1,
        1.51986665636164571966e-2,
        5.47593808499534494600e-4,
        1.05075007164441684324e-9,
    ),
    dtype=float64,
)
_E = np.array(
    (
        6.65790464350110377720,
        5.46378491116411436990,
        1.78482653991729133580,
        2.96560571828504891230e-1,
        2.65321895265761230930e-2,
        1.24266094738807843860e-3,
        2.71155556874348757815e-5,
        2.01033439929228813265e-7,
    ),
    dtype=float64,
)
_F = np.array(
    (
        1.0,
        5.99832206555887937690e-1,
        1.36929880922735805310e-1,
        1.48753612908506148525e-2,
        7.86869131145613259100e-4,
        1.84631831751005468180e-5,
        1.42151175831644588870e-7,
        2.04426310338993978564e-15,
    ),
    dtype=float64,
)


@njit(cache=True)
def _horner(coeffs: NDF, x: float64) -> float64:
    y = float64(0.0)
    for c in coeffs[::-1]:
        y = y * x + c
    return y


@njit(cache=True)
def ndtri_as241(p: float64) -> float64:
    if p <= 0.0:
        return float64(-np.inf)
    if p >= 1.0:
        return float64(np.inf)

    q = float64(p - 0.5)

    if np.abs(q) <= 0.425:
        r = float64(0.180625 - q * q)
        return float64(q * _horner(_A, r) / _horner(_B, r))

    r = p if q < 0.0 else (1.0 - p)
    r = float64(np.sqrt(-np.log(r)))

    if r <= 5.0:
        r -= 1.6
        x = float64(_horner(_C, r) / _horner(_D, r))
    else:
        r -= 5.0
        x = float64(_horner(_E, r) / _horner(_F, r))

    return float64(-x if q < 0.0 else x)


@njit(cache=True)
def erfinv_from_as241(y: float64) -> float64:
    return float64(ndtri_as241(float64(0.5 * (y + 1.0)) / np.sqrt(2.0)))
