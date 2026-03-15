import numpy as np
from numpy import float64
from numpy.typing import NDArray
from numba import njit

_A = (
    3.3871328727963666080,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
)
A = np.array(_A, dtype=float64)

_B = (
    1.0,
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
)
B = np.array(_B, dtype=float64)

_C = (
    1.42343711074968357734,
    4.63033784615654529590,
    5.76949722146069140550,
    3.64784832476320460504,
    1.27045825245236838258,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
)
C = np.array(_C, dtype=float64)

_D = (
    1.0,
    2.05319162663775882187,
    1.67638483018380384940,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
)
D = np.array(_D, dtype=float64)

_E = (
    6.65790464350110377720,
    5.46378491116411436990,
    1.78482653991729133580,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
)
E = np.array(_E, dtype=float64)

_F = (
    1.0,
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
)
F = np.array(_F, dtype=float64)


@njit(cache=True)
def horner(coeffs: NDArray[float64], x: float64) -> float64:
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
        return float64(q * horner(A, r) / horner(B, r))

    r = p if q < 0.0 else (1.0 - p)
    r = float64(np.sqrt(-np.log(r)))

    if r <= 5.0:
        r -= 1.6
        x = float64(horner(C, r) / horner(D, r))
    else:
        r -= 5.0
        x = float64(horner(E, r) / horner(F, r))

    return float64(-x if q < 0.0 else x)


@njit(cache=True)
def erfinv_from_as241(y: float64) -> float64:
    return float64(ndtri_as241(float64(0.5 * (y + 1.0)) / np.sqrt(2.0)))
