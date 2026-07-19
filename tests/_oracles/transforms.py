"""Independent reference implementations for the native ``_ckernels.transforms``
kernels, retained as parity oracles.

These recreate the numpy/scipy math the pure-Python ``LogTransform`` /
``LogitTransform`` / ``ProbitTransform`` carried before they were rewired to call
the C kernels. The native C is the production path; these exist only so the
parity tests can pin it.

``probit`` deliberately uses scipy's ``norm`` (a different inverse-normal
implementation than the AS 241 C the production kernel calls), so the oracle is a
genuine independent check rather than a mirror of production. Each function
accepts a scalar or an array and returns numpy, matching the kernel contract.
"""

from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.stats import norm

NDF = NDArray[float64]


def _sigmoid(y: object) -> NDF:
    return 1.0 / (1.0 + np.exp(-np.asarray(y, dtype=float64)))


# --- log --------------------------------------------------------------------
def log_fwd(x):
    return np.log(x)


def log_inv(y):
    return np.exp(y)


def log_grad_fwd(x):
    return 1.0 / np.asarray(x, dtype=float64)


def log_grad_inv(y):
    return np.exp(y)


def log_ldet_abs_jac_fwd(x):
    return -np.log(x)


def log_ldet_abs_jac_inv(y):
    return np.asarray(y, dtype=float64)


def log_grad_ldet_abs_jac_inv(y):
    return np.ones_like(np.asarray(y, dtype=float64))


# --- logit ------------------------------------------------------------------
def logit_fwd(x):
    x = np.asarray(x, dtype=float64)
    return np.log(x / (1.0 - x))


def logit_inv(y):
    return _sigmoid(y)


def logit_grad_fwd(x):
    x = np.asarray(x, dtype=float64)
    return 1.0 / (x * (1.0 - x))


def logit_grad_inv(y):
    p = _sigmoid(y)
    return p * (1.0 - p)


def logit_ldet_abs_jac_fwd(x):
    x = np.asarray(x, dtype=float64)
    return -np.log(x) - np.log(1.0 - x)


def logit_ldet_abs_jac_inv(y):
    y = np.asarray(y, dtype=float64)
    return -y - 2.0 * np.log(1.0 + np.exp(-y))


def logit_grad_ldet_abs_jac_inv(y):
    return 1.0 - 2.0 * _sigmoid(y)


# --- probit (scipy norm: independent of the AS 241 C) -----------------------
def probit_fwd(x):
    return norm.ppf(x)


def probit_inv(y):
    return norm.cdf(y)


def probit_grad_fwd(x):
    return 1.0 / norm.pdf(norm.ppf(x))


def probit_grad_inv(y):
    return norm.pdf(y)


def probit_ldet_abs_jac_fwd(x):
    return -np.log(norm.pdf(norm.ppf(x)))


def probit_ldet_abs_jac_inv(y):
    return np.log(norm.pdf(y))


def probit_grad_ldet_abs_jac_inv(y):
    return -np.asarray(y, dtype=float64)


# --- affine logit ((low, high) -> R via a unit-logit) -----------------------
def aff_logit_fwd(x, low, high):
    z = (np.asarray(x, dtype=float64) - low) / (high - low)
    return np.log(z / (1.0 - z))


def aff_logit_inv(y, low, high):
    return _sigmoid(y) * (high - low) + low


def aff_logit_grad_fwd(x, low, high):
    span = high - low
    z = (np.asarray(x, dtype=float64) - low) / span
    return 1.0 / (span * z * (1.0 - z))


def aff_logit_grad_inv(y, low, high):
    p = _sigmoid(y)
    return (high - low) * p * (1.0 - p)


def aff_logit_ldet_abs_jac_fwd(x, low, high):
    span = high - low
    z = (np.asarray(x, dtype=float64) - low) / span
    return -np.log(span) - np.log(z) - np.log(1.0 - z)


def aff_logit_ldet_abs_jac_inv(y, low, high):
    y = np.asarray(y, dtype=float64)
    return np.log(high - low) - y - 2.0 * np.log(1.0 + np.exp(-y))


def aff_logit_grad_ldet_abs_jac_inv(y):
    return 1.0 - 2.0 * _sigmoid(y)


# --- affine probit ((low, high) -> R via a unit-probit) ---------------------
def aff_probit_fwd(x, low, high):
    z = (np.asarray(x, dtype=float64) - low) / (high - low)
    return norm.ppf(z)


def aff_probit_inv(y, low, high):
    return norm.cdf(y) * (high - low) + low


def aff_probit_grad_fwd(x, low, high):
    span = high - low
    z = (np.asarray(x, dtype=float64) - low) / span
    return 1.0 / (span * norm.pdf(norm.ppf(z)))


def aff_probit_grad_inv(y, low, high):
    return (high - low) * norm.pdf(y)


def aff_probit_ldet_abs_jac_fwd(x, low, high):
    span = high - low
    z = (np.asarray(x, dtype=float64) - low) / span
    return -np.log(span) - np.log(norm.pdf(norm.ppf(z)))


def aff_probit_ldet_abs_jac_inv(y, low, high):
    return np.log(high - low) + np.log(norm.pdf(y))


def aff_probit_grad_ldet_abs_jac_inv(y):
    return -np.asarray(y, dtype=float64)


# --- softplus ((0, inf) <-> R) ----------------------------------------------
def softplus_fwd(x):
    return np.log(np.expm1(x))


def softplus_inv(y):
    return np.logaddexp(float64(0.0), np.asarray(y, dtype=float64))


def softplus_grad_fwd(x):
    return 1.0 + 1.0 / np.expm1(np.asarray(x, dtype=float64))


def softplus_grad_inv(y):
    return _sigmoid(y)


def softplus_ldet_abs_jac_fwd(x):
    x = np.asarray(x, dtype=float64)
    return x - np.log(np.expm1(x))


def softplus_ldet_abs_jac_inv(y):
    return -np.logaddexp(float64(0.0), -np.asarray(y, dtype=float64))


def softplus_grad_ldet_abs_jac_inv(y):
    return 1.0 - _sigmoid(y)


# --- lower bounded ((low, inf) <-> R) ---------------------------------------
def lower_fwd(x, low):
    return np.log(np.asarray(x, dtype=float64) - low)


def lower_inv(y, low):
    return low + np.exp(y)


def lower_grad_fwd(x, low):
    return 1.0 / (np.asarray(x, dtype=float64) - low)


def lower_grad_inv(y):
    return np.exp(y)


def lower_ldet_abs_jac_fwd(x, low):
    return -np.log(np.asarray(x, dtype=float64) - low)


def lower_ldet_abs_jac_inv(y):
    return np.asarray(y, dtype=float64)


def lower_grad_ldet_abs_jac_inv(y):
    return np.ones_like(np.asarray(y, dtype=float64))


# --- upper bounded ((-inf, high) <-> R) -------------------------------------
def upper_fwd(x, high):
    return np.log(high - np.asarray(x, dtype=float64))


def upper_inv(y, high):
    return high - np.exp(y)


def upper_grad_fwd(x, high):
    return -1.0 / (high - np.asarray(x, dtype=float64))


def upper_grad_inv(y):
    return -np.exp(y)


def upper_ldet_abs_jac_fwd(x, high):
    return -np.log(high - np.asarray(x, dtype=float64))


def upper_ldet_abs_jac_inv(y):
    return np.asarray(y, dtype=float64)


def upper_grad_ldet_abs_jac_inv(y):
    return np.ones_like(np.asarray(y, dtype=float64))


# --- tanh ((-1, 1) <-> R) ---------------------------------------------------
# grad_inv / ldj_inv use 1/cosh^2 and -2*log(cosh) rather than 1 - tanh^2 to
# avoid the cancellation as tanh -> +-1 (matches the kernel's stable forms).
def tanh_fwd(x):
    return np.arctanh(x)


def tanh_inv(y):
    return np.tanh(y)


def tanh_grad_fwd(x):
    x = np.asarray(x, dtype=float64)
    return 1.0 / ((1.0 - x) * (1.0 + x))


def tanh_grad_inv(y):
    return 1.0 / np.cosh(np.asarray(y, dtype=float64)) ** 2


def tanh_ldet_abs_jac_fwd(x):
    x = np.asarray(x, dtype=float64)
    return -np.log((1.0 - x) * (1.0 + x))


def tanh_ldet_abs_jac_inv(y):
    return -2.0 * np.log(np.cosh(np.asarray(y, dtype=float64)))


def tanh_grad_ldet_abs_jac_inv(y):
    return -2.0 * np.tanh(y)
