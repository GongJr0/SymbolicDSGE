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
