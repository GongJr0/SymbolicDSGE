from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from numba import njit
from numpy import float64
from numpy.typing import NDArray

from ..bayesian.distributions.beta import Beta
from ..bayesian.distributions.gamma import Gamma
from ..bayesian.distributions.half_cauchy import HalfCauchy
from ..bayesian.distributions.half_norm import HalfNormal
from ..bayesian.distributions.inv_gamma import InvGamma
from ..bayesian.distributions.lkj_chol import LKJChol, _log_lkj_normalizer_C
from ..bayesian.distributions.log_norm import LogNormal
from ..bayesian.distributions.norm import Normal
from ..bayesian.distributions.trunc_norm import TruncNormal
from ..bayesian.distributions.uniform import Uniform
from ..bayesian.priors import Prior
from ..bayesian.transforms.affine_logit import AffineLogitTransform
from ..bayesian.transforms.affine_probit import AffineProbitTransform
from ..bayesian.transforms.cholesky_corr import CholeskyCorrTransform
from ..bayesian.transforms.identity import Identity
from ..bayesian.transforms.log import LogTransform
from ..bayesian.transforms.logit import LogitTransform
from ..bayesian.transforms.lower_bounded import LowerBoundedTransform
from ..bayesian.transforms.probit import ProbitTransform
from ..bayesian.transforms.softplus import SoftplusTransform
from ..bayesian.transforms.upper_bounded import UpperBoundedTransform

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]

DIST_NORMAL = 1
DIST_LOG_NORMAL = 2
DIST_HALF_NORMAL = 3
DIST_TRUNC_NORMAL = 4
DIST_HALF_CAUCHY = 5
DIST_BETA = 6
DIST_GAMMA = 7
DIST_INV_GAMMA = 8
DIST_UNIFORM = 9

TRANSFORM_IDENTITY = 1
TRANSFORM_LOG = 2
TRANSFORM_SOFTPLUS = 3
TRANSFORM_LOGIT = 4
TRANSFORM_PROBIT = 5
TRANSFORM_AFFINE_LOGIT = 6
TRANSFORM_AFFINE_PROBIT = 7
TRANSFORM_LOWER_BOUNDED = 8
TRANSFORM_UPPER_BOUNDED = 9

N_DIST_PARAMS = 5
N_TRANSFORM_PARAMS = 3


@dataclass(frozen=True)
class PackedLogPrior:
    scalar_indices: NDI
    scalar_dist_codes: NDI
    scalar_transform_codes: NDI
    scalar_dist_params: NDF
    scalar_transform_params: NDF
    matrix_indices: NDI
    matrix_dims: NDI
    matrix_lengths: NDI
    matrix_etas: NDF
    matrix_log_constants: NDF
    prior_keys: tuple[str, ...]
    prior_object_ids: tuple[int, ...]

    def matches(self, priors: Mapping[str, Any] | None) -> bool:
        if priors is None:
            return False
        if tuple(priors.keys()) != self.prior_keys:
            return False
        return (
            tuple(id(priors[key]) for key in self.prior_keys) == self.prior_object_ids
        )

    def logpdf(self, theta: NDF) -> float64:
        return float64(
            _evaluate_logprior_program(
                theta,
                self.scalar_indices,
                self.scalar_dist_codes,
                self.scalar_transform_codes,
                self.scalar_dist_params,
                self.scalar_transform_params,
                self.matrix_indices,
                self.matrix_dims,
                self.matrix_lengths,
                self.matrix_etas,
                self.matrix_log_constants,
            )
        )


def build_packed_logprior(
    *,
    priors: Mapping[str, Any] | None,
    param_index: Mapping[str, int],
    matrix_blocks: Mapping[str, Any],
    matrix_member_names: set[str],
) -> PackedLogPrior | None:
    if priors is None:
        return None

    scalar_indices: list[int] = []
    scalar_dist_codes: list[int] = []
    scalar_transform_codes: list[int] = []
    scalar_dist_params: list[list[float]] = []
    scalar_transform_params: list[list[float]] = []

    matrix_rows: list[list[int]] = []
    matrix_dims: list[int] = []
    matrix_lengths: list[int] = []
    matrix_etas: list[float] = []
    matrix_log_constants: list[float] = []

    for name, prior in priors.items():
        if name in matrix_blocks:
            block = matrix_blocks[name]
            if not isinstance(prior, Prior):
                return None
            if not isinstance(prior.dist, LKJChol) or not isinstance(
                prior.transform, CholeskyCorrTransform
            ):
                return None
            dim = int(block.dim)
            matrix_rows.append([int(idx) for idx in block.theta_indices])
            matrix_dims.append(dim)
            matrix_lengths.append(len(matrix_rows[-1]))
            eta = float(getattr(prior.dist, "_eta"))
            matrix_etas.append(eta)
            matrix_log_constants.append(float(_log_lkj_normalizer_C(dim, eta)))
            continue

        if name in matrix_member_names:
            continue
        if name not in param_index:
            return None
        if not isinstance(prior, Prior):
            return None

        dist_code, dist_params = _pack_distribution(prior.dist)
        transform_code, transform_params = _pack_transform(prior.transform)
        if dist_code is None or transform_code is None:
            return None

        scalar_indices.append(int(param_index[name]))
        scalar_dist_codes.append(dist_code)
        scalar_transform_codes.append(transform_code)
        scalar_dist_params.append(dist_params)
        scalar_transform_params.append(transform_params)

    n_scalar = len(scalar_indices)
    max_matrix_len = max((len(row) for row in matrix_rows), default=0)
    matrix_index_array = np.full((len(matrix_rows), max_matrix_len), -1, dtype=np.int64)
    for row_idx, row in enumerate(matrix_rows):
        matrix_index_array[row_idx, : len(row)] = np.asarray(row, dtype=np.int64)

    return PackedLogPrior(
        scalar_indices=np.asarray(scalar_indices, dtype=np.int64),
        scalar_dist_codes=np.asarray(scalar_dist_codes, dtype=np.int64),
        scalar_transform_codes=np.asarray(scalar_transform_codes, dtype=np.int64),
        scalar_dist_params=np.asarray(
            scalar_dist_params if n_scalar else np.empty((0, N_DIST_PARAMS)),
            dtype=float64,
        ).reshape(n_scalar, N_DIST_PARAMS),
        scalar_transform_params=np.asarray(
            scalar_transform_params if n_scalar else np.empty((0, N_TRANSFORM_PARAMS)),
            dtype=float64,
        ).reshape(n_scalar, N_TRANSFORM_PARAMS),
        matrix_indices=matrix_index_array,
        matrix_dims=np.asarray(matrix_dims, dtype=np.int64),
        matrix_lengths=np.asarray(matrix_lengths, dtype=np.int64),
        matrix_etas=np.asarray(matrix_etas, dtype=float64),
        matrix_log_constants=np.asarray(matrix_log_constants, dtype=float64),
        prior_keys=tuple(priors.keys()),
        prior_object_ids=tuple(id(prior) for prior in priors.values()),
    )


def _blank_dist_params() -> list[float]:
    return [0.0] * N_DIST_PARAMS


def _blank_transform_params() -> list[float]:
    return [0.0] * N_TRANSFORM_PARAMS


def _pack_distribution(dist: Any) -> tuple[int | None, list[float]]:
    params = _blank_dist_params()
    if isinstance(dist, Normal):
        params[0] = float(getattr(dist, "_mean"))
        params[1] = float(getattr(dist, "_var"))
        return DIST_NORMAL, params
    if isinstance(dist, LogNormal):
        params[0] = float(getattr(dist, "_meanlog"))
        params[1] = float(getattr(dist, "_stdlog"))
        return DIST_LOG_NORMAL, params
    if isinstance(dist, HalfNormal):
        params[0] = float(getattr(dist, "_std"))
        return DIST_HALF_NORMAL, params
    if isinstance(dist, TruncNormal):
        params[0] = float(getattr(dist, "_mean"))
        params[1] = float(getattr(dist, "_std"))
        params[2] = float(getattr(dist, "_low_trunc"))
        params[3] = float(getattr(dist, "_high_trunc"))
        params[4] = float(getattr(dist, "_log_norm"))
        return DIST_TRUNC_NORMAL, params
    if isinstance(dist, HalfCauchy):
        params[0] = float(getattr(dist, "_gamma"))
        return DIST_HALF_CAUCHY, params
    if isinstance(dist, Beta):
        params[0] = float(getattr(dist, "_a"))
        params[1] = float(getattr(dist, "_b"))
        params[2] = float(getattr(dist, "_log_norm"))
        return DIST_BETA, params
    if isinstance(dist, Gamma):
        params[0] = float(getattr(dist, "_a"))
        params[1] = float(getattr(dist, "_theta"))
        params[2] = float(getattr(dist, "_log_norm"))
        return DIST_GAMMA, params
    if isinstance(dist, InvGamma):
        params[0] = float(getattr(dist, "_a"))
        params[1] = float(getattr(dist, "_beta"))
        params[2] = float(getattr(dist, "_log_prefactor"))
        return DIST_INV_GAMMA, params
    if isinstance(dist, Uniform):
        params[0] = float(getattr(dist, "_low"))
        params[1] = float(getattr(dist, "_high"))
        params[2] = float(getattr(dist, "_width"))
        return DIST_UNIFORM, params
    return None, params


def _pack_transform(transform: Any) -> tuple[int | None, list[float]]:
    params = _blank_transform_params()
    if isinstance(transform, Identity):
        return TRANSFORM_IDENTITY, params
    if isinstance(transform, LogTransform):
        return TRANSFORM_LOG, params
    if isinstance(transform, SoftplusTransform):
        return TRANSFORM_SOFTPLUS, params
    if isinstance(transform, LogitTransform):
        return TRANSFORM_LOGIT, params
    if isinstance(transform, ProbitTransform):
        return TRANSFORM_PROBIT, params
    if isinstance(transform, AffineLogitTransform):
        params[0] = float(transform.low)
        params[1] = float(transform.high)
        params[2] = float(transform.high - transform.low)
        return TRANSFORM_AFFINE_LOGIT, params
    if isinstance(transform, AffineProbitTransform):
        params[0] = float(transform.low)
        params[1] = float(transform.high)
        params[2] = float(transform.high - transform.low)
        return TRANSFORM_AFFINE_PROBIT, params
    if isinstance(transform, LowerBoundedTransform):
        params[0] = float(transform.low)
        return TRANSFORM_LOWER_BOUNDED, params
    if isinstance(transform, UpperBoundedTransform):
        params[0] = float(transform.high)
        return TRANSFORM_UPPER_BOUNDED, params
    return None, params


@njit(cache=True)
def _softplus_scalar(x: float64) -> float64:
    if x > 0.0:
        return float64(x + math.log1p(math.exp(-float(x))))
    return float64(math.log1p(math.exp(float(x))))


@njit(cache=True)
def _log_sigmoid_scalar(x: float64) -> float64:
    if x > 0.0:
        return float64(-math.log1p(math.exp(-float(x))))
    return float64(x - math.log1p(math.exp(float(x))))


@njit(cache=True)
def _sigmoid_scalar(x: float64) -> float64:
    if x >= 0.0:
        exp_neg = math.exp(-float(x))
        return float64(1.0 / (1.0 + exp_neg))
    exp_pos = math.exp(float(x))
    return float64(exp_pos / (1.0 + exp_pos))


@njit(cache=True)
def _std_norm_cdf(x: float64) -> float64:
    return float64(0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0))))


@njit(cache=True)
def _std_norm_logpdf(x: float64) -> float64:
    return float64(-0.5 * x * x - 0.5 * math.log(2.0 * math.pi))


@njit(cache=True)
def _transform_inverse_and_logjac(
    code: int,
    params: NDF,
    z: float64,
) -> tuple[float64, float64]:
    if code == TRANSFORM_IDENTITY:
        return z, float64(0.0)
    if code == TRANSFORM_LOG:
        return float64(math.exp(float(z))), z
    if code == TRANSFORM_SOFTPLUS:
        return _softplus_scalar(z), _log_sigmoid_scalar(z)
    if code == TRANSFORM_LOGIT:
        return _sigmoid_scalar(z), float64(
            _log_sigmoid_scalar(z) + _log_sigmoid_scalar(-z)
        )
    if code == TRANSFORM_PROBIT:
        return _std_norm_cdf(z), _std_norm_logpdf(z)
    if code == TRANSFORM_AFFINE_LOGIT:
        low = params[0]
        span = params[2]
        sig = _sigmoid_scalar(z)
        x = float64(low + span * sig)
        logjac = float64(
            math.log(float(span)) + _log_sigmoid_scalar(z) + _log_sigmoid_scalar(-z)
        )
        return x, logjac
    if code == TRANSFORM_AFFINE_PROBIT:
        low = params[0]
        span = params[2]
        cdf = _std_norm_cdf(z)
        return float64(low + span * cdf), float64(
            math.log(float(span)) + _std_norm_logpdf(z)
        )
    if code == TRANSFORM_LOWER_BOUNDED:
        return float64(params[0] + math.exp(float(z))), z
    if code == TRANSFORM_UPPER_BOUNDED:
        return float64(params[0] - math.exp(float(z))), z
    return float64(np.nan), float64(np.nan)


@njit(cache=True)
def _dist_logpdf(code: int, params: NDF, x: float64) -> float64:
    if code == DIST_NORMAL:
        mean = params[0]
        var = params[1]
        return float64(
            -0.5 * math.log(2.0 * math.pi * float(var)) - 0.5 * ((x - mean) ** 2) / var
        )
    if code == DIST_LOG_NORMAL:
        if x <= 0.0:
            return float64(np.nan)
        meanlog = params[0]
        stdlog = params[1]
        return float64(
            -math.log(float(stdlog))
            - math.log(float(x))
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * ((math.log(float(x)) - meanlog) / stdlog) ** 2
        )
    if code == DIST_HALF_NORMAL:
        if x < 0.0:
            return float64(np.nan)
        std = params[0]
        return float64(
            0.5 * math.log(2.0 / math.pi) - math.log(float(std)) - 0.5 * (x / std) ** 2
        )
    if code == DIST_TRUNC_NORMAL:
        mean = params[0]
        std = params[1]
        low = params[2]
        high = params[3]
        log_norm = params[4]
        if x < low or x > high:
            return float64(np.nan)
        z = float64((x - mean) / std)
        return float64(-0.5 * z * z - log_norm)
    if code == DIST_HALF_CAUCHY:
        if x < 0.0:
            return float64(np.nan)
        gamma = params[0]
        centered = x / gamma
        return float64(
            math.log(2.0 / math.pi)
            - math.log(float(gamma))
            - math.log1p(float(centered * centered))
        )
    if code == DIST_BETA:
        if x < 0.0 or x > 1.0:
            return float64(np.nan)
        a = params[0]
        b = params[1]
        log_norm = params[2]
        out = float64(0.0)
        if a != 1.0:
            out += float64((a - 1.0) * math.log(float(x)))
        if b != 1.0:
            out += float64((b - 1.0) * math.log1p(float(-x)))
        return float64(out - log_norm)
    if code == DIST_GAMMA:
        if x < 0.0:
            return float64(np.nan)
        a = params[0]
        theta = params[1]
        log_norm = params[2]
        out = float64(0.0)
        if a != 1.0:
            out += float64((a - 1.0) * math.log(float(x)))
        return float64(out - x / theta - log_norm)
    if code == DIST_INV_GAMMA:
        if x <= 0.0:
            return float64(np.nan)
        a = params[0]
        beta = params[1]
        log_prefactor = params[2]
        return float64(log_prefactor - (a + 1.0) * math.log(float(x)) - beta / x)
    if code == DIST_UNIFORM:
        low = params[0]
        high = params[1]
        width = params[2]
        if x < low or x > high:
            return float64(np.nan)
        return float64(-math.log(float(width)))
    return float64(np.nan)


@njit(cache=True)
def _lkj_chol_logjac(z: NDF, dim: int, length: int) -> float64:
    total = float64(0.0)
    idx = 0
    for k in range(1, dim):
        rem = float64(1.0)
        for _ in range(k):
            if idx >= length:
                return float64(np.nan)
            cpc_i = float64(math.tanh(float(z[idx])))
            total += float64(0.5 * math.log(max(float(rem), 1e-300)))
            total += float64(math.log1p(float(-(cpc_i * cpc_i))))
            rem = float64(rem * (1.0 - cpc_i * cpc_i))
            idx += 1
    return total


@njit(cache=True)
def _lkj_chol_logpdf_from_z(
    z: NDF,
    dim: int,
    length: int,
    eta: float64,
    log_constant: float64,
) -> float64:
    log_kernel = float64(0.0)
    idx = 0
    for i in range(1, dim):
        rem = float64(1.0)
        for _ in range(i):
            if idx >= length:
                return float64(np.nan)
            cpc_i = float64(math.tanh(float(z[idx])))
            rem = float64(rem * (1.0 - cpc_i * cpc_i))
            idx += 1
        diag = float64(math.sqrt(max(float(rem), 1e-14)))
        exponent = float64(dim - i + 2.0 * eta - 3.0)
        log_kernel += float64(exponent * math.log(float(diag)))
    return float64(log_constant + log_kernel + _lkj_chol_logjac(z, dim, length))


@njit(cache=True)
def _evaluate_logprior_program(
    theta: NDF,
    scalar_indices: NDI,
    scalar_dist_codes: NDI,
    scalar_transform_codes: NDI,
    scalar_dist_params: NDF,
    scalar_transform_params: NDF,
    matrix_indices: NDI,
    matrix_dims: NDI,
    matrix_lengths: NDI,
    matrix_etas: NDF,
    matrix_log_constants: NDF,
) -> float64:
    lp = float64(0.0)
    for i in range(scalar_indices.shape[0]):
        z = float64(theta[scalar_indices[i]])
        x, logjac = _transform_inverse_and_logjac(
            int(scalar_transform_codes[i]), scalar_transform_params[i], z
        )
        if np.isnan(x) or np.isnan(logjac):
            return float64(np.nan)
        logp = _dist_logpdf(int(scalar_dist_codes[i]), scalar_dist_params[i], x)
        if np.isnan(logp):
            return float64(np.nan)
        lp += float64(logp + logjac)

    for block_idx in range(matrix_dims.shape[0]):
        length = int(matrix_lengths[block_idx])
        z_block = np.empty((length,), dtype=float64)
        for j in range(length):
            z_block[j] = float64(theta[matrix_indices[block_idx, j]])
        block_lp = _lkj_chol_logpdf_from_z(
            z_block,
            int(matrix_dims[block_idx]),
            length,
            float64(matrix_etas[block_idx]),
            float64(matrix_log_constants[block_idx]),
        )
        if np.isnan(block_lp):
            return float64(np.nan)
        lp += block_lp

    return lp
