"""Numba reference kernel for the packed log-prior program.

Relocated from ``SymbolicDSGE.estimation.prior_program`` when that module went
native-only. A numba reimplementation of the native ``logprior_program`` kernel;
the parity tests check the native kernel against it.

Kept ``@njit`` -- and this is load-bearing, not perf: numba's ``math.log`` /
``math.log1p`` mirror C libm and return ``-inf``/``nan`` on a domain error, but
CPython's ``math.log`` *raises* ``ValueError``. The parity tests feed
distribution-support-boundary values (e.g. ``x = 0`` for Beta/Gamma) where the
native kernel returns ``-inf``/``nan``, so a plain-Python transcription raises
instead of matching. The dispatch-code enums stay in the library (the packer's
contract with the native kernel) and are imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from numba import njit

from SymbolicDSGE.estimation.prior_program import DistCode, TransformCode

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


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
    if code == TransformCode.IDENTITY:
        return z, float64(0.0)
    if code == TransformCode.LOG:
        return float64(math.exp(float(z))), z
    if code == TransformCode.SOFTPLUS:
        return _softplus_scalar(z), _log_sigmoid_scalar(z)
    if code == TransformCode.LOGIT:
        return _sigmoid_scalar(z), float64(
            _log_sigmoid_scalar(z) + _log_sigmoid_scalar(-z)
        )
    if code == TransformCode.PROBIT:
        return _std_norm_cdf(z), _std_norm_logpdf(z)
    if code == TransformCode.AFFINE_LOGIT:
        low = params[0]
        span = params[2]
        sig = _sigmoid_scalar(z)
        x = float64(low + span * sig)
        logjac = float64(
            math.log(float(span)) + _log_sigmoid_scalar(z) + _log_sigmoid_scalar(-z)
        )
        return x, logjac
    if code == TransformCode.AFFINE_PROBIT:
        low = params[0]
        span = params[2]
        cdf = _std_norm_cdf(z)
        return float64(low + span * cdf), float64(
            math.log(float(span)) + _std_norm_logpdf(z)
        )
    if code == TransformCode.LOWER_BOUNDED:
        return float64(params[0] + math.exp(float(z))), z
    if code == TransformCode.UPPER_BOUNDED:
        return float64(params[0] - math.exp(float(z))), z
    if code == TransformCode.TANH:
        # log(sech^2(z)); mirrors sdsge_log_sech2 in transforms.c / the estimation
        # scatter dispatch. Avoids the 1 - tanh^2 cancellation and cosh overflow.
        ay = abs(float(z))
        return float64(math.tanh(float(z))), float64(
            2.0 * (0.6931471805599453 - ay - math.log1p(math.exp(-2.0 * ay)))
        )
    return float64(np.nan), float64(np.nan)


@njit(cache=True)
def _dist_logpdf(code: int, params: NDF, x: float64) -> float64:
    if code == DistCode.NORMAL:
        mean = params[0]
        var = params[1]
        return float64(
            -0.5 * math.log(2.0 * math.pi * float(var)) - 0.5 * ((x - mean) ** 2) / var
        )
    if code == DistCode.LOG_NORMAL:
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
    if code == DistCode.HALF_NORMAL:
        if x < 0.0:
            return float64(np.nan)
        std = params[0]
        return float64(
            0.5 * math.log(2.0 / math.pi) - math.log(float(std)) - 0.5 * (x / std) ** 2
        )
    if code == DistCode.TRUNC_NORMAL:
        mean = params[0]
        std = params[1]
        low = params[2]
        high = params[3]
        log_norm = params[4]
        if x < low or x > high:
            return float64(np.nan)
        z = float64((x - mean) / std)
        return float64(-0.5 * z * z - log_norm)
    if code == DistCode.HALF_CAUCHY:
        if x < 0.0:
            return float64(np.nan)
        gamma = params[0]
        centered = x / gamma
        return float64(
            math.log(2.0 / math.pi)
            - math.log(float(gamma))
            - math.log1p(float(centered * centered))
        )
    if code == DistCode.BETA:
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
    if code == DistCode.GAMMA:
        if x < 0.0:
            return float64(np.nan)
        a = params[0]
        theta = params[1]
        log_norm = params[2]
        out = float64(0.0)
        if a != 1.0:
            out += float64((a - 1.0) * math.log(float(x)))
        return float64(out - x / theta - log_norm)
    if code == DistCode.INV_GAMMA:
        if x <= 0.0:
            return float64(np.nan)
        a = params[0]
        beta = params[1]
        log_prefactor = params[2]
        return float64(log_prefactor - (a + 1.0) * math.log(float(x)) - beta / x)
    if code == DistCode.UNIFORM:
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
    matrix_offsets: NDI,
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
        offset = int(matrix_offsets[block_idx])
        z_block = np.empty((length,), dtype=float64)
        for j in range(length):
            z_block[j] = float64(theta[offset + j])
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


# Correlation / covariance reparameterization references. Numba oracles for the
# native ``cov_from_unconstrained`` (Cholesky factor L + covariance) and
# ``unconstrained_from_corr_chol`` (its inverse). Relocated from
# ``SymbolicDSGE.estimation.backend`` when those transforms went native-only.
@njit(cache=True)
def _corr_chol_from_unconstrained(z: NDF, K: int) -> NDF:
    cpc: NDF = np.tanh(z)
    L: NDF = np.zeros((K, K), dtype=float64)
    L[0, 0] = 1.0
    idx: int = 0
    for k in range(1, K):
        rem: float64 = float64(1.0)
        for j in range(k):
            v = float64(np.sqrt(max(rem, 1e-14)))
            L[k, j] = float64(cpc[idx] * v)
            rem = float64(rem - L[k, j] * L[k, j])
            idx += 1
        L[k, k] = float64(np.sqrt(max(rem, 1e-14)))
    return L


@njit(cache=True)
def _unconstrained_from_corr_chol(L: NDF) -> NDF:
    K = L.shape[0]
    n_cpc = (K * (K - 1)) // 2
    z = np.empty((n_cpc,), dtype=float64)
    idx = 0
    for k in range(1, K):
        rem = float64(1.0)
        for j in range(k):
            v = float64(np.sqrt(max(rem, 1e-14)))
            cpc = float64(L[k, j] / v) if v > 0.0 else float64(0.0)
            if cpc < (-1.0 + 1e-14):
                cpc = float64(-1.0 + 1e-14)
            elif cpc > (1.0 - 1e-14):
                cpc = float64(1.0 - 1e-14)
            z[idx] = float64(np.arctanh(cpc))
            rem = float64(rem - L[k, j] * L[k, j])
            idx += 1
    return z


@dataclass(frozen=True)
class RWMReference:
    """Frozen output of the numpy-era adaptive-RWM reference chain.

    ``kept`` / ``kept_lp`` are in **theta** (unconstrained) space, the space the
    chain actually walks; callers map ``kept`` to named parameters the same way
    ``Estimator.mcmc`` does (``theta_to_params`` per row) when a param-space
    comparison is wanted. ``n_accepted`` / ``accept_rate`` count over all
    ``total_steps`` (burn-in included), matching the estimator.
    """

    kept: NDF
    kept_lp: NDF
    accept_rate: float
    n_accepted: int
    total_steps: int


def adaptive_rwm_reference(
    logpost,
    theta0: NDF,
    rng: np.random.Generator,
    *,
    n_draws: int,
    burn_in: int = 1000,
    thin: int = 1,
    adapt: bool = True,
    adapt_start: int = 100,
    adapt_interval: int = 25,
    proposal_scale: float = 0.1,
    adapt_epsilon: float = 1e-8,
) -> RWMReference:
    """Verbatim transcription of the numpy-era ``Estimator.mcmc`` mainloop.

    This is the **pre-port oracle**: the Haario adaptive random-walk Metropolis
    chain exactly as it ran on numpy (``rng.multivariate_normal`` SVD proposal +
    ``np.cov`` batch adaptation + ``2.38**2/d`` scaling), lifted out of the lib so
    it stays frozen once ``Estimator.mcmc`` is rewritten onto the native frozen
    stream. The native chain is validated against the marginals this produces
    (KS + acceptance rate), per the statistical-equivalence contract (#331); it is
    deliberately NOT bit-reproduced.

    ``logpost(theta) -> float`` is the caller's safe log-posterior (``-inf`` on a
    BK violation / non-finite eval, auto-rejected here via the finiteness gate).
    RNG consumption order is load-bearing for the faithfulness test: one
    ``multivariate_normal`` per step, then one ``random()`` only when the proposal
    logpost is finite -- identical to the estimator.
    """
    current = np.asarray(theta0, dtype=float64).copy()
    d = current.shape[0]
    total_steps = burn_in + n_draws * thin
    cov = (float64(proposal_scale) ** 2) * np.eye(d, dtype=float64)
    scale = float64((2.38**2) / d)

    cur_lp = float64(logpost(current))
    accepted = 0

    history = np.empty((total_steps, d), dtype=float64)
    kept = np.empty((n_draws, d), dtype=float64)
    kept_lp = np.empty((n_draws,), dtype=float64)

    keep_i = 0
    eye_d = np.eye(d, dtype=float64)

    for t in range(total_steps):
        prop = rng.multivariate_normal(current, cov)
        prop_lp = float64(logpost(prop))

        if np.isfinite(prop_lp):
            log_alpha = prop_lp - cur_lp
            if np.log(rng.random()) < log_alpha:
                current = prop
                cur_lp = prop_lp
                accepted += 1

        history[t] = current

        if adapt and t < burn_in and t >= adapt_start and (t + 1) % adapt_interval == 0:
            hist = history[: t + 1]
            if d == 1:
                var = np.var(hist[:, 0], ddof=1) if hist.shape[0] > 1 else float64(1.0)
                cov = np.array([[scale * var + adapt_epsilon]], dtype=float64)
            else:
                emp = np.cov(hist.T, ddof=1) if hist.shape[0] > 1 else eye_d
                cov = scale * (np.asarray(emp, dtype=float64) + adapt_epsilon * eye_d)

        if t >= burn_in and (t - burn_in) % thin == 0:
            kept[keep_i] = current
            kept_lp[keep_i] = cur_lp
            keep_i += 1

    return RWMReference(
        kept=kept,
        kept_lp=kept_lp,
        accept_rate=float(accepted / total_steps),
        n_accepted=int(accepted),
        total_steps=int(total_steps),
    )


@njit(cache=True)
def _R_from_unconstrained(u: NDF, K: int) -> tuple[NDF, NDF, NDF]:
    log_std = u[:K]
    z = u[K:]
    std = np.exp(log_std).astype(float64)
    Lcorr: NDF = _corr_chol_from_unconstrained(z.astype(float64), K)
    LcorrT: NDF = np.ascontiguousarray(Lcorr.T)
    corr: NDF = Lcorr @ LcorrT
    std_col = std.reshape((K, 1))
    std_row = std.reshape((1, K))
    R = corr * std_col * std_row
    return (R.astype(float64), std, Lcorr)
