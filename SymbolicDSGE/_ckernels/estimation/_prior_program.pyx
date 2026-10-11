# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim for the native packed log-prior kernels.

Buffer to pointer marshalling and the GIL release only; the algorithms live in
prior_program.c. Parity oracle: SymbolicDSGE/estimation/prior_program.py.
``logprior_program`` is the per-replication hot path; the leaves are exposed for
the parity tests. A NaN result means out-of-support or an unknown code, which
reaches the optimizer as a rejected point: the estimation stack is native-only
and has no Python leg to fall back to.
"""

from libc.stdint cimport int64_t

import numpy as np


cdef extern from "prior_program.h":
    int SDSGE_N_DIST_PARAMS
    int SDSGE_N_TRANSFORM_PARAMS

    ctypedef struct sdsge_prior_tables:
        int has_prior
        const int64_t *dist_codes
        const int64_t *transform_codes
        const double *dist_params
        const double *transform_params
        int64_t n_theta
        int include_logjac

    void sdsge_dist_logpdf(int64_t code, double *params, double x,
                           double *out_logpdf) nogil
    void sdsge_transform_inverse_and_logjac(int64_t code, double *params,
                                            double z, double *out_x,
                                            double *out_logjac) nogil
    void sdsge_lkj_chol_logjac(double *z, int64_t dim, int64_t length,
                               double *out_logjac) nogil
    void sdsge_lkj_chol_logpdf_from_z(double *z, int64_t dim, int64_t length,
                                      double eta, double log_const,
                                      double *out_logpdf) nogil
    double sdsge_logprior_program(double *theta,
                                  const sdsge_prior_tables *pr) nogil
    void sdsge_cov_from_unconstrained(double *z, double *std, int64_t K,
                                      double *scratch_M, double *out) nogil
    void sdsge_unconstrained_from_corr_chol(double *L, int64_t K,
                                            double *out_z) nogil


def dist_logpdf(int64_t code, double[::1] params, double x):
    """Scalar family log-density at x. Returns the logpdf (NaN out-of-support)."""
    cdef double out = 0.0
    with nogil:
        sdsge_dist_logpdf(code, &params[0], x, &out)
    return out


def transform_inverse_and_logjac(int64_t code, double[::1] params, double z):
    """Inverse transform z -> x and its log-jacobian. Returns (x, logjac)."""
    cdef double out_x = 0.0
    cdef double out_logjac = 0.0
    with nogil:
        sdsge_transform_inverse_and_logjac(code, &params[0], z, &out_x,
                                           &out_logjac)
    return out_x, out_logjac


def lkj_chol_logjac(double[::1] z, int64_t dim, int64_t length):
    """LKJ-Cholesky log-jacobian. Returns the logjac (NaN if length too short)."""
    cdef double *zp = &z[0] if z.shape[0] > 0 else NULL
    cdef double out = 0.0
    with nogil:
        sdsge_lkj_chol_logjac(zp, dim, length, &out)
    return out


def lkj_chol_logpdf_from_z(double[::1] z, int64_t dim, int64_t length,
                           double eta, double log_const):
    """LKJ-Cholesky log-density from the unconstrained z block."""
    cdef double *zp = &z[0] if z.shape[0] > 0 else NULL
    cdef double out = 0.0
    with nogil:
        sdsge_lkj_chol_logpdf_from_z(zp, dim, length, eta, log_const, &out)
    return out


def logprior_program(theta not None,
                     int64_t[::1] dist_codes,
                     int64_t[::1] transform_codes,
                     double[:, ::1] dist_params,
                     double[:, ::1] transform_params,
                     bint include_logjac=True):
    """Full packed log-prior over one theta layout's tables.

    Every column runs to ``n_theta``, which the kernel walks once: a block's
    row repeats across all of its slots, and the kernel reads its ``K`` off the
    transform row to skip the run it heads, so the block offsets and lengths
    are the layout rather than arguments.

    ``include_logjac`` picks the density: with it, the prior over theta, the
    change of variables a sampler walks; without, the prior over the parameters
    read at that theta. The C objectives make the same choice per entry point,
    off ``sdsge_prior_tables.include_logjac``."""
    cdef double[::1] thetav = np.ascontiguousarray(theta, dtype=np.float64)
    cdef int64_t n_theta = thetav.shape[0]
    if n_theta == 0:
        return 0.0

    if (dist_codes.shape[0] != n_theta
            or transform_codes.shape[0] != n_theta
            or dist_params.shape[0] != n_theta
            or transform_params.shape[0] != n_theta):
        raise ValueError(
            "every table column must run to theta's length."
        )
    if (dist_params.shape[1] != SDSGE_N_DIST_PARAMS
            or transform_params.shape[1] != SDSGE_N_TRANSFORM_PARAMS):
        raise ValueError(
            f"packed rows must be {SDSGE_N_DIST_PARAMS} and "
            f"{SDSGE_N_TRANSFORM_PARAMS} wide."
        )

    cdef sdsge_prior_tables pr
    # Unread by the program: it is the estimator's record of which leg packed
    # the tables, and a table reaching here carries densities by construction.
    pr.has_prior = 1
    pr.dist_codes = &dist_codes[0]
    pr.transform_codes = &transform_codes[0]
    pr.dist_params = &dist_params[0, 0]
    pr.transform_params = &transform_params[0, 0]
    pr.n_theta = n_theta
    pr.include_logjac = include_logjac

    cdef double out
    with nogil:
        out = sdsge_logprior_program(&thetav[0], &pr)
    return out


def cov_from_unconstrained(z, std):
    """Unconstrained CPC values + stds -> (K x K covariance, K x K correlation
    Cholesky factor L). ``K`` is taken from ``std``; ``z`` has length K(K-1)/2
    (empty for K == 1). ``L`` is lower-triangular (upper stays zero)."""
    z = np.ascontiguousarray(z, dtype=np.float64)
    std = np.ascontiguousarray(std, dtype=np.float64)
    cdef int64_t K = std.shape[0]
    cdef double[::1] z_mv = z
    cdef double[::1] std_mv = std
    # L zero-initialized: the kernel writes only the lower triangle + diagonal.
    L = np.zeros((K, K), dtype=np.float64)
    out = np.empty((K, K), dtype=np.float64)
    cdef double[:, ::1] L_mv = L
    cdef double[:, ::1] out_mv = out
    cdef double *zp = &z_mv[0] if z_mv.shape[0] > 0 else NULL
    cdef double *sp = &std_mv[0] if K > 0 else NULL
    cdef double *lp = &L_mv[0, 0] if K > 0 else NULL
    cdef double *op = &out_mv[0, 0] if K > 0 else NULL
    with nogil:
        sdsge_cov_from_unconstrained(zp, sp, K, lp, op)
    return out, L


def unconstrained_from_corr_chol(L):
    """Correlation Cholesky factor (K x K) -> unconstrained CPC values of length
    K(K-1)/2 (empty for K == 1). Inverse of the Cholesky stage above."""
    L = np.ascontiguousarray(L, dtype=np.float64)
    cdef int64_t K = L.shape[0]
    cdef int64_t n_cpc = (K * (K - 1)) // 2
    cdef double[:, ::1] L_mv = L
    out_z = np.empty((n_cpc,), dtype=np.float64)
    cdef double[::1] out_mv = out_z
    cdef double *lp = &L_mv[0, 0] if K > 0 else NULL
    cdef double *op = &out_mv[0] if n_cpc > 0 else NULL
    with nogil:
        sdsge_unconstrained_from_corr_chol(lp, K, op)
    return out_z
