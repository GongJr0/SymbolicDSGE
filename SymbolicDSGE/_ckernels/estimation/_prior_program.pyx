# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim for the native packed log-prior kernels.

No numeric logic here -- only buffer->pointer marshalling and the GIL release.
The algorithms live in prior_program.c; each ``def`` mirrors the matching numba
helper in SymbolicDSGE/estimation/prior_program.py. The leaves are exposed so the
parity tests can hit them directly; ``logprior_program`` is the per-replication
hot path the estimator dispatches to. A NaN result means "fall back to the numba
path" (out-of-support / unknown code), matching the numba kernel.
"""

from libc.stdint cimport int64_t

import numpy as np


cdef extern from "prior_program.h":
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
    double sdsge_logprior_program(
        double *theta, int64_t *scalar_indices, int64_t *scalar_dist_codes,
        int64_t *scalar_transform_codes, double *scalar_dist_params,
        double *scalar_transform_params, int64_t n_scalar,
        int64_t *matrix_indices, int64_t *matrix_dims, int64_t *matrix_lengths,
        double *matrix_etas, double *matrix_log_constants, int64_t n_blocks,
        int64_t max_matrix_len) nogil
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


def logprior_program(double[::1] theta,
                     int64_t[::1] scalar_indices,
                     int64_t[::1] scalar_dist_codes,
                     int64_t[::1] scalar_transform_codes,
                     double[:, ::1] scalar_dist_params,
                     double[:, ::1] scalar_transform_params,
                     int64_t[:, ::1] matrix_indices,
                     int64_t[::1] matrix_dims,
                     int64_t[::1] matrix_lengths,
                     double[::1] matrix_etas,
                     double[::1] matrix_log_constants):
    """Full packed log-prior. Returns the scalar logprior (NaN -> numba fallback)."""
    cdef int64_t n_scalar = scalar_indices.shape[0]
    cdef int64_t n_blocks = matrix_dims.shape[0]
    cdef int64_t max_matrix_len = (
        matrix_indices.shape[1] if matrix_indices.shape[0] > 0 else 0
    )

    cdef double *theta_p = &theta[0] if theta.shape[0] > 0 else NULL
    cdef int64_t *si = &scalar_indices[0] if n_scalar > 0 else NULL
    cdef int64_t *sdc = &scalar_dist_codes[0] if n_scalar > 0 else NULL
    cdef int64_t *stc = &scalar_transform_codes[0] if n_scalar > 0 else NULL
    cdef double *sdp = &scalar_dist_params[0, 0] if n_scalar > 0 else NULL
    cdef double *stp = &scalar_transform_params[0, 0] if n_scalar > 0 else NULL
    cdef int64_t *mi = &matrix_indices[0, 0] if n_blocks > 0 else NULL
    cdef int64_t *md = &matrix_dims[0] if n_blocks > 0 else NULL
    cdef int64_t *ml = &matrix_lengths[0] if n_blocks > 0 else NULL
    cdef double *me = &matrix_etas[0] if n_blocks > 0 else NULL
    cdef double *mlc = &matrix_log_constants[0] if n_blocks > 0 else NULL

    cdef double out
    with nogil:
        out = sdsge_logprior_program(theta_p, si, sdc, stc, sdp, stp, n_scalar,
                                     mi, md, ml, me, mlc, n_blocks,
                                     max_matrix_len)
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
