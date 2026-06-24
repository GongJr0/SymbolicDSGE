# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython shim for the linear Kalman filter hot loop.

Allocates the history output arrays, marshals the NumPy buffers into the C
kf_inputs/kf_outputs structs, runs kf_hot_loop under nogil, and translates the
status code back into the exceptions KalmanFilter.run already handles. The
public signature and return shape match the numba `_kalman_hot_loop` so the two
are interchangeable at the call site.
"""

import numpy as np
from numpy.linalg import LinAlgError

from libc.stdint cimport int64_t


cdef extern from "kalman.h":
    int KF_OK
    int KF_ERR_MATRIX_CONDITION
    int KF_ERR_ALLOC

    ctypedef struct kf_inputs:
        int64_t n
        int64_t m
        int64_t k
        int64_t T
        double *A
        double *B
        double *C
        double *d
        double *Q
        double *R
        double *y
        double *x0
        double *P0
        int symmetrize
        double jitter
        int return_shocks
        int store_history

    ctypedef struct kf_outputs:
        double *x_pred
        double *x_filt
        double *P_pred
        double *P_filt
        double *y_pred
        double *y_filt
        double *innov
        double *std_innov
        double *S
        double *eps_hat
        double *loglik

    int kf_hot_loop(const kf_inputs *inp, kf_outputs *outp) nogil


def kalman_hot_loop(
    int64_t T,
    nmk,
    double[:, ::1] A,
    double[:, ::1] B,
    double[:, ::1] C,
    double[::1] d,
    double[:, ::1] Q,
    double[:, ::1] R,
    double[:, ::1] y,
    double[::1] x0,
    double[:, ::1] P0,
    bint symmetrize,
    double jitter,
    bint return_shocks=False,
    bint store_history=True,
):
    """Run the linear Kalman filter; mirrors numba `_kalman_hot_loop`.

    Returns ``(KF_OK, (0.0, 0.0, 0.0), out)`` where ``out`` is the history tuple
    + loglik. Raises ``numpy.linalg.LinAlgError`` on a non-PD innovation
    covariance (so KalmanFilter.run's existing ``except`` maps it to
    ``MatrixConditionError``).
    """
    cdef int64_t n = nmk[0]
    cdef int64_t m = nmk[1]
    cdef int64_t k = nmk[2]

    cdef int64_t hist_T = T if store_history else 0
    cdef int64_t shock_T = T if (return_shocks and store_history) else 0

    x_pred = np.zeros((hist_T, n), dtype=np.float64)
    x_filt = np.zeros((hist_T, n), dtype=np.float64)
    P_pred = np.zeros((hist_T, n, n), dtype=np.float64)
    P_filt = np.zeros((hist_T, n, n), dtype=np.float64)
    y_pred = np.zeros((hist_T, m), dtype=np.float64)
    y_filt = np.zeros((hist_T, m), dtype=np.float64)
    innov = np.zeros((hist_T, m), dtype=np.float64)
    std_innov = np.zeros((hist_T, m), dtype=np.float64)
    S = np.zeros((hist_T, m, m), dtype=np.float64)
    eps_hat = np.zeros((shock_T, k), dtype=np.float64)
    cdef double loglik = 0.0

    cdef double[:, ::1] x_pred_mv = x_pred
    cdef double[:, ::1] x_filt_mv = x_filt
    cdef double[:, :, ::1] P_pred_mv = P_pred
    cdef double[:, :, ::1] P_filt_mv = P_filt
    cdef double[:, ::1] y_pred_mv = y_pred
    cdef double[:, ::1] y_filt_mv = y_filt
    cdef double[:, ::1] innov_mv = innov
    cdef double[:, ::1] std_innov_mv = std_innov
    cdef double[:, :, ::1] S_mv = S
    cdef double[:, ::1] eps_hat_mv = eps_hat

    cdef kf_inputs inp
    inp.n = n
    inp.m = m
    inp.k = k
    inp.T = T
    inp.A = &A[0, 0]
    inp.B = &B[0, 0]
    inp.C = &C[0, 0]
    inp.d = &d[0]
    inp.Q = &Q[0, 0]
    inp.R = &R[0, 0]
    inp.y = &y[0, 0] if T > 0 else NULL
    inp.x0 = &x0[0]
    inp.P0 = &P0[0, 0]
    inp.symmetrize = symmetrize
    inp.jitter = jitter
    inp.return_shocks = return_shocks
    inp.store_history = store_history

    cdef kf_outputs outp
    outp.x_pred = &x_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.x_filt = &x_filt_mv[0, 0] if hist_T > 0 else NULL
    outp.P_pred = &P_pred_mv[0, 0, 0] if hist_T > 0 else NULL
    outp.P_filt = &P_filt_mv[0, 0, 0] if hist_T > 0 else NULL
    outp.y_pred = &y_pred_mv[0, 0] if hist_T > 0 else NULL
    outp.y_filt = &y_filt_mv[0, 0] if hist_T > 0 else NULL
    outp.innov = &innov_mv[0, 0] if hist_T > 0 else NULL
    outp.std_innov = &std_innov_mv[0, 0] if hist_T > 0 else NULL
    outp.S = &S_mv[0, 0, 0] if hist_T > 0 else NULL
    outp.eps_hat = &eps_hat_mv[0, 0] if shock_T > 0 else NULL
    outp.loglik = &loglik

    cdef int status
    with nogil:
        status = kf_hot_loop(&inp, &outp)

    if status == KF_ERR_MATRIX_CONDITION:
        raise LinAlgError("Innovation covariance is not positive definite.")
    if status == KF_ERR_ALLOC:
        raise MemoryError("kf_hot_loop: scratch allocation failed.")

    return (
        KF_OK,
        (0.0, 0.0, 0.0),
        (
            x_pred,
            x_filt,
            P_pred,
            P_filt,
            y_pred,
            y_filt,
            innov,
            std_innov,
            S,
            eps_hat,
            loglik,
        ),
    )
