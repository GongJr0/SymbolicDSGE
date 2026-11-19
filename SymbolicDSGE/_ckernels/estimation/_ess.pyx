# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport int64_t
import numpy as np
import os

cdef extern from "sdsge_common.h":
    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int

cdef extern from "ess.h":
    int c_SDSGE_ESS_OK "SDSGE_ESS_OK"
    int c_SDSGE_ESS_EBADSHAPE "SDSGE_ESS_EBADSHAPE"
    int c_SDSGE_ESS_EUDEFVARIANCE "SDSGE_ESS_EUDEFVARIANCE"
    int c_SDSGE_ESS_ENODECAY "SDSGE_ESS_ENODECAY"
    int c_SDSGE_ESS_ENONPOSTAU "SDSGE_ESS_ENONPOSTAU"

    arena_size sdsge_ess_rhat_arena_size(const int64_t N,
                                         const int64_t n_chains,
                                         const int64_t n_workers) nogil
    void sdsge_ess_rhat(const double *x, const int64_t N,
                        const int64_t n_par, const int64_t n_chains,
                        const int64_t n_workers, double *arena, double *ess,
                        double *rhat, int *status) nogil


SDSGE_ESS_OK = c_SDSGE_ESS_OK
SDSGE_ESS_EBADSHAPE = c_SDSGE_ESS_EBADSHAPE
SDSGE_ESS_EUDEFVARIANCE = c_SDSGE_ESS_EUDEFVARIANCE
SDSGE_ESS_ENODECAY = c_SDSGE_ESS_ENODECAY
SDSGE_ESS_ENONPOSTAU = c_SDSGE_ESS_ENONPOSTAU


def ess_rhat(x):
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")

    cdef int64_t n_chains = x.shape[0]
    cdef int64_t N = x.shape[1]
    cdef int64_t n_par = x.shape[2]

    if n_chains < 1:
        raise ValueError("Input must have at least one chain.")

    if N < 4 or n_par == 0:
        raise ValueError(
            "Input samples must have at least four elements per "
            "column and at least one parameter being sampled."
        )
    cdef int64_t n_workers = worker_count(n_par)
    cdef int64_t asz = sdsge_ess_rhat_arena_size(N, n_chains, n_workers).n_float
    cdef double[::1] arena = np.empty(asz, dtype=np.float64)

    cdef double[::1] outv, rhatv
    cdef int[::1] statusv

    out = np.empty(n_par, dtype=np.float64)
    rhat = np.empty(n_par, dtype=np.float64)
    status = np.empty(n_par, dtype=np.intc)

    outv = out
    statusv = status
    rhatv = rhat

    # parameter-major pooled view, discarded after computation.
    cdef double[::1] xv = np.ascontiguousarray(
        x.transpose(2, 0, 1), dtype=np.float64
    ).reshape(-1)

    with nogil:
        sdsge_ess_rhat(
            &xv[0], N, n_par, n_chains, n_workers, &arena[0],
            &outv[0], &rhatv[0], &statusv[0],
        )
    return out, rhat, status.astype(np.int64)


def worker_count(object n_items):
    cpu_count = (
        os.process_cpu_count()
        if hasattr(os, "process_cpu_count")
        else os.cpu_count() or 1
    )
    if n_items is None:
        return cpu_count
    return min(n_items, cpu_count)
