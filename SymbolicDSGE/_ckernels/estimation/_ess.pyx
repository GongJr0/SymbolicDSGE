# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
from libc.stdint cimport int64_t
import numpy as np

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

    arena_size sdsge_ess_1d_arena_size(int64_t n) nogil
    int sdsge_ess_1d(const double *x, int64_t n,
                     double *arena, double *out) nogil


SDSGE_ESS_OK = c_SDSGE_ESS_OK
SDSGE_ESS_EBADSHAPE = c_SDSGE_ESS_EBADSHAPE
SDSGE_ESS_EUDEFVARIANCE = c_SDSGE_ESS_EUDEFVARIANCE
SDSGE_ESS_ENODECAY = c_SDSGE_ESS_ENODECAY
SDSGE_ESS_ENONPOSTAU = c_SDSGE_ESS_ENONPOSTAU


def ess1d(x):
    cdef int64_t n = x.shape[0]
    if n < 2:
        raise ValueError("Input sample must have at least two elements")

    cdef double[::1] xv = np.ascontiguousarray(x, dtype=np.float64)
    cdef int64_t asz = sdsge_ess_1d_arena_size(n).n_float
    cdef double[::1] arena = np.empty(asz, dtype=np.float64)
    cdef double out = 0.0

    with nogil:
        sdsge_ess_1d(&xv[0], n, &arena[0], &out)
    return float(out)


def ess(x):
    if x.ndim != 2:
        raise ValueError("Input must be a 2D array")

    cdef int64_t n = x.shape[0]
    cdef int64_t p = x.shape[1]

    if n < 2 or p == 0:
        raise ValueError(
            "Input samples must have at least two elements per column "
            "and at least one column."
        )

    cdef double[::1] col
    cdef int64_t asz = sdsge_ess_1d_arena_size(n).n_float
    cdef double[::1] arena = np.empty(asz, dtype=np.float64)

    out_arr = np.empty(p, dtype=np.float64)
    cdef double[::1] out = out_arr

    cdef int64_t i
    for i in range(p):
        col = np.ascontiguousarray(x[:, i], dtype=np.float64)
        with nogil:
            sdsge_ess_1d(&col[0], n, &arena[0], &out[i])
    return out_arr.tolist()


def ess1d_kernel_probe(x):
    cdef int64_t n = x.shape[0]
    cdef double[::1] xv = np.ascontiguousarray(x, dtype=np.float64)
    cdef int64_t asz = sdsge_ess_1d_arena_size(n).n_float
    cdef double[::1] arena = np.empty(asz, dtype=np.float64)
    cdef double out = 0.0
    cdef int rc

    with nogil:
        rc = sdsge_ess_1d(&xv[0], n, &arena[0], &out)
    return int(rc), float(out)
