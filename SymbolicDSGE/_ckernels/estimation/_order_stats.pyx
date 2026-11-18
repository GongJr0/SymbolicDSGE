# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from libc.stdint cimport int64_t
import numpy as np
import os

cdef extern from "order_stats.h":
    void sdsge_hpd_intervals(double *pooled, const double alpha, const int64_t S,
                             const int64_t n_par, const int64_t n_workers,
                             double *lo, double *hi) nogil


def hpd_intervals(samples, double alpha):
    if samples.ndim != 3:
        raise ValueError("Input must be a 3D array")
    if not (0 <= alpha < 1):
        raise ValueError("`alpha` must be in the interval (0, 1].")

    cdef int64_t n_par = samples.shape[2]
    cdef int64_t S = samples.shape[0] * samples.shape[1]

    if S < 1 or n_par == 0:
        raise ValueError(
            "Input samples must have at least one element and "
            "at least one parameter being sampled."
        )

    cdef int64_t n_workers = worker_count(n_par)
    cdef double[::1] pooled, lov, hiv

    pooled = np.ascontiguousarray(
        samples.transpose(2, 0, 1), dtype=np.float64
    ).reshape(-1)
    lo = np.empty(n_par, dtype=np.float64)
    hi = np.empty(n_par, dtype=np.float64)
    lov = lo
    hiv = hi

    with nogil:
        sdsge_hpd_intervals(&pooled[0], alpha, S, n_par, n_workers, &lov[0], &hiv[0])

    return lo, hi


def worker_count(object n_items):
    cpu_count = (
        os.process_cpu_count()
        if hasattr(os, "process_cpu_count")
        else os.cpu_count() or 1
    )
    if n_items is None:
        return cpu_count
    return min(n_items, cpu_count)
