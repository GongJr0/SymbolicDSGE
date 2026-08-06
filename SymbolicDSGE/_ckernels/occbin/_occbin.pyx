# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim mapping NumPy buffers to the pure-C OccBin kernels.

Buffer to pointer marshalling and the GIL release only; the algorithms live in
occbin.c. The kernels' f64/i8/i64 are exactly double/int8_t/int64_t, so the
externs are declared with those.
"""

from libc.stdint cimport int8_t, int64_t

import numpy as np


cdef extern from "occbin.h" nogil:
    ctypedef void (*sdsge_constraint_fn)(
        double *cur, double *par, int8_t *flags) noexcept
    int64_t sdsge_constraint_path(
        sdsge_constraint_fn cond, double *path, double *par,
        const int8_t *regime_in, int8_t *regime_out,
        int64_t T, int64_t n_var, int64_t n_constraint)


#: Constraints the native flag buffer is sized for (``i8 flags[4]`` in occbin.c,
#: two slots per constraint). Mirrors the OccBin cap in model_parser.
MAX_CONSTRAINTS = 2


def constraint_path(size_t cond_addr, path, par, regime_in,
                    int64_t n_constraint, out=None):
    """Latched regime mask ``(T,)`` from a constraint @cfunc over a path.

    ``path`` is ``(T, n_var)`` in cur-variable order and in levels, not
    deviations from the reference steady state. ``regime_in`` is the incoming
    ``(T,)`` mask over declaration-ordered constraints, which the latch carries
    across guess-and-verify iterations at a fixed period; periods never interact.

    ``out`` is an optional C-contiguous int8 buffer written in place, and may be
    ``regime_in`` itself to latch in place. Other inputs are coerced. Returns
    ``(out, changed)``, where ``changed`` counts the periods whose mask moved and
    so is 0 exactly at a fixed point.
    """
    if not 0 < n_constraint <= MAX_CONSTRAINTS:
        raise ValueError(
            f"n_constraint must be in 1..{MAX_CONSTRAINTS}, got {n_constraint}."
        )

    cdef double[:, ::1] pv = np.ascontiguousarray(path, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(par, dtype=np.float64)
    cdef const int8_t[::1] inv = np.ascontiguousarray(regime_in, dtype=np.int8)
    cdef int64_t T = pv.shape[0]
    cdef int64_t n_var = pv.shape[1]
    cdef int8_t[::1] ov
    cdef int64_t changed

    if inv.shape[0] != T:
        raise ValueError(
            f"regime_in has length {inv.shape[0]}, expected {T} to match path."
        )
    if out is None:
        out = np.empty((T,), dtype=np.int8)
    ov = out
    if ov.shape[0] != T:
        raise ValueError(
            f"out has length {ov.shape[0]}, expected {T} to match path."
        )

    cdef double *path_ptr = &pv[0, 0] if (T * n_var) > 0 else NULL
    cdef double *par_ptr = &parv[0] if parv.shape[0] > 0 else NULL
    cdef const int8_t *in_ptr = &inv[0] if T > 0 else NULL
    cdef int8_t *out_ptr = &ov[0] if T > 0 else NULL
    cdef sdsge_constraint_fn fn = <sdsge_constraint_fn><void*>cond_addr
    with nogil:
        changed = sdsge_constraint_path(
            fn, path_ptr, par_ptr, in_ptr, out_ptr, T, n_var, n_constraint
        )
    return out, changed
