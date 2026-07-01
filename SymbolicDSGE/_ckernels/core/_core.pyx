# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim mapping NumPy buffers to the pure-C core kernels.

No numeric logic here -- only the buffer->pointer marshalling and the GIL
release. The algorithms live in core.c. At the ABI level the kernels' f64/i64
are exactly double/int64_t, so the extern is declared with those.
"""

from libc.stdint cimport int64_t

import numpy as np

cdef extern from "core.h" nogil:
    void sdsge_simulate_linear_states(
        const double *A, const double *B, const double *x0,
        const double *shock, double *out, int64_t T, int64_t n, int64_t k)
    void sdsge_affine_observations(
        const double *states, const double *C, const double *d,
        int64_t state_start, double *out, int64_t T, int64_t m, int64_t n)


cdef extern from "../_common/sdsge_complex.h":
    ctypedef struct c128:
        double re
        double im


cdef extern from "klein_postproc.h" nogil:
    int64_t klein_postproc(
        const c128 *s, const c128 *t, const c128 *z, int64_t n_s, int64_t n_cs,
        c128 *f, c128 *p, int64_t *stab, c128 *eig)


def simulate_linear_states_into(
    double[:, ::1] A,
    double[:, ::1] B,
    double[::1] x0,
    double[:, ::1] shock_mat,
    double[:, ::1] out,
):
    """out[(T+1, n)] <- linear state recursion. Mirrors the numba kernel."""
    cdef int64_t n = A.shape[0]
    cdef int64_t k = B.shape[1]
    cdef int64_t T = shock_mat.shape[0]
    # out[0] = x0 is written even when T == 0, so the C call always runs; only
    # the shock pointer can dangle on an empty (0, k) buffer.
    cdef const double *shock_ptr = &shock_mat[0, 0] if T > 0 else NULL
    with nogil:
        sdsge_simulate_linear_states(
            &A[0, 0], &B[0, 0], &x0[0], shock_ptr, &out[0, 0], T, n, k
        )


def affine_observations_into(
    double[:, ::1] states,
    double[:, ::1] C,
    double[::1] d,
    int64_t state_start,
    double[:, ::1] out,
):
    """out[(T, m)] <- d + C @ states[state_start + t]. Mirrors the numba kernel."""
    cdef int64_t T = out.shape[0]
    if T == 0:
        return
    cdef int64_t m = C.shape[0]
    cdef int64_t n = C.shape[1]
    with nogil:
        sdsge_affine_observations(
            &states[0, 0], &C[0, 0], &d[0], state_start, &out[0, 0], T, m, n
        )


def klein_postprocess(
    double complex[:, ::1] s,
    double complex[:, ::1] t,
    double complex[:, ::1] z,
    int64_t n_states,
):
    """Klein Schur-to-solution post-proc. Returns ``(f, p, stab, eig)``.

    ``s``, ``t``, ``z`` are the ordered generalized-Schur factors (complex128,
    N x N). Mirrors the live path of ``_linearsolve._klein_postprocess``.
    """
    cdef int64_t N = s.shape[0]
    cdef int64_t n_s = n_states
    cdef int64_t n_cs = N - n_s
    if n_s <= 0:
        raise ValueError("klein_postprocess requires n_states >= 1.")
    if n_s > N:
        raise ValueError("n_states exceeds the matrix dimension.")

    f = np.empty((n_cs, n_s), dtype=np.complex128)
    p = np.empty((n_s, n_s), dtype=np.complex128)
    eig = np.empty(N, dtype=np.complex128)
    cdef double complex[:, ::1] fv = f
    cdef double complex[:, ::1] pv = p
    cdef double complex[::1] ev = eig
    cdef int64_t stab = 0
    cdef int64_t err
    with nogil:
        err = klein_postproc(
            <c128 *>&s[0, 0], <c128 *>&t[0, 0], <c128 *>&z[0, 0], n_s, n_cs,
            <c128 *>&fv[0, 0] if n_cs > 0 else NULL,
            <c128 *>&pv[0, 0], &stab, <c128 *>&ev[0])
    if err == -1:
        raise MemoryError("klein_postprocess: allocation failed.")
    if err == -2:
        raise ValueError(
            "klein_postprocess: singular z11/s11 (Blanchard-Kahn failure)."
        )
    if err == -3:
        raise ValueError("klein_postprocess: model has no states.")
    return f, p, int(stab), eig
