# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim mapping NumPy buffers to the pure-C core kernels.

No numeric logic here -- only the buffer->pointer marshalling and the GIL
release. The algorithms live in core.c. At the ABI level the kernels' f64/i64
are exactly double/int64_t, so the extern is declared with those.
"""

from libc.stdint cimport int64_t

cdef extern from "core.h" nogil:
    void sdsge_simulate_linear_states(
        const double *A, const double *B, const double *x0,
        const double *shock, double *out, int64_t T, int64_t n, int64_t k)
    void sdsge_affine_observations(
        const double *states, const double *C, const double *d,
        int64_t state_start, double *out, int64_t T, int64_t m, int64_t n)


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
