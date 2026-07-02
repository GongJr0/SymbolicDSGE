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


cdef extern from "../_common/sdsge_bicomplex.h" nogil:
    ctypedef struct bc256:
        c128 a
        c128 b
    bc256 bc256_add(bc256 x, bc256 y)
    bc256 bc256_sub(bc256 x, bc256 y)
    bc256 bc256_neg(bc256 x)
    bc256 bc256_mul(bc256 x, bc256 y)
    bc256 bc256_div(bc256 x, bc256 y)
    bc256 bc256_real_scale(bc256 x, double s)
    bc256 bc256_i_conj(bc256 x)
    bc256 bc256_j_conj(bc256 x)
    bc256 bc256_conj(bc256 x)
    bc256 bc256_exp(bc256 x)
    bc256 bc256_log(bc256 x)
    bc256 bc256_spow(bc256 x, double p)
    bc256 bc256_ipow(bc256 x, int64_t p)
    bc256 bc256_cpow(bc256 x, bc256 y)
    double bc256_real(bc256 x)
    double bc256_i(bc256 x)
    double bc256_j(bc256 x)
    double bc256_ij(bc256 x)
    void bc256_proj(bc256 x, c128 *p1, c128 *p2)
    bc256 bc256_reconst(c128 a, c128 b)


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


# --- bicomplex (bc256) primitive wrappers -------------------------------------
# Scalar-arithmetic surface backing the second-order (bicomplex-step) perturbation
# preproc. Exposed to Python only so the parity/derivative tests can exercise the
# `static inline` ops in sdsge_bicomplex.h against a reference; the native driver
# calls these in C, not through here. A bc256 crosses the boundary as the 4-tuple
# (real, i, j, ij) = (a.re, a.im, b.re, b.im).

cdef bc256 _bc_pack(x):
    cdef bc256 v
    v.a.re = x[0]
    v.a.im = x[1]
    v.b.re = x[2]
    v.b.im = x[3]
    return v


cdef tuple _bc_unpack(bc256 v):
    return (v.a.re, v.a.im, v.b.re, v.b.im)


def bc_add(x, y):
    return _bc_unpack(bc256_add(_bc_pack(x), _bc_pack(y)))


def bc_sub(x, y):
    return _bc_unpack(bc256_sub(_bc_pack(x), _bc_pack(y)))


def bc_neg(x):
    return _bc_unpack(bc256_neg(_bc_pack(x)))


def bc_mul(x, y):
    return _bc_unpack(bc256_mul(_bc_pack(x), _bc_pack(y)))


def bc_div(x, y):
    return _bc_unpack(bc256_div(_bc_pack(x), _bc_pack(y)))


def bc_real_scale(x, double s):
    return _bc_unpack(bc256_real_scale(_bc_pack(x), s))


def bc_i_conj(x):
    return _bc_unpack(bc256_i_conj(_bc_pack(x)))


def bc_j_conj(x):
    return _bc_unpack(bc256_j_conj(_bc_pack(x)))


def bc_conj(x):
    return _bc_unpack(bc256_conj(_bc_pack(x)))


def bc_exp(x):
    return _bc_unpack(bc256_exp(_bc_pack(x)))


def bc_log(x):
    return _bc_unpack(bc256_log(_bc_pack(x)))


def bc_spow(x, double p):
    return _bc_unpack(bc256_spow(_bc_pack(x), p))


def bc_ipow(x, int64_t p):
    return _bc_unpack(bc256_ipow(_bx_pack(x), p))


def bc_cpow(x, y):
    return _bc_unpack(bc256_cpow(_bc_pack(x), _bc_pack(y)))


def bc_accessors(x):
    """(real, i, j, ij) read back through the C f64 accessors."""
    cdef bc256 v = _bc_pack(x)
    return (bc256_real(v), bc256_i(v), bc256_j(v), bc256_ij(v))


def bc_proj(x):
    """Idempotent projection -> (p1.re, p1.im, p2.re, p2.im)."""
    cdef bc256 v = _bc_pack(x)
    cdef c128 p1
    cdef c128 p2
    bc256_proj(v, &p1, &p2)
    return (p1.re, p1.im, p2.re, p2.im)


def bc_reconst(p):
    """Inverse of bc_proj; p = (p1.re, p1.im, p2.re, p2.im) -> bc256 4-tuple."""
    cdef c128 a
    cdef c128 b
    a.re = p[0]
    a.im = p[1]
    b.re = p[2]
    b.im = p[3]
    return _bc_unpack(bc256_reconst(a, b))
