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
    c128 c128_sqrt(c128 a)


cdef extern from "klein_postproc.h" nogil:
    int64_t klein_postproc(
        const c128 *s, const c128 *t, const c128 *z, int64_t n_s, int64_t n_cs,
        c128 *f, c128 *p, int64_t *stab, c128 *eig)


cdef extern from "spike.h" nogil:
    ctypedef void (*spike_residual_fn)(
        c128 *a, c128 *b, c128 *out, int64_t n)
    void spike_call(
        spike_residual_fn fn, c128 *a, c128 *b, c128 *out, int64_t n)


cdef extern from "klein_preproc.h" nogil:
    ctypedef void (*sdsge_residual_fn)(
        c128 *fwd, c128 *cur, c128 *par, c128 *out)
    int64_t klein_preproc(
        sdsge_residual_fn resid, const double *ss, const double *par,
        int64_t n_var, int64_t n_par, int64_t n_eq, int64_t log_linear,
        double *a, double *b)


cdef extern from "residual_path.h" nogil:
    int64_t sdsge_residual_path(
        sdsge_residual_fn resid, const c128 *cur, const c128 *fwd,
        const c128 *par, int64_t n_steps, int64_t n_var, int64_t n_eq,
        double *residuals)


cdef extern from "steady_state.h" nogil:
    int64_t sdsge_steady_state_newton(
        sdsge_residual_fn residual, const double *seed, const double *par,
        int64_t n_var, int64_t n_par, int64_t max_iter, double tol,
        double *ss, int64_t *iters)


cdef extern from "second_order.h" nogil:
    int64_t sdsge_second_order(
        const double *a, const double *b, const double *f_xx,
        const double *gx, const double *hx, int64_t n, int64_t nx,
        double *gxx, double *hxx)
    int64_t sdsge_second_order_risk(
        const double *a, const double *b, const double *f_xx,
        const double *gx, const double *gxx, const double *eta,
        int64_t n, int64_t nx, int64_t ne, double *gss, double *hss)


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
    bc256 bc256_sqrt(bc256 x)
    bc256 bc256_cpow(bc256 x, bc256 y)
    double bc256_real(bc256 x)
    double bc256_i(bc256 x)
    double bc256_j(bc256 x)
    double bc256_ij(bc256 x)
    void bc256_proj(bc256 x, c128 *p1, c128 *p2)
    bc256 bc256_reconst(c128 a, c128 b)


cdef extern from "bicomplex_hessian.h" nogil:
    ctypedef void (*bc_residual_fn)(
        const bc256 *fwd, const bc256 *cur, const bc256 *par, bc256 *out)
    int64_t sdsge_bicomplex_hessian(
        bc_residual_fn residual, const double *ss, const double *par,
        int64_t n_var, int64_t n_par, int64_t n_eq, double step,
        double *hessian)


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


def spike_drive(
    size_t fn_addr,
    double complex[::1] a,
    double complex[::1] b,
    double complex[::1] out,
):
    """Stage-0 (#248): invoke a numba @cfunc (given its ``.address``) from the
    hand-written C ``spike_call``, GIL released. ``fn(a, b, out, n)`` writes into
    ``out``. Correct results here prove the numba->native ABI + nogil path on MSVC.
    """
    cdef int64_t n = a.shape[0]
    cdef spike_residual_fn fn = <spike_residual_fn><void*>fn_addr
    with nogil:
        spike_call(fn, <c128 *>&a[0], <c128 *>&b[0], <c128 *>&out[0], n)


def klein_preprocess(
    size_t residual_addr,
    double[::1] steady_state,
    double[::1] params,
    int64_t n_eq,
    bint log_linear,
):
    """Complex-step first-order pencil ``(a, b)`` from a numba residual @cfunc
    (``build_cfunc``) given its ``.address``. Native twin of
    ``klein._approximate_system_numeric``; ``a``/``b`` feed ``scipy.ordqz``.
    ``a = d resid/d fwd``, ``b = -(d resid/d cur)``, each ``(n_eq, n_var)``.
    """
    cdef int64_t n_var = steady_state.shape[0]
    cdef int64_t n_par = params.shape[0]

    a = np.empty((n_eq, n_var), dtype=np.float64)
    b = np.empty((n_eq, n_var), dtype=np.float64)
    cdef double[:, ::1] av = a
    cdef double[:, ::1] bv = b

    cdef const double *ss_ptr = &steady_state[0] if n_var > 0 else NULL
    cdef const double *par_ptr = &params[0] if n_par > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    cdef int64_t err
    with nogil:
        err = klein_preproc(
            resid, ss_ptr, par_ptr, n_var, n_par, n_eq, log_linear,
            &av[0, 0], &bv[0, 0])
    if err != 0:
        raise MemoryError("klein_preprocess: allocation failed.")
    return a, b


def steady_state_newton(
    size_t residual_addr,
    double[::1] seed,
    double[::1] params,
    int64_t max_iter=50,
    double tol=1e-12,
):
    """Newton solve of ``F(ss, ss) = 0`` from ``seed``, driving a numba residual
    @cfunc (``build_cfunc``) by its ``.address``. The Jacobian ``a - b`` comes from
    ``klein_preproc`` each step; the update is ``sdsge_solve``. Returns
    ``(ss, iters)``; raises on singular Jacobian or non-convergence.
    """
    cdef int64_t n_var = seed.shape[0]
    cdef int64_t n_par = params.shape[0]

    ss = np.empty(n_var, dtype=np.float64)
    cdef double[::1] ssv = ss

    cdef const double *seed_ptr = &seed[0] if n_var > 0 else NULL
    cdef const double *par_ptr = &params[0] if n_par > 0 else NULL
    cdef double *ss_ptr = &ssv[0] if n_var > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    cdef int64_t iters = 0
    cdef int64_t err
    with nogil:
        err = sdsge_steady_state_newton(
            resid, seed_ptr, par_ptr, n_var, n_par, max_iter, tol,
            ss_ptr, &iters)
    if err == -1:
        raise MemoryError("steady_state_newton: allocation failed.")
    if err == -2:
        raise ValueError("steady_state_newton: singular Jacobian (a - b).")
    if err == -3:
        raise ValueError(
            "steady_state_newton: did not converge within max_iter "
            "(or the residual went non-finite)."
        )
    return ss, int(iters)


def second_order(a, b, f_xx, gx, hx, int64_t n_state):
    """SGU second-order tensors ``(gxx, hxx)`` -- native transcription of
    ``core.second_order.solve_second_order``. ``a``/``b`` are the first-order
    pencil ``(n, n)``, ``f_xx`` the residual Hessian ``(n, 2n, 2n)``, ``gx``
    ``(ny, nx)``, ``hx`` ``(nx, nx)``. Returns ``gxx (ny, nx, nx)``,
    ``hxx (nx, nx, nx)``.

    Inputs are coerced to C-contiguous f64 (``gx``/``hx`` arrive as real-part
    views of the complex Klein solution, so a copy is expected here).
    """
    cdef double[:, ::1] av = np.ascontiguousarray(a, dtype=np.float64)
    cdef double[:, ::1] bv = np.ascontiguousarray(b, dtype=np.float64)
    cdef double[:, :, ::1] fxxv = np.ascontiguousarray(f_xx, dtype=np.float64)
    cdef double[:, ::1] gxv = np.ascontiguousarray(gx, dtype=np.float64)
    cdef double[:, ::1] hxv = np.ascontiguousarray(hx, dtype=np.float64)

    cdef int64_t n = av.shape[0]
    cdef int64_t nx = n_state
    cdef int64_t ny = n - nx

    gxx = np.empty((ny, nx, nx), dtype=np.float64)
    hxx = np.empty((nx, nx, nx), dtype=np.float64)
    cdef double[:, :, ::1] gv = gxx
    cdef double[:, :, ::1] hv = hxx

    cdef const double *gx_ptr = &gxv[0, 0] if ny > 0 else NULL
    cdef double *gv_ptr = &gv[0, 0, 0] if ny > 0 else NULL
    cdef int64_t err
    with nogil:
        err = sdsge_second_order(
            &av[0, 0], &bv[0, 0], &fxxv[0, 0, 0], gx_ptr, &hxv[0, 0], n, nx,
            gv_ptr, &hv[0, 0, 0])
    if err == -1:
        raise MemoryError("second_order: allocation failed.")
    if err == -2:
        raise ValueError("second_order: singular symmetry-reduced system.")
    return gxx, hxx


def second_order_risk(a, b, f_xx, gx, gxx, eta, int64_t n_state):
    """Sigma^2 risk correction ``(gss, hss)`` -- native transcription of
    ``core.second_order.solve_second_order_risk``. ``gxx`` is the second-order
    controls ``(ny, nx, nx)``; ``eta`` the shock loading ``(nx, ne)``. Returns
    ``gss (ny,)``, ``hss (nx,)``. Inputs coerced to C-contiguous f64.
    """
    cdef double[:, ::1] av = np.ascontiguousarray(a, dtype=np.float64)
    cdef double[:, ::1] bv = np.ascontiguousarray(b, dtype=np.float64)
    cdef double[:, :, ::1] fxxv = np.ascontiguousarray(f_xx, dtype=np.float64)
    cdef double[:, ::1] gxv = np.ascontiguousarray(gx, dtype=np.float64)
    cdef double[:, :, ::1] gxxv = np.ascontiguousarray(gxx, dtype=np.float64)
    cdef double[:, ::1] etav = np.ascontiguousarray(eta, dtype=np.float64)

    cdef int64_t n = av.shape[0]
    cdef int64_t nx = n_state
    cdef int64_t ny = n - nx
    cdef int64_t ne = etav.shape[1]

    gss = np.empty(ny, dtype=np.float64)
    hss = np.empty(nx, dtype=np.float64)
    cdef double[::1] gssv = gss
    cdef double[::1] hssv = hss

    cdef const double *gx_ptr = &gxv[0, 0] if ny > 0 else NULL
    cdef const double *gxx_ptr = &gxxv[0, 0, 0] if ny > 0 else NULL
    cdef const double *eta_ptr = &etav[0, 0] if ne > 0 else NULL
    cdef double *gss_ptr = &gssv[0] if ny > 0 else NULL
    cdef int64_t err
    with nogil:
        err = sdsge_second_order_risk(
            &av[0, 0], &bv[0, 0], &fxxv[0, 0, 0], gx_ptr, gxx_ptr, eta_ptr,
            n, nx, ne, gss_ptr, &hssv[0])
    if err == -1:
        raise MemoryError("second_order_risk: allocation failed.")
    if err == -2:
        raise ValueError("second_order_risk: singular [Qg Qh] system.")
    return gss, hss


def residual_path(
    size_t residual_addr,
    double complex[:, ::1] cur_states,
    double complex[:, ::1] fwd_states,
    double complex[::1] params,
    int64_t n_eq,
):
    """Real residual matrix ``(n_steps, n_eq)`` from a residual @cfunc
    (``build_cfunc``) evaluated over a simulated path. Native backend for the
    Den Haan-Marcet moment builder -- reuses the solve's cfunc, so it never
    triggers the numba residual compile.
    """
    cdef int64_t n_steps = cur_states.shape[0]
    cdef int64_t n_var = cur_states.shape[1]
    residuals = np.empty((n_steps, n_eq), dtype=np.float64)
    cdef double[:, ::1] rv = residuals

    cdef c128 *cur_ptr = <c128 *>&cur_states[0, 0] if n_steps > 0 else NULL
    cdef c128 *fwd_ptr = <c128 *>&fwd_states[0, 0] if n_steps > 0 else NULL
    cdef c128 *par_ptr = <c128 *>&params[0] if params.shape[0] > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    cdef int64_t err
    with nogil:
        err = sdsge_residual_path(
            resid, cur_ptr, fwd_ptr, par_ptr, n_steps, n_var, n_eq, &rv[0, 0])
    if err != 0:
        raise MemoryError("residual_path: allocation failed.")
    return residuals


def bicomplex_hessian(
    size_t residual_addr,
    double[::1] steady_state,
    double[::1] params,
    int64_t n_eq,
    double step=1e-4,
):
    """Residual Hessian ``F_xx`` (n_eq, 2*n_var, 2*n_var) via the bicomplex step,
    from a bicomplex residual @cfunc (``build_cfunc(..., BicomplexOps())``) given
    its ``.address``. Second-order native preproc; feeds the g_xx assembly.
    """
    cdef int64_t n_var = steady_state.shape[0]
    cdef int64_t n_par = params.shape[0]
    cdef int64_t n2 = 2 * n_var

    hessian = np.empty((n_eq, n2, n2), dtype=np.float64)
    cdef double[:, :, ::1] hv = hessian

    cdef const double *ss_ptr = &steady_state[0] if n_var > 0 else NULL
    cdef const double *par_ptr = &params[0] if n_par > 0 else NULL
    cdef bc_residual_fn residual = <bc_residual_fn><void*>residual_addr
    cdef int64_t err
    with nogil:
        err = sdsge_bicomplex_hessian(
            residual, ss_ptr, par_ptr, n_var, n_par, n_eq, step, &hv[0, 0, 0])
    if err != 0:
        raise MemoryError("bicomplex_hessian: allocation failed.")
    return hessian


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
    return _bc_unpack(bc256_ipow(_bc_pack(x), p))


def bc_sqrt(x):
    return _bc_unpack(bc256_sqrt(_bc_pack(x)))


def c_sqrt(z):
    """Principal complex sqrt; z = (re, im) -> (re, im)."""
    cdef c128 v
    v.re = z[0]
    v.im = z[1]
    cdef c128 r = c128_sqrt(v)
    return (r.re, r.im)


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
