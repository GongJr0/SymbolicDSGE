# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim mapping NumPy buffers to the pure-C core kernels.

No numeric logic here -- only the buffer->pointer marshalling and the GIL
release. The algorithms live in core.c. At the ABI level the kernels' f64/i64
are exactly double/int64_t, so the extern is declared with those.
"""

from libc.stdint cimport int64_t

from cpython.pycapsule cimport PyCapsule_GetName, PyCapsule_GetPointer

import numpy as np
import scipy.linalg.cython_lapack as _cython_lapack

cdef extern from "core.h" nogil:
    void sdsge_simulate_linear_states(
        const double *A, const double *B, const double *x0,
        const double *shock, double *out, int64_t T, int64_t n, int64_t k)
    void sdsge_affine_observations(
        const double *states, const double *C, const double *d,
        int64_t state_start, double *out, int64_t T, int64_t m, int64_t n)
    int64_t sdsge_simulate_second_order_pruned(
        const double *hx, const double *gx, const double *bx,
        const double *hxx, const double *gxx,
        const double *hss, const double *gss,
        const double *x0, const double *shock,
        int64_t T, int64_t nx, int64_t ny, int64_t n_exog,
        double *x_out, double *y_out)


cdef extern from "../_common/sdsge_complex.h":
    ctypedef struct c128:
        double re
        double im
    c128 c128_sqrt(c128 a)


cdef extern from "klein_postproc.h" nogil:
    int64_t klein_postproc(
        const c128 *s, const c128 *t, const c128 *z, int64_t n_s, int64_t n_cs,
        c128 *f, c128 *p, int64_t *stab, c128 *eig)


cdef extern from "klein_qz.h" nogil:
    # Opaque function-pointer alias; the real zgges signature lives in the
    # header. We only reinterpret the scipy cython_lapack ``zgges`` capsule
    # pointer to this type and hand it straight to the C routine.
    ctypedef void (*klein_zgges_fn)()
    int64_t c_klein_qz "klein_qz" (
        klein_zgges_fn zgges_ptr, int64_t n, c128 *s, c128 *t, c128 *z)
    int KLEIN_QZ_OK
    int KLEIN_QZ_ALLOC_FAIL
    int KLEIN_QZ_LAPACK_FAIL


# LAPACK ``zgges`` reached through its scipy ``cython_lapack`` capsule address
# (no build-time LAPACK link), cast to the C routine's expected pointer type
# once at import. This is the exact runtime-address mechanism the native
# estimation objective (#327) uses.
cdef object _zgges_capsule = _cython_lapack.__pyx_capi__["zgges"]
cdef klein_zgges_fn _zgges = <klein_zgges_fn>PyCapsule_GetPointer(
    _zgges_capsule, PyCapsule_GetName(_zgges_capsule)
)


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


# Measurement / observable-jacobian @cfunc (build_measurement_cfunc, real ABI):
# ``void(vars*, par*, out*)``. Held Python-side; called here by ``.address``, nogil.
ctypedef void (*sdsge_measurement_fn)(
    double *vars, double *par, double *out) noexcept nogil


def simulate_linear_states_into(A, B, x0, shock_mat, double[:, ::1] out):
    """out[(T+1, n)] <- linear state recursion. ``out`` is the caller's
    C-contiguous f64 output buffer, written in place; inputs are coerced."""
    cdef double[:, ::1] Av = np.ascontiguousarray(A, dtype=np.float64)
    cdef double[:, ::1] Bv = np.ascontiguousarray(B, dtype=np.float64)
    cdef double[::1] x0v = np.ascontiguousarray(x0, dtype=np.float64)
    cdef double[:, ::1] shockv = np.ascontiguousarray(shock_mat, dtype=np.float64)
    cdef int64_t n = Av.shape[0]
    cdef int64_t k = Bv.shape[1]
    cdef int64_t T = shockv.shape[0]
    # out[0] = x0 is written even when T == 0, so the C call always runs; only
    # the shock pointer can dangle on an empty (0, k) buffer.
    cdef const double *shock_ptr = &shockv[0, 0] if T > 0 else NULL
    with nogil:
        sdsge_simulate_linear_states(
            &Av[0, 0], &Bv[0, 0], &x0v[0], shock_ptr, &out[0, 0], T, n, k
        )


def affine_observations_into(states, C, d, int64_t state_start, double[:, ::1] out):
    """out[(T, m)] <- d + C @ states[state_start + t]. ``out`` is the caller's
    C-contiguous f64 output buffer, written in place; inputs are coerced."""
    cdef int64_t T = out.shape[0]
    if T == 0:
        return
    cdef double[:, ::1] statesv = np.ascontiguousarray(states, dtype=np.float64)
    cdef double[:, ::1] Cv = np.ascontiguousarray(C, dtype=np.float64)
    cdef double[::1] dv = np.ascontiguousarray(d, dtype=np.float64)
    cdef int64_t m = Cv.shape[0]
    cdef int64_t n = Cv.shape[1]
    with nogil:
        sdsge_affine_observations(
            &statesv[0, 0], &Cv[0, 0], &dv[0], state_start, &out[0, 0], T, m, n
        )


def simulate_second_order_pruned(hx, gx, bx, hxx, gxx, hss, gss, x0, shock_mat):
    """Pruned second order simulation. Returns the split state and jump paths."""
    cdef double[:, ::1] hxv = np.ascontiguousarray(hx, dtype=np.float64)
    cdef double[:, ::1] gxv = np.ascontiguousarray(gx, dtype=np.float64)
    cdef double[:, ::1] bxv = np.ascontiguousarray(bx, dtype=np.float64)
    cdef double[:, :, ::1] hxxv = np.ascontiguousarray(hxx, dtype=np.float64)
    cdef double[:, :, ::1] gxxv = np.ascontiguousarray(gxx, dtype=np.float64)
    cdef double[::1] hssv = np.ascontiguousarray(hss, dtype=np.float64)
    cdef double[::1] gssv = np.ascontiguousarray(gss, dtype=np.float64)
    cdef double[::1] x0v = np.ascontiguousarray(x0, dtype=np.float64)
    cdef double[:, ::1] shockv = np.ascontiguousarray(shock_mat, dtype=np.float64)

    cdef int64_t nx = hxv.shape[0]
    cdef int64_t ny = gxv.shape[0]
    cdef int64_t n_exog = bxv.shape[1]
    cdef int64_t T = shockv.shape[0]
    cdef int64_t err

    cdef const double *gx_ptr = NULL
    cdef const double *bx_ptr = NULL
    cdef const double *gxx_ptr = NULL
    cdef const double *gss_ptr = NULL
    cdef const double *shock_ptr = NULL
    cdef double *y_ptr = NULL
    cdef double[:, ::1] xoutv
    cdef double[:, ::1] youtv
    cdef double *x_ptr

    if nx <= 0:
        raise ValueError("simulate_second_order_pruned requires nx >= 1.")
    if hxv.shape[1] != nx:
        raise ValueError("hx must have shape (nx, nx).")
    if gxv.shape[1] != nx:
        raise ValueError("gx must have shape (ny, nx).")
    if bxv.shape[0] != nx:
        raise ValueError("bx must have shape (nx, n_exog).")
    if hxxv.shape[0] != nx or hxxv.shape[1] != nx or hxxv.shape[2] != nx:
        raise ValueError("hxx must have shape (nx, nx, nx).")
    if gxxv.shape[0] != ny or gxxv.shape[1] != nx or gxxv.shape[2] != nx:
        raise ValueError("gxx must have shape (ny, nx, nx).")
    if hssv.shape[0] != nx:
        raise ValueError("hss must have shape (nx,).")
    if gssv.shape[0] != ny:
        raise ValueError("gss must have shape (ny,).")
    if x0v.shape[0] != nx:
        raise ValueError("x0 must have shape (nx,).")
    if shockv.shape[1] != n_exog:
        raise ValueError("shock_mat must have shape (T, n_exog).")

    x_out = np.empty((T + 1, nx), dtype=np.float64)
    y_out = np.empty((T + 1, ny), dtype=np.float64)
    xoutv = x_out
    youtv = y_out
    x_ptr = &xoutv[0, 0]

    if ny > 0:
        gx_ptr = &gxv[0, 0]
        gxx_ptr = &gxxv[0, 0, 0]
        gss_ptr = &gssv[0]
        y_ptr = &youtv[0, 0]
    if n_exog > 0:
        bx_ptr = &bxv[0, 0]
        if T > 0:
            shock_ptr = &shockv[0, 0]

    with nogil:
        err = sdsge_simulate_second_order_pruned(
            &hxv[0, 0], gx_ptr, bx_ptr, &hxxv[0, 0, 0], gxx_ptr,
            &hssv[0], gss_ptr, &x0v[0], shock_ptr,
            T, nx, ny, n_exog, x_ptr, y_ptr)
    if err == -1:
        raise MemoryError("simulate_second_order_pruned: allocation failed.")
    if err != 0:
        raise RuntimeError(
            f"simulate_second_order_pruned: native kernel failed with code {err}."
        )
    return x_out, y_out


def klein_postprocess(s, t, z, int64_t n_states):
    """Klein Schur-to-solution post-proc. Returns ``(f, p, stab, eig)``.

    ``s``, ``t``, ``z`` are the ordered generalized-Schur factors (complex128,
    N x N). Mirrors the live path of ``_linearsolve._klein_postprocess``.
    """
    cdef double complex[:, ::1] sv = np.ascontiguousarray(s, dtype=np.complex128)
    cdef double complex[:, ::1] tv = np.ascontiguousarray(t, dtype=np.complex128)
    cdef double complex[:, ::1] zv = np.ascontiguousarray(z, dtype=np.complex128)
    cdef int64_t N = sv.shape[0]
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
            <c128 *>&sv[0, 0], <c128 *>&tv[0, 0], <c128 *>&zv[0, 0], n_s, n_cs,
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
    steady_state,
    params,
    int64_t n_eq,
    bint log_linear,
):
    """Complex-step first-order pencil ``(a, b)`` from a numba residual @cfunc
    (``build_cfunc``) given its ``.address``. Native twin of
    ``klein._approximate_system_numeric``; ``a``/``b`` feed ``scipy.ordqz``.
    ``a = d resid/d fwd``, ``b = -(d resid/d cur)``, each ``(n_eq, n_var)``.
    """
    cdef double[::1] ssv = np.ascontiguousarray(steady_state, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(params, dtype=np.float64)
    cdef int64_t n_var = ssv.shape[0]
    cdef int64_t n_par = parv.shape[0]

    a = np.empty((n_eq, n_var), dtype=np.float64)
    b = np.empty((n_eq, n_var), dtype=np.float64)
    cdef double[:, ::1] av = a
    cdef double[:, ::1] bv = b

    cdef const double *ss_ptr = &ssv[0] if n_var > 0 else NULL
    cdef const double *par_ptr = &parv[0] if n_par > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    cdef int64_t err
    with nogil:
        err = klein_preproc(
            resid, ss_ptr, par_ptr, n_var, n_par, n_eq, log_linear,
            &av[0, 0], &bv[0, 0])
    if err != 0:
        raise MemoryError("klein_preprocess: allocation failed.")
    return a, b


def klein_qz(a, b):
    """Native generalized Schur (QZ) with the Klein 'ouc' ordering, via LAPACK
    ``zgges`` (reached through the scipy ``cython_lapack`` capsule pointer, no
    build-time LAPACK link). Returns ``(s, t, z)`` == ``scipy.linalg.ordqz(a, b,
    sort="ouc", output="complex")`` indices ``[0, 1, 5]``: ordered Schur factors
    ``S``/``T`` and right Schur vectors ``Z``, ready for ``klein_postprocess``.

    Thin buffer-marshalling shim: the workspace query, ``zgges`` calls, and the
    'ouc' selctg all live in the C routine ``klein_qz`` (``klein_qz.c``), shared
    with the native estimation objective.
    """
    a_f = np.asfortranarray(a, dtype=np.complex128)
    b_f = np.asfortranarray(b, dtype=np.complex128)
    cdef int64_t n = a_f.shape[0]
    if a_f.shape[1] != n or b_f.shape[0] != n or b_f.shape[1] != n:
        raise ValueError("klein_qz requires square, identically shaped a and b.")
    if n == 0:
        return a_f, b_f, np.zeros((0, 0), dtype=np.complex128)

    # ``s``/``t`` are the pencil on input, overwritten in place to the ordered
    # Schur factors; ``z`` receives the right Schur vectors.
    z = np.zeros((n, n), dtype=np.complex128, order="F")
    cdef double complex[::1, :] av = a_f
    cdef double complex[::1, :] bv = b_f
    cdef double complex[::1, :] zv = z
    cdef int64_t status
    with nogil:
        status = c_klein_qz(
            _zgges, n,
            <c128 *>&av[0, 0], <c128 *>&bv[0, 0], <c128 *>&zv[0, 0])
    if status == KLEIN_QZ_ALLOC_FAIL:
        raise MemoryError("klein_qz: workspace allocation failed.")
    if status != KLEIN_QZ_OK:
        raise RuntimeError("klein_qz: LAPACK zgges failed.")
    return a_f, b_f, z


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
        raise MemoryError("solve_second_order: allocation failed.")
    if err == -2:
        raise ValueError("solve_second_order: singular symmetry-reduced system.")
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
        raise MemoryError("solve_second_order_risk: allocation failed.")
    if err == -2:
        raise ValueError("solve_second_order_risk: singular [Qg Qh] system.")
    return gss, hss


def residual_path(size_t residual_addr, cur_states, fwd_states, params, int64_t n_eq):
    """Real residual matrix ``(n_steps, n_eq)`` from a residual @cfunc
    (``build_cfunc``) evaluated over a simulated path. Native backend for the
    Den Haan-Marcet moment builder -- reuses the solve's cfunc, so it never
    triggers the numba residual compile. Inputs are coerced to contiguous
    complex128 here.
    """
    cdef double complex[:, ::1] curv = np.ascontiguousarray(
        cur_states, dtype=np.complex128)
    cdef double complex[:, ::1] fwdv = np.ascontiguousarray(
        fwd_states, dtype=np.complex128)
    cdef double complex[::1] parv = np.ascontiguousarray(
        params, dtype=np.complex128).reshape(-1)
    cdef int64_t n_steps = curv.shape[0]
    cdef int64_t n_var = curv.shape[1]
    residuals = np.empty((n_steps, n_eq), dtype=np.float64)
    cdef double[:, ::1] rv = residuals

    cdef c128 *cur_ptr = <c128 *>&curv[0, 0] if n_steps > 0 else NULL
    cdef c128 *fwd_ptr = <c128 *>&fwdv[0, 0] if n_steps > 0 else NULL
    cdef c128 *par_ptr = <c128 *>&parv[0] if parv.shape[0] > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    cdef int64_t err
    with nogil:
        err = sdsge_residual_path(
            resid, cur_ptr, fwd_ptr, par_ptr, n_steps, n_var, n_eq, &rv[0, 0])
    if err != 0:
        raise MemoryError("residual_path: allocation failed.")
    return residuals


def measurement_eval(size_t meas_addr, vars, par, int64_t n_obs):
    """Measurement vector ``h(vars, par)`` of length ``n_obs`` from a measurement
    @cfunc (``build_measurement_cfunc``) given its ``.address``. Single point;
    inputs are coerced to contiguous float64.
    """
    cdef double[::1] vv = np.ascontiguousarray(vars, dtype=np.float64)
    cdef double[::1] pv = np.ascontiguousarray(par, dtype=np.float64)
    out = np.empty((n_obs,), dtype=np.float64)
    cdef double[::1] ov = out

    cdef double *vars_ptr = &vv[0] if vv.shape[0] > 0 else NULL
    cdef double *par_ptr = &pv[0] if pv.shape[0] > 0 else NULL
    cdef double *out_ptr = &ov[0] if n_obs > 0 else NULL
    cdef sdsge_measurement_fn fn = <sdsge_measurement_fn><void*>meas_addr
    with nogil:
        fn(vars_ptr, par_ptr, out_ptr)
    return out


def jacobian_eval(size_t jac_addr, vars, par, int64_t n_obs, int64_t n_var):
    """Observable jacobian ``dh/dvars`` (n_obs, n_var) from a jacobian @cfunc
    (``build_measurement_cfunc`` over the flattened jacobian exprs) by ``.address``.
    Single point; the cfunc writes the row-major (obs, var) flat buffer.
    """
    cdef double[::1] vv = np.ascontiguousarray(vars, dtype=np.float64)
    cdef double[::1] pv = np.ascontiguousarray(par, dtype=np.float64)
    out = np.empty((n_obs, n_var), dtype=np.float64)
    cdef double[:, ::1] ov = out

    cdef double *vars_ptr = &vv[0] if vv.shape[0] > 0 else NULL
    cdef double *par_ptr = &pv[0] if pv.shape[0] > 0 else NULL
    cdef double *out_ptr = &ov[0, 0] if (n_obs * n_var) > 0 else NULL
    cdef sdsge_measurement_fn fn = <sdsge_measurement_fn><void*>jac_addr
    with nogil:
        fn(vars_ptr, par_ptr, out_ptr)
    return out


def measurement_path(size_t meas_addr, states, par, int64_t n_obs):
    """Measurement matrix ``(T, n_obs)`` from a measurement @cfunc over a state
    path. ``states`` is ``(T, n_var)`` in cur-variable order; coerced to
    contiguous float64.
    """
    cdef double[:, ::1] sv = np.ascontiguousarray(states, dtype=np.float64)
    cdef double[::1] pv = np.ascontiguousarray(par, dtype=np.float64)
    cdef int64_t T = sv.shape[0]
    out = np.empty((T, n_obs), dtype=np.float64)
    cdef double[:, ::1] ov = out

    cdef double *par_ptr = &pv[0] if pv.shape[0] > 0 else NULL
    cdef sdsge_measurement_fn fn = <sdsge_measurement_fn><void*>meas_addr
    cdef int64_t tt
    if n_obs > 0 and T > 0:
        with nogil:
            for tt in range(T):
                fn(&sv[tt, 0], par_ptr, &ov[tt, 0])
    return out


def residual_eval(size_t residual_addr, fwd, cur, params, int64_t n_eq):
    """Complex residual vector ``F(fwd, cur, par)`` of length ``n_eq`` from a
    residual @cfunc (``build_cfunc``) given its ``.address``. Single-point native
    evaluation -- the path ``CompiledModel.equations`` takes instead of the numba
    vector kernel. Inputs are coerced to contiguous complex128 here.
    """
    cdef double complex[::1] fwdv = np.ascontiguousarray(
        fwd, dtype=np.complex128).reshape(-1)
    cdef double complex[::1] curv = np.ascontiguousarray(
        cur, dtype=np.complex128).reshape(-1)
    cdef double complex[::1] parv = np.ascontiguousarray(
        params, dtype=np.complex128).reshape(-1)
    out = np.empty((n_eq,), dtype=np.complex128)
    cdef double complex[::1] ov = out

    cdef c128 *fwd_ptr = <c128 *>&fwdv[0] if fwdv.shape[0] > 0 else NULL
    cdef c128 *cur_ptr = <c128 *>&curv[0] if curv.shape[0] > 0 else NULL
    cdef c128 *par_ptr = <c128 *>&parv[0] if parv.shape[0] > 0 else NULL
    cdef c128 *out_ptr = <c128 *>&ov[0] if n_eq > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    with nogil:
        resid(fwd_ptr, cur_ptr, par_ptr, out_ptr)
    return out


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
