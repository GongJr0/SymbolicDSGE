# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim mapping NumPy buffers to the pure-C core kernels.

Buffer to pointer marshalling and the GIL release only; the algorithms live in
core.c. The kernels' f64/i64 are exactly double/int64_t, so the externs are
declared with those.
"""

from libc.stdint cimport int64_t

from cpython.pycapsule cimport PyCapsule_GetName, PyCapsule_GetPointer

import numpy as np
import scipy.linalg.cython_lapack as _cython_lapack


cdef extern from "../_common/sdsge_complex.h":
    ctypedef struct c128:
        double re
        double im
    c128 c128_sqrt(c128 a)

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

cdef extern from "core.h" nogil:
    int SDSGE_CORE_ALLOC_FAIL
    ctypedef void (*sdsge_measurement_fn)(
        double *vars, double *par, double *out) noexcept
    void sdsge_assemble_transition(
        const double *p, const double *f, const int64_t n_state,
        int64_t n_control, double *A)
    void sdsge_simulate_linear_states(
        const double *A, const double *B, const double *x0,
        const double *shock, double *out, int64_t T, int64_t n, int64_t k)
    void sdsge_affine_observations(
        const double *states, const double *C, const double *d,
        double *out, int64_t T, int64_t m, int64_t n)
    int64_t sdsge_simulate_second_order_pruned(
        const double *hx, const double *gx, const double *bx,
        const double *hxx, const double *gxx,
        const double *hss, const double *gss,
        const double *x0, const double *shock,
        int64_t T, int64_t nx, int64_t ny, int64_t n_exog,
        double *out)

cdef extern from "../_common/sdsge_common.h" nogil:
    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int

cdef extern from "sdsge_linalg.h":
    int sdsge_chol(const double *S, double jitter, double *L, int64_t n) nogil

cdef extern from "bicomplex_hessian.h" nogil:
    ctypedef void (*bc_residual_fn)(
        const bc256 *fwd, const bc256 *cur, const bc256 *prev, const bc256 *eps,
        const bc256 *par, bc256 *out)
    arena_size sdsge_bicomplex_hessian_arena_size(
        int64_t n_var, int64_t n_par, int64_t n_exog, int64_t n_eq)
    void sdsge_bicomplex_hessian(
        bc_residual_fn residual, const double *ss, const double *par,
        int64_t n_var, int64_t n_par, int64_t n_exog, int64_t n_eq,
        double *hessian, double *arena)

cdef extern from "pencil.h" nogil:
    ctypedef void (*sdsge_dgeqrf_fn)()
    ctypedef void (*sdsge_dormqr_fn)()
    int SDSGE_INC_LAG
    int SDSGE_INC_CUR
    int SDSGE_INC_LEAD
    int64_t sdsge_pencil_dim(const signed char *incidence, int64_t n_var)


# The incidence bits are a wire format between the compiler and the solve, so
# they are re-exported rather than restated on the Python side.
INC_LAG = SDSGE_INC_LAG
INC_CUR = SDSGE_INC_CUR
INC_LEAD = SDSGE_INC_LEAD


cdef inline int64_t _nspred(signed char[::1] incidence) noexcept:
    """Variables occurring at t-1: the predetermined count the pencil splits on."""
    cdef int64_t k, out = 0
    for k in range(incidence.shape[0]):
        if incidence[k] & SDSGE_INC_LAG:
            out += 1
    return out


def pencil_dim(incidence, int64_t n_var):
    """Size of the pencil an incidence implies, ``ndynamic + n_both``.

    ``n_var`` does not bound it: a variable carrying both a lag and a lead needs
    a companion row. Callers own the Schur and eigenvalue buffers, so they size
    them from here."""
    cdef signed char[::1] incv = np.ascontiguousarray(incidence, dtype=np.int8)
    return int(sdsge_pencil_dim(&incv[0], n_var))


cdef extern from "klein_preproc.h" nogil:
    ctypedef void (*sdsge_residual_fn)(
        c128 *fwd, c128 *cur, c128 *prev, c128 *eps, c128 *par, c128 *out)
    arena_size klein_preproc_arena_size(
        int64_t n_var, int64_t n_par, int64_t n_exog, int64_t n_eq)
    void klein_preproc(
        sdsge_residual_fn resid, const double *ss, const double *par,
        int64_t n_var, int64_t n_par, int64_t n_exog, int64_t n_eq,
        double *a, double *b, double *c, double *d, double *arena)

cdef extern from "klein_postproc.h" nogil:
    arena_size klein_postproc_arena_size(int64_t n_s, int64_t n_cs)
    int64_t klein_postproc(
        const c128 *s, const c128 *t, const c128 *z, int64_t n_s, int64_t n_cs,
        c128 *f, c128 *p, int64_t *stab, c128 *eig, double *arena,
        int64_t *iarena)
    int SDSGE_KLEIN_POSTPROC_SINGULAR
    int SDSGE_KLEIN_POSTPROC_INVALID


cdef extern from "klein_qz.h" nogil:
    # Opaque function-pointer alias; the real zgges signature lives in the
    # header. We only reinterpret the scipy cython_lapack ``zgges`` capsule
    # pointer to this type and hand it straight to the C routine.
    ctypedef void (*klein_zgges_fn)()
    arena_size klein_qz_arena_size(int64_t n)
    int64_t c_klein_qz "klein_qz" (
        klein_zgges_fn zgges_ptr, int64_t n, c128 *s, c128 *t, c128 *z,
        double *arena, int64_t *iarena)
    int KLEIN_QZ_OK
    int KLEIN_QZ_LAPACK_FAIL


# LAPACK ``zgges`` reached through its scipy ``cython_lapack`` capsule address
# (no build-time LAPACK link), cast to the C routine's expected pointer type
# once at import. This is the exact runtime-address mechanism the native
# estimation objective (#327) uses.
cdef object _zgges_capsule = _cython_lapack.__pyx_capi__["zgges"]
cdef klein_zgges_fn _zgges = <klein_zgges_fn>PyCapsule_GetPointer(
    _zgges_capsule, PyCapsule_GetName(_zgges_capsule)
)

# The static rotation's QR, reached the same way. `Q` is never formed: dgeqrf
# leaves the reflectors in place and dormqr applies Q' straight to each block.
cdef object _dgeqrf_capsule = _cython_lapack.__pyx_capi__["dgeqrf"]
cdef sdsge_dgeqrf_fn _dgeqrf = <sdsge_dgeqrf_fn>PyCapsule_GetPointer(
    _dgeqrf_capsule, PyCapsule_GetName(_dgeqrf_capsule)
)
cdef object _dormqr_capsule = _cython_lapack.__pyx_capi__["dormqr"]
cdef sdsge_dormqr_fn _dormqr = <sdsge_dormqr_fn>PyCapsule_GetPointer(
    _dormqr_capsule, PyCapsule_GetName(_dormqr_capsule)
)


cdef extern from "spike.h" nogil:
    ctypedef void (*spike_residual_fn)(
        c128 *a, c128 *b, c128 *out, int64_t n)
    void spike_call(
        spike_residual_fn fn, c128 *a, c128 *b, c128 *out, int64_t n)

cdef extern from "residual_path.h" nogil:
    int64_t sdsge_residual_path(
        sdsge_residual_fn resid, const c128 *cur, const c128 *fwd,
        const c128 *prev, const c128 *eps, const c128 *par, int64_t n_steps,
        int64_t n_var, int64_t n_exog, int64_t n_eq, double *residuals)

cdef extern from "klein_solve.h" nogil:
    ctypedef struct klein_spec:
        sdsge_residual_fn residual
        klein_zgges_fn zgges
        sdsge_dgeqrf_fn dgeqrf
        sdsge_dormqr_fn dormqr
        const double *ss_seed
        const double *params
        const signed char *incidence
        int64_t n_var
        int64_t n_state
        int64_t n_ctrl
        int64_t n_exog
        int64_t n_par

    ctypedef struct sdsge_solve1:
        double *ss
        double *a_real
        double *b_real
        double *c_real
        double *d_real
        c128 *s
        c128 *t
        c128 *z
        double *f
        double *p
        c128 *eig
        int64_t stab
        double *A
        double *B
        int64_t *order
        int64_t n_static
        int64_t n_pred
        int64_t n_both
        int64_t n_fwd

    ctypedef struct sgu_klein_spec:
        klein_spec first
        bc_residual_fn bc_residual
        double *chol
    ctypedef struct sdsge_solve2:
        double *f_xx
        double *bx
        double *gxx
        double *hxx
        double *gss
        double *hss

    arena_size sdsge_klein_solve1_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t nd)
    int64_t sdsge_klein_solve1(const klein_spec *spec, sdsge_solve1 *out,
                               double *arena, int64_t *iarena)
    int SDSGE_KLEIN_SOLVE_SS_SINGULAR
    int SDSGE_KLEIN_SOLVE_SS_NO_CONVERGE
    int SDSGE_KLEIN_SOLVE_QZ
    int SDSGE_KLEIN_SOLVE_SINGULAR
    int SDSGE_KLEIN_SOLVE_NO_STATES
    int SDSGE_KLEIN_SOLVE_SECOND_ORDER
    int SDSGE_KLEIN_SOLVE_RISK

    arena_size sdsge_sgu_klein_solve2_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t nd)
    int64_t sdsge_sgu_klein_solve2(const sgu_klein_spec *spec,
                                   sdsge_solve1 *out1, sdsge_solve2 *out2,
                                   double *arena, int64_t *iarena)

cdef extern from "steady_state.h" nogil:
    arena_size sdsge_newton_arena_size(
        int64_t n_var, int64_t n_par, int64_t n_exog)
    int64_t sdsge_steady_state_newton(
        sdsge_residual_fn residual, const double *seed, const double *par,
        int64_t n_var, int64_t n_par, int64_t n_exog, int64_t max_iter,
        double tol, double *ss, int64_t *iters, double *arena, int64_t *iarena)
    int SDSGE_NEWTON_SINGULAR
    int SDSGE_NEWTON_NO_CONVERGE


cdef extern from "second_order.h" nogil:
    int SDSGE_SECOND_ORDER_SINGULAR
    arena_size sdsge_second_order_arena_size(int64_t n, int64_t nx)
    int64_t sdsge_second_order(
        const double *a, const double *b, const double *f_xx,
        const double *gx, const double *hx, int64_t n, int64_t nx,
        double *gxx, double *hxx, double *arena, int64_t *iarena)
    arena_size sdsge_second_order_risk_arena_size(
        int64_t n, int64_t nx, int64_t ne)
    int64_t sdsge_second_order_risk(
        const double *a, const double *b, const double *f_xx,
        const double *bx, const double *gx, const double *gxx,
        const double *chol, int64_t n, int64_t nx, int64_t ne,
        double *gss, double *hss, double *arena, int64_t *iarena)


cdef _raise_solve_error(int64_t err, str who):
    """Map a fused-solve status onto the staged shims' messages, verbatim:
    callers match on them.
    """
    if err == SDSGE_KLEIN_SOLVE_SS_SINGULAR:
        raise ValueError("steady_state_newton: singular Jacobian (a - b).")
    if err == SDSGE_KLEIN_SOLVE_SS_NO_CONVERGE:
        raise ValueError(
            "steady_state_newton: did not converge within max_iter "
            "(or the residual went non-finite)."
        )
    if err == SDSGE_KLEIN_SOLVE_QZ:
        raise RuntimeError("klein_qz: LAPACK zgges failed.")
    if err == SDSGE_KLEIN_SOLVE_SINGULAR:
        raise ValueError(
            "klein_postprocess: singular z11/s11 (Blanchard-Kahn failure)."
        )
    if err == SDSGE_KLEIN_SOLVE_NO_STATES:
        raise ValueError("klein_postprocess: model has no states.")
    if err == SDSGE_KLEIN_SOLVE_SECOND_ORDER:
        raise ValueError("solve_second_order: singular symmetry-reduced system.")
    if err == SDSGE_KLEIN_SOLVE_RISK:
        raise ValueError("solve_second_order_risk: singular [Qg Qh] system.")


def assemble_transition(p, f, n_state, n_control):
    """Transition matrix ``A`` from a solution ``(p, f)``.

    The shock loading is not assembled from these: it is the pencil stage's own
    output, one solve spanning every variable rather than a state block the
    controls inherit through ``f``."""
    cdef double[:, ::1] pv = np.ascontiguousarray(p, dtype=np.float64)
    cdef double[:, ::1] fv = np.ascontiguousarray(f, dtype=np.float64)

    n = n_state + n_control
    cdef int64_t n_s = <int64_t>n_state
    cdef int64_t n_c = <int64_t>(n - n_state)

    A = np.empty((n, n), dtype=np.float64)
    cdef double[:, ::1] Av = A

    with nogil:
        sdsge_assemble_transition(
            &pv[0, 0], &fv[0, 0], n_s, n_c, &Av[0, 0]
        )
    return A


def simulate_linear_states_into(A, B, x0, shock_mat, double[:, ::1] out):
    """out[(T, n)] <- post-shock linear state recursion. ``out`` is the caller's
    C-contiguous f64 output buffer, written in place; inputs are coerced."""
    cdef double[:, ::1] Av = np.ascontiguousarray(A, dtype=np.float64)
    cdef double[:, ::1] Bv = np.ascontiguousarray(B, dtype=np.float64)
    cdef double[::1] x0v = np.ascontiguousarray(x0, dtype=np.float64)
    cdef double[:, ::1] shockv = np.ascontiguousarray(shock_mat, dtype=np.float64)
    cdef int64_t n = Av.shape[0]
    cdef int64_t k = Bv.shape[1]
    cdef int64_t T = shockv.shape[0]
    cdef const double *shock_ptr = &shockv[0, 0]
    with nogil:
        sdsge_simulate_linear_states(
            &Av[0, 0], &Bv[0, 0], &x0v[0], shock_ptr, &out[0, 0], T, n, k
        )


def affine_observations_into(states, C, d, double[:, ::1] out):
    """out[(T, m)] <- d + C @ states[t]. ``out`` is the caller's
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
            &statesv[0, 0], &Cv[0, 0], &dv[0], &out[0, 0], T, m, n
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
    cdef double[:, ::1] outv
    cdef double *out_ptr = NULL

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

    out = np.empty((T, nx + ny), dtype=np.float64)
    outv = out
    out_ptr = &outv[0, 0]

    if ny > 0:
        gx_ptr = &gxv[0, 0]
        gxx_ptr = &gxxv[0, 0, 0]
        gss_ptr = &gssv[0]
    if n_exog > 0:
        bx_ptr = &bxv[0, 0]
        if T > 0:
            shock_ptr = &shockv[0, 0]

    with nogil:
        err = sdsge_simulate_second_order_pruned(
            &hxv[0, 0], gx_ptr, bx_ptr, &hxxv[0, 0, 0], gxx_ptr,
            &hssv[0], gss_ptr, &x0v[0], shock_ptr,
            T, nx, ny, n_exog, out_ptr)
    if err == SDSGE_CORE_ALLOC_FAIL:
        raise MemoryError("simulate_second_order_pruned: allocation failed.")
    if err != 0:
        raise RuntimeError(
            f"simulate_second_order_pruned: native kernel failed with code {err}."
        )
    return out


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
    cdef arena_size sz = klein_postproc_arena_size(n_s, n_cs)
    arena = np.empty(sz.n_float, dtype=np.float64)
    iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iarv = iarena
    with nogil:
        err = klein_postproc(
            <c128 *>&sv[0, 0], <c128 *>&tv[0, 0], <c128 *>&zv[0, 0], n_s, n_cs,
            <c128 *>&fv[0, 0] if n_cs > 0 else NULL,
            <c128 *>&pv[0, 0], &stab, <c128 *>&ev[0], &arv[0], &iarv[0])
    if err == SDSGE_KLEIN_POSTPROC_SINGULAR:
        raise ValueError(
            "klein_postprocess: singular z11/s11 (Blanchard-Kahn failure)."
        )
    if err == SDSGE_KLEIN_POSTPROC_INVALID:
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
    int64_t n_exog,
):
    """Complex-step Jacobian blocks ``(a, b, c, d)`` from a numba residual
    @cfunc (``build_cfunc``) given its ``.address``. ``a = d resid/d fwd``,
    ``b = -(d resid/d cur)``, ``c = -(d resid/d prev)``, each ``(n_eq, n_var)``,
    and ``d = -(d resid/d eps)``, ``(n_eq, n_exog)``, so the system reads
    ``a y' = b y + c y_prev + d eps``.
    """
    cdef double[::1] ssv = np.ascontiguousarray(steady_state, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(params, dtype=np.float64)
    cdef int64_t n_var = ssv.shape[0]
    cdef int64_t n_par = parv.shape[0]

    a = np.empty((n_eq, n_var), dtype=np.float64)
    b = np.empty((n_eq, n_var), dtype=np.float64)
    c = np.empty((n_eq, n_var), dtype=np.float64)
    d = np.empty((n_eq, n_exog), dtype=np.float64)
    cdef double[:, ::1] av = a
    cdef double[:, ::1] bv = b
    cdef double[:, ::1] cv = c
    cdef double[:, ::1] dv = d

    cdef const double *ss_ptr = &ssv[0] if n_var > 0 else NULL
    cdef const double *par_ptr = &parv[0] if n_par > 0 else NULL
    cdef double *d_ptr = &dv[0, 0] if n_exog > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    arena = np.empty(
        klein_preproc_arena_size(n_var, n_par, n_exog, n_eq).n_float,
        dtype=np.float64,
    )
    cdef double[::1] arv = arena
    with nogil:
        klein_preproc(
            resid, ss_ptr, par_ptr, n_var, n_par, n_exog, n_eq,
            &av[0, 0], &bv[0, 0], &cv[0, 0], d_ptr, &arv[0])
    return a, b, c, d


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
    cdef arena_size sz = klein_qz_arena_size(n)
    arena = np.empty(sz.n_float, dtype=np.float64)
    iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iarv = iarena
    with nogil:
        status = c_klein_qz(
            _zgges, n,
            <c128 *>&av[0, 0], <c128 *>&bv[0, 0], <c128 *>&zv[0, 0],
            &arv[0], &iarv[0])
    if status != KLEIN_QZ_OK:
        raise RuntimeError("klein_qz: LAPACK zgges failed.")
    return a_f, b_f, z


def steady_state_newton(
    size_t residual_addr,
    seed,
    params,
    int64_t n_exog,
    int64_t max_iter=50,
    double tol=1e-12,
):
    """Newton solve of ``F(ss, ss, ss) = 0`` at a zero innovation, from ``seed``,
    driving a numba residual @cfunc (``build_cfunc``) by its ``.address``. The
    Jacobian ``a - b - c`` comes from ``klein_preproc`` each step; the update is
    an in-place LU. Returns ``(ss, iters)``; raises on singular Jacobian or
    non-convergence.
    """
    cdef int64_t n_var = seed.shape[0]
    cdef int64_t n_par = params.shape[0]

    cdef double[::1] seedv = np.ascontiguousarray(seed, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(params, dtype=np.float64)

    cdef const double *seed_ptr = &seedv[0] if n_var > 0 else NULL
    cdef const double *par_ptr = &parv[0] if n_par > 0 else NULL

    ss = np.empty(n_var, dtype=np.float64)
    cdef double[::1] ssv = ss

    cdef double *ss_ptr = &ssv[0] if n_var > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    cdef int64_t iters = 0
    cdef int64_t err
    cdef arena_size sz = sdsge_newton_arena_size(n_var, n_par, n_exog)
    arena = np.empty(sz.n_float, dtype=np.float64)
    iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iarv = iarena
    with nogil:
        err = sdsge_steady_state_newton(
            resid, seed_ptr, par_ptr, n_var, n_par, n_exog, max_iter, tol,
            ss_ptr, &iters, &arv[0], &iarv[0])
    if err == SDSGE_NEWTON_SINGULAR:
        raise ValueError("steady_state_newton: singular Jacobian (a - b - c).")
    if err == SDSGE_NEWTON_NO_CONVERGE:
        raise ValueError(
            "steady_state_newton: did not converge within max_iter "
            "(or the residual went non-finite)."
        )
    return ss, int(iters)


def klein_solve1(
    size_t residual_addr,
    seed,
    params,
    incidence,
    int64_t n_state,
    int64_t n_exog=0,
):
    """One-shot first-order Klein solve, in a single GIL release.

    Fuses ``steady_state_newton`` -> ``klein_preprocess`` -> ``klein_qz`` ->
    ``klein_postprocess`` -> ``assemble_state_space``, driving the same C
    routine as the native estimation objective. Fusing removes the layout
    round-trip the staged path pays: ``klein_qz`` emits column-major and
    ``klein_postprocess`` reads row-major, so staging bridges them by copying
    where the driver transposes in place.

    Returns ``(ss, a, b, f, p, stab, eig, A, B)``. ``f``/``p`` are real: the
    Schur form's imaginary parts are roundoff on a real pencil and the native
    solve projects them once. ``a``/``b`` are the pencil the solve linearized at,
    handed back so a second-order caller need not rebuild it. ``stab`` is
    reported, never raised on: whether a Blanchard-Kahn stability/uniqueness
    violation is fatal is the caller's decision.
    """
    cdef double[::1] seedv = np.ascontiguousarray(seed, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(params, dtype=np.float64)
    cdef int64_t n_var = seedv.shape[0]
    cdef int64_t n_par = parv.shape[0]
    cdef int64_t n_ctrl = n_var - n_state
    cdef signed char[::1] incv = np.ascontiguousarray(incidence, dtype=np.int8)
    cdef int64_t nd = sdsge_pencil_dim(&incv[0], n_var)

    if n_state < 1:
        raise ValueError("klein_solve1 requires n_states >= 1.")
    if n_ctrl < 0:
        raise ValueError("n_states exceeds the matrix dimension.")
    if _nspred(incv) != n_state:
        raise ValueError(
            f"n_state ({n_state}) disagrees with the incidence, which reports "
            f"{_nspred(incv)} variables at t-1. The solve indexes its rules by "
            f"one and walks them by the other."
        )

    ss = np.empty(n_var, dtype=np.float64)
    a = np.empty((n_var, n_var), dtype=np.float64)
    b = np.empty((n_var, n_var), dtype=np.float64)
    c = np.empty((n_var, n_var), dtype=np.float64)
    d = np.empty((n_var, n_exog), dtype=np.float64)
    s = np.empty((nd, nd), dtype=np.complex128)
    t = np.empty((nd, nd), dtype=np.complex128)
    z = np.empty((nd, nd), dtype=np.complex128)
    f = np.empty((n_ctrl, n_state), dtype=np.float64)
    p = np.empty((n_state, n_state), dtype=np.float64)
    eig = np.empty(nd, dtype=np.complex128)
    A = np.empty((n_var, n_var), dtype=np.float64)
    B = np.empty((n_var, n_exog), dtype=np.float64)
    order = np.empty(n_var, dtype=np.int64)

    cdef double[::1] ssv = ss
    cdef double[:, ::1] av = a
    cdef double[:, ::1] bv = b
    cdef double[:, ::1] cv = c
    cdef double[:, ::1] dv = d
    cdef double complex[:, ::1] sv = s
    cdef double complex[:, ::1] tv = t
    cdef double complex[:, ::1] zv = z
    cdef double[:, ::1] fv = f
    cdef double[:, ::1] pv = p
    cdef double complex[::1] eigv = eig
    cdef double[:, ::1] Av = A
    cdef double[:, ::1] Bv = B
    cdef int64_t[::1] orderv = order

    cdef klein_spec spec
    spec.residual = <sdsge_residual_fn><void*>residual_addr
    spec.zgges = _zgges
    spec.dgeqrf = _dgeqrf
    spec.dormqr = _dormqr
    spec.ss_seed = &seedv[0]
    spec.params = &parv[0] if n_par > 0 else NULL
    spec.incidence = &incv[0]
    spec.n_var = n_var
    spec.n_state = n_state
    spec.n_ctrl = n_ctrl
    spec.n_exog = n_exog
    spec.n_par = n_par

    cdef sdsge_solve1 out
    out.ss = &ssv[0]
    out.a_real = &av[0, 0]
    out.b_real = &bv[0, 0]
    out.c_real = &cv[0, 0]
    out.d_real = &dv[0, 0] if n_exog > 0 else NULL
    out.s = <c128 *>&sv[0, 0]
    out.t = <c128 *>&tv[0, 0]
    out.z = <c128 *>&zv[0, 0]
    out.f = &fv[0, 0] if n_ctrl > 0 else NULL
    out.p = &pv[0, 0]
    out.eig = <c128 *>&eigv[0]
    out.stab = 0
    out.A = &Av[0, 0]
    out.B = &Bv[0, 0] if n_exog > 0 else NULL
    out.order = &orderv[0]

    cdef int64_t err
    cdef arena_size sz = sdsge_klein_solve1_arena_size(
        n_var, n_state, n_ctrl, n_par, n_exog, nd
    )
    arena = np.empty(sz.n_float, dtype=np.float64)
    iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iarv = iarena
    with nogil:
        err = sdsge_klein_solve1(&spec, &out, &arv[0], &iarv[0])

    _raise_solve_error(err, "klein_solve1")
    return ss, f, p, int(out.stab), eig, A, B


def sgu_klein_solve2(
    size_t residual_addr,
    size_t bc_residual_addr,
    seed,
    params,
    Q,
    incidence,
    int64_t n_state,
    int64_t n_exog=0,
):
    """One-shot second-order (SGU) solve, in a single GIL release.

    Runs ``klein_solve1`` and then the second-order tail: the bicomplex residual
    Hessian at the resolved steady state, the SGU tensors, and the sigma^2 risk
    correction. ``bc_residual_addr`` is the bicomplex residual @cfunc
    (``construct_objective_cfunc_bicomplex()``); ``Q`` is the ``(n_exog,
    n_exog)`` shock covariance, which the solve factors and loads through.

    Returns ``(ss, f, p, stab, eig, gxx, hxx, gss, hss, A, B)``.
    ``stab`` is reported, never raised on.
    """
    cdef double[::1] seedv = np.ascontiguousarray(seed, dtype=np.float64)
    cdef double[::1] parv = np.ascontiguousarray(params, dtype=np.float64)
    cdef double[:, ::1] Qv = np.ascontiguousarray(Q, dtype=np.float64)
    cdef int64_t n_var = seedv.shape[0]
    cdef int64_t n_par = parv.shape[0]
    cdef int64_t n_ctrl = n_var - n_state
    cdef int64_t n2 = 2 * n_var
    cdef signed char[::1] incv = np.ascontiguousarray(incidence, dtype=np.int8)
    cdef int64_t nd = sdsge_pencil_dim(&incv[0], n_var)

    if n_state < 1:
        raise ValueError("sgu_klein_solve2 requires n_states >= 1.")
    if n_ctrl < 0:
        raise ValueError("n_states exceeds the matrix dimension.")
    if _nspred(incv) != n_state:
        raise ValueError(
            f"n_state ({n_state}) disagrees with the incidence, which reports "
            f"{_nspred(incv)} variables at t-1. The solve indexes its rules by "
            f"one and walks them by the other."
        )

    cdef double[:, ::1] cholv = np.empty((n_exog, n_exog), dtype=np.float64)
    if n_exog > 0:
        if sdsge_chol(&Qv[0, 0], 0.0, &cholv[0, 0], n_exog) != 0:
            raise ValueError("Cholesky factorization of Q failed: "
                             "Q is not positive definite.")

    ss = np.empty(n_var, dtype=np.float64)
    a = np.empty((n_var, n_var), dtype=np.float64)
    b = np.empty((n_var, n_var), dtype=np.float64)
    c = np.empty((n_var, n_var), dtype=np.float64)
    d = np.empty((n_var, n_exog), dtype=np.float64)
    s = np.empty((nd, nd), dtype=np.complex128)
    t = np.empty((nd, nd), dtype=np.complex128)
    z = np.empty((nd, nd), dtype=np.complex128)
    f = np.empty((n_ctrl, n_state), dtype=np.float64)
    p = np.empty((n_state, n_state), dtype=np.float64)
    eig = np.empty(nd, dtype=np.complex128)
    A = np.empty((n_var, n_var), dtype=np.float64)
    B = np.empty((n_var, n_exog), dtype=np.float64)
    order = np.empty(n_var, dtype=np.int64)

    f_xx = np.empty((n_var, n2, n2), dtype=np.float64)
    bx = np.empty((n_state, n_exog), dtype=np.float64)
    gxx = np.empty((n_ctrl, n_state, n_state), dtype=np.float64)
    hxx = np.empty((n_state, n_state, n_state), dtype=np.float64)
    gss = np.empty(n_ctrl, dtype=np.float64)
    hss = np.empty(n_state, dtype=np.float64)

    cdef double[::1] ssv = ss
    cdef double[:, ::1] av = a
    cdef double[:, ::1] bv = b
    cdef double[:, ::1] cv = c
    cdef double[:, ::1] dv = d
    cdef double complex[:, ::1] sv = s
    cdef double complex[:, ::1] tv = t
    cdef double complex[:, ::1] zv = z
    cdef double[:, ::1] fv = f
    cdef double[:, ::1] pv = p
    cdef double complex[::1] eigv = eig
    cdef double[:, ::1] Av = A
    cdef double[:, ::1] Bv = B
    cdef int64_t[::1] orderv = order

    cdef double[:, :, ::1] fxxv = f_xx
    cdef double[:, ::1] bxv = bx
    cdef double[:, :, ::1] gxxv = gxx
    cdef double[:, :, ::1] hxxv = hxx
    cdef double[::1] gssv = gss
    cdef double[::1] hssv = hss

    cdef sgu_klein_spec spec
    spec.first.residual = <sdsge_residual_fn><void*>residual_addr
    spec.first.zgges = _zgges
    spec.first.dgeqrf = _dgeqrf
    spec.first.dormqr = _dormqr
    spec.first.ss_seed = &seedv[0]
    spec.first.params = &parv[0] if n_par > 0 else NULL
    spec.first.incidence = &incv[0]
    spec.first.n_var = n_var
    spec.first.n_state = n_state
    spec.first.n_ctrl = n_ctrl
    spec.first.n_exog = n_exog
    spec.first.n_par = n_par
    spec.bc_residual = <bc_residual_fn><void*>bc_residual_addr
    spec.chol = &cholv[0, 0] if n_exog > 0 else NULL

    cdef sdsge_solve1 out
    out.ss = &ssv[0]
    out.a_real = &av[0, 0]
    out.b_real = &bv[0, 0]
    out.c_real = &cv[0, 0]
    out.d_real = &dv[0, 0] if n_exog > 0 else NULL
    out.s = <c128 *>&sv[0, 0]
    out.t = <c128 *>&tv[0, 0]
    out.z = <c128 *>&zv[0, 0]
    out.f = &fv[0, 0] if n_ctrl > 0 else NULL
    out.p = &pv[0, 0]
    out.eig = <c128 *>&eigv[0]
    out.stab = 0
    out.A = &Av[0, 0]
    out.B = &Bv[0, 0] if n_exog > 0 else NULL
    out.order = &orderv[0]

    cdef sdsge_solve2 out2
    out2.f_xx = &fxxv[0, 0, 0]
    out2.bx = &bxv[0, 0] if n_exog > 0 else NULL
    out2.gxx = &gxxv[0, 0, 0] if n_ctrl > 0 else NULL
    out2.hxx = &hxxv[0, 0, 0]
    out2.gss = &gssv[0] if n_ctrl > 0 else NULL
    out2.hss = &hssv[0]

    cdef int64_t err
    cdef arena_size sz = sdsge_sgu_klein_solve2_arena_size(
        n_var, n_state, n_ctrl, n_par, n_exog, nd
    )
    arena = np.empty(sz.n_float, dtype=np.float64)
    iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iarv = iarena
    with nogil:
        err = sdsge_sgu_klein_solve2(&spec, &out, &out2, &arv[0], &iarv[0])

    _raise_solve_error(err, "sgu_klein_solve2")
    return ss, f, p, int(out.stab), eig, gxx, hxx, gss, hss, A, B


def second_order(a, b, f_xx, gx, hx, int64_t n_state):
    """SGU second-order tensors ``(gxx, hxx)``. Parity oracle:
    ``core.second_order.solve_second_order``. ``a``/``b`` are the first-order
    pencil ``(n, n)``, ``f_xx`` the residual Hessian ``(n, 2n, 2n)``, ``gx``
    ``(ny, nx)``, ``hx`` ``(nx, nx)``. Returns ``gxx (ny, nx, nx)``,
    ``hxx (nx, nx, nx)``.

    Inputs are coerced to C-contiguous f64; ``gx``/``hx`` are the Klein
    solution's ``f``/``p``, already real.
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
    cdef arena_size sz = sdsge_second_order_arena_size(n, nx)
    arena = np.empty(sz.n_float, dtype=np.float64)
    iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iarv = iarena
    with nogil:
        err = sdsge_second_order(
            &av[0, 0], &bv[0, 0], &fxxv[0, 0, 0], gx_ptr, &hxv[0, 0], n, nx,
            gv_ptr, &hv[0, 0, 0], &arv[0], &iarv[0])
    if err == SDSGE_SECOND_ORDER_SINGULAR:
        raise ValueError("solve_second_order: singular symmetry-reduced system.")
    return gxx, hxx


def second_order_risk(a, b, f_xx, bx, gx, gxx, chol, int64_t n_state):
    """Sigma^2 risk correction ``(gss, hss)``. Parity oracle:
    ``core.second_order.solve_second_order_risk``. ``gxx`` is the second-order
    controls ``(ny, nx, nx)``; ``bx`` the state rows of ``B`` ``(nx, ne)`` and
    ``chol`` the shock covariance factor ``(ne, ne)``, which the kernel composes
    into the loading itself. Returns ``gss (ny,)``, ``hss (nx,)``. Inputs
    coerced to C-contiguous f64.
    """
    cdef double[:, ::1] av = np.ascontiguousarray(a, dtype=np.float64)
    cdef double[:, ::1] bv = np.ascontiguousarray(b, dtype=np.float64)
    cdef double[:, :, ::1] fxxv = np.ascontiguousarray(f_xx, dtype=np.float64)
    cdef double[:, ::1] gxv = np.ascontiguousarray(gx, dtype=np.float64)
    cdef double[:, :, ::1] gxxv = np.ascontiguousarray(gxx, dtype=np.float64)
    cdef double[:, ::1] bxv = np.ascontiguousarray(bx, dtype=np.float64)
    cdef double[:, ::1] cholv = np.ascontiguousarray(chol, dtype=np.float64)

    cdef int64_t n = av.shape[0]
    cdef int64_t nx = n_state
    cdef int64_t ny = n - nx
    cdef int64_t ne = cholv.shape[1]

    gss = np.empty(ny, dtype=np.float64)
    hss = np.empty(nx, dtype=np.float64)
    cdef double[::1] gssv = gss
    cdef double[::1] hssv = hss

    cdef const double *gx_ptr = &gxv[0, 0] if ny > 0 else NULL
    cdef const double *gxx_ptr = &gxxv[0, 0, 0] if ny > 0 else NULL
    cdef const double *bx_ptr = &bxv[0, 0] if ne > 0 else NULL
    cdef const double *chol_ptr = &cholv[0, 0] if ne > 0 else NULL
    cdef double *gss_ptr = &gssv[0] if ny > 0 else NULL
    cdef int64_t err
    cdef arena_size sz = sdsge_second_order_risk_arena_size(n, nx, ne)
    arena = np.empty(sz.n_float, dtype=np.float64)
    iarena = np.empty(sz.n_int, dtype=np.int64)
    cdef double[::1] arv = arena
    cdef int64_t[::1] iarv = iarena
    with nogil:
        err = sdsge_second_order_risk(
            &av[0, 0], &bv[0, 0], &fxxv[0, 0, 0], bx_ptr, gx_ptr, gxx_ptr,
            chol_ptr, n, nx, ne, gss_ptr, &hssv[0], &arv[0], &iarv[0])
    if err == SDSGE_SECOND_ORDER_SINGULAR:
        raise ValueError("solve_second_order_risk: singular [Qg Qh] system.")
    return gss, hss


def residual_path(
    size_t residual_addr, cur_states, fwd_states, prev_states, shocks, params,
    int64_t n_eq,
):
    """Real residual matrix ``(n_steps, n_eq)`` from a residual @cfunc
    (``build_cfunc``) evaluated over a simulated path. Native backend for the
    Den Haan-Marcet moment builder, reusing the solve's cfunc so it never
    triggers the numba residual compile. ``prev_states`` is ``(n_steps, n_var)``
    and ``shocks`` ``(n_steps, n_exog)``, dated like the rest of the path.
    Inputs are coerced to contiguous complex128 here.
    """
    cdef double complex[:, ::1] curv = np.ascontiguousarray(
        cur_states, dtype=np.complex128)
    cdef double complex[:, ::1] fwdv = np.ascontiguousarray(
        fwd_states, dtype=np.complex128)
    cdef double complex[:, ::1] prevv = np.ascontiguousarray(
        prev_states, dtype=np.complex128)
    cdef double complex[:, ::1] epsv = np.ascontiguousarray(
        shocks, dtype=np.complex128)
    cdef double complex[::1] parv = np.ascontiguousarray(
        params, dtype=np.complex128).reshape(-1)
    cdef int64_t n_steps = curv.shape[0]
    cdef int64_t n_var = curv.shape[1]
    cdef int64_t n_exog = epsv.shape[1]
    residuals = np.empty((n_steps, n_eq), dtype=np.float64)
    cdef double[:, ::1] rv = residuals

    cdef c128 *cur_ptr = <c128 *>&curv[0, 0] if n_steps > 0 else NULL
    cdef c128 *fwd_ptr = <c128 *>&fwdv[0, 0] if n_steps > 0 else NULL
    cdef c128 *prev_ptr = <c128 *>&prevv[0, 0] if n_steps > 0 else NULL
    cdef c128 *eps_ptr = (
        <c128 *>&epsv[0, 0] if n_steps > 0 and n_exog > 0 else NULL)
    cdef c128 *par_ptr = <c128 *>&parv[0] if parv.shape[0] > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    cdef int64_t err
    with nogil:
        err = sdsge_residual_path(
            resid, cur_ptr, fwd_ptr, prev_ptr, eps_ptr, par_ptr, n_steps,
            n_var, n_exog, n_eq, &rv[0, 0])
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


def residual_eval(size_t residual_addr, fwd, cur, prev, eps, params, int64_t n_eq):
    """Complex residual vector ``F(fwd, cur, prev, eps, par)`` of length ``n_eq``
    from a residual @cfunc (``build_cfunc``) given its ``.address``. Single-point
    native evaluation, the path ``CompiledModel.equations`` takes instead of the
    numba vector kernel. Inputs are coerced to contiguous complex128 here.
    """
    cdef double complex[::1] fwdv = np.ascontiguousarray(
        fwd, dtype=np.complex128).reshape(-1)
    cdef double complex[::1] curv = np.ascontiguousarray(
        cur, dtype=np.complex128).reshape(-1)
    cdef double complex[::1] prevv = np.ascontiguousarray(
        prev, dtype=np.complex128).reshape(-1)
    cdef double complex[::1] epsv = np.ascontiguousarray(
        eps, dtype=np.complex128).reshape(-1)
    cdef double complex[::1] parv = np.ascontiguousarray(
        params, dtype=np.complex128).reshape(-1)
    out = np.empty((n_eq,), dtype=np.complex128)
    cdef double complex[::1] ov = out

    cdef c128 *fwd_ptr = <c128 *>&fwdv[0] if fwdv.shape[0] > 0 else NULL
    cdef c128 *cur_ptr = <c128 *>&curv[0] if curv.shape[0] > 0 else NULL
    cdef c128 *prev_ptr = <c128 *>&prevv[0] if prevv.shape[0] > 0 else NULL
    cdef c128 *eps_ptr = <c128 *>&epsv[0] if epsv.shape[0] > 0 else NULL
    cdef c128 *par_ptr = <c128 *>&parv[0] if parv.shape[0] > 0 else NULL
    cdef c128 *out_ptr = <c128 *>&ov[0] if n_eq > 0 else NULL
    cdef sdsge_residual_fn resid = <sdsge_residual_fn><void*>residual_addr
    with nogil:
        resid(fwd_ptr, cur_ptr, prev_ptr, eps_ptr, par_ptr, out_ptr)
    return out


def bicomplex_hessian(
    size_t residual_addr,
    double[::1] steady_state,
    double[::1] params,
    int64_t n_exog,
    int64_t n_eq,
):
    """Residual Hessian ``F_xx`` (n_eq, 2*n_var, 2*n_var) via the bicomplex step,
    from a bicomplex residual @cfunc (``build_cfunc(..., BicomplexOps())``) given
    its ``.address``. Second-order native preproc; feeds the g_xx assembly.

    The sweep spans the ``(fwd, cur)`` pair only. ``prev`` is held at the steady
    state and ``eps`` at zero, so derivatives in either are absent from the
    result.

    The step is tuned and fixed at ``SDSGE_HESSIAN_STEP``;
    see the C header for what sets it.
    """
    cdef int64_t n_var = steady_state.shape[0]
    cdef int64_t n_par = params.shape[0]
    cdef int64_t n2 = 2 * n_var

    hessian = np.empty((n_eq, n2, n2), dtype=np.float64)
    cdef double[:, :, ::1] hv = hessian

    cdef const double *ss_ptr = &steady_state[0] if n_var > 0 else NULL
    cdef const double *par_ptr = &params[0] if n_par > 0 else NULL
    cdef bc_residual_fn residual = <bc_residual_fn><void*>residual_addr
    arena = np.empty(
        sdsge_bicomplex_hessian_arena_size(n_var, n_par, n_exog, n_eq).n_float,
        dtype=np.float64,
    )
    cdef double[::1] arv = arena
    with nogil:
        sdsge_bicomplex_hessian(
            residual, ss_ptr, par_ptr, n_var, n_par, n_exog, n_eq, &hv[0, 0, 0],
            &arv[0])
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
