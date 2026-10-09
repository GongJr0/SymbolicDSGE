# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython composer for the native estimation objective.

Marshals Python-side prep (packed arrays + cfunc/LAPACK addresses) into the C
context struct and calls the native objective. Python never assembles the struct
by hand.
"""

from libc.stdint cimport int64_t

from cpython.pycapsule cimport (
    PyCapsule_GetName,
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
)

import numpy as np
import scipy.linalg.cython_lapack as _cython_lapack

from numpy.random cimport bitgen_t


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


# --- DTO -> C struct fill helpers -------------------------------------------
#
# `hold` is why the normalization is safe here at all. When the input is NOT
# already conformant, ascontiguousarray returns a NEW array that nothing else
# holds. Appending the result keeps it alive for the caller's frame, which is
# where the native call happens.
#
# A zero-length table hands back its base address like any other. No kernel
# branches on these pointers being NULL  and every C reader walks the paired
# count before touching the table.

cdef inline object _held(object a, list hold):
    """Root an array this frame owns. Returns it so an allocation, its keepalive
    and its typed view stay one statement."""
    hold.append(a)
    return a


cdef inline const double *_f64p(object a, list hold) except NULL:
    arr = np.ascontiguousarray(a, dtype=np.float64)
    if arr is not a:
        hold.append(arr)
    cdef const double[::1] v = arr
    return &v[0]


cdef inline const double *_f64p2(object a, list hold) except NULL:
    arr = np.ascontiguousarray(a, dtype=np.float64)
    if arr is not a:
        hold.append(arr)
    cdef const double[:, ::1] v = arr
    return &v[0, 0]


cdef inline const int64_t *_i64p(object a, list hold) except NULL:
    arr = np.ascontiguousarray(a, dtype=np.int64)
    if arr is not a:
        hold.append(arr)
    cdef const int64_t[::1] v = arr
    return &v[0]


cdef inline const signed char *_i8p(object a, list hold) except NULL:
    arr = np.ascontiguousarray(a, dtype=np.int8)
    if arr is not a:
        hold.append(arr)
    cdef const signed char[::1] v = arr
    return &v[0]


cdef inline int64_t _rows(object a):
    return <int64_t>a.shape[0]


cdef void _fill_param_map(sdsge_param_map *pm, object pmap, list hold) except *:
    pm.base_params = _f64p(pmap.base_params, hold)
    pm.param_slot = _i64p(pmap.param_slot, hold)
    pm.n_theta = _rows(pmap.param_slot)


cdef void _fill_cov_build(sdsge_cov_build *cb, object sp, list hold) except *:
    cb.spec.is_constant = <int>bool(sp.is_constant)
    cb.spec.K = <int64_t>sp.K
    cb.spec.corr_from_block = <int>bool(sp.corr_from_block)
    cb.spec.block_theta_off = <int64_t>sp.block_theta_off
    cb.spec.block_theta_len = <int64_t>sp.block_theta_len
    if sp.is_constant:
        cb.spec.constant = _f64p2(sp.constant, hold)
        cb.spec.std_slots = NULL
        cb.spec.pair_i = NULL
        cb.spec.pair_j = NULL
        cb.spec.pair_slot = NULL
        cb.spec.n_pairs = 0
    else:
        cb.spec.constant = NULL
        cb.spec.std_slots = _i64p(sp.std_slots, hold)
        cb.spec.pair_i = _i64p(sp.pair_i, hold)
        cb.spec.pair_j = _i64p(sp.pair_j, hold)
        cb.spec.pair_slot = _i64p(sp.pair_slot, hold)
        cb.spec.n_pairs = _rows(sp.pair_i)


cdef void _fill_prior(sdsge_prior_tables *pr, object pt, list hold) except *:
    pr.has_prior = <int>bool(pt.has_prior)
    pr.include_logjac = 0
    pr.dist_codes = _i64p(pt.dist_codes, hold)
    pr.transform_codes = _i64p(pt.transform_codes, hold)
    pr.dist_params = _f64p2(pt.dist_params, hold)
    pr.transform_params = _f64p2(pt.transform_params, hold)
    pr.n_theta = _rows(pt.dist_codes)


cdef void _fill_klein_spec(klein_spec *sp, object sol_dto,
                           list hold, const double *params,
                           const signed char *incidence) except *:

    sp.residual = <sdsge_residual_fn><void*><size_t>sol_dto.residual_addr
    sp.zgges = _zgges
    sp.dgeqrf = _dgeqrf
    sp.dormqr = _dormqr
    # Reordering doesn't change determinancy status;
    # estimation rejects either ordering.
    sp.ztgexc = NULL
    sp.ss_seed = _f64p(sol_dto.ss_seed, hold)
    sp.params = params
    sp.incidence = incidence
    sp.n_var = <int64_t>sol_dto.n_var
    sp.n_state = <int64_t>sol_dto.n_state
    sp.n_ctrl = <int64_t>sol_dto.n_ctrl
    sp.n_exog = <int64_t>sol_dto.n_exog
    sp.n_par = <int64_t>sol_dto.n_par


cdef inline const double *_cov_ptr(sdsge_cov_build *cb):
    return cb.spec.constant if cb.spec.is_constant else cb.out


# --- allocators -------------------------------------------------------------
#
# Each one allocates a struct's buffers and publishes their addresses in the same
# breath, so a buffer and the field that points at it are never apart. The arrays
# are rooted in `hold`, a list the CALLER owns, which is what lets the allocation
# live in a helper while the pointer outlives it.

cdef object _alloc_base(sdsge_obj_common *b, object sol_dto, int64_t n_obs,
                        arena_size asz, list hold):
    cdef int64_t n_par = sol_dto.n_par
    cdef int64_t n_exog = sol_dto.n_exog

    params = _held(np.empty(n_par, np.float64), hold)
    cdef double[::1] pv = params
    b.params = &pv[0]

    cdef double[:, ::1] qo = _held(np.empty((n_exog, n_exog), np.float64), hold)
    cdef double[:, ::1] qc = _held(np.empty((n_exog, n_exog), np.float64), hold)
    cdef double[::1] qs = _held(np.empty(n_exog, np.float64), hold)
    b.q.out = &qo[0, 0]
    b.q.corr = &qc[0, 0]
    b.q.std = &qs[0]

    cdef double[:, ::1] ro = _held(np.empty((n_obs, n_obs), np.float64), hold)
    cdef double[:, ::1] rc = _held(np.empty((n_obs, n_obs), np.float64), hold)
    cdef double[::1] rs = _held(np.empty(n_obs, np.float64), hold)
    b.r.out = &ro[0, 0]
    b.r.corr = &rc[0, 0]
    b.r.std = &rs[0]

    cdef double[::1] ar = _held(np.empty(asz.n_float, np.float64), hold)
    cdef int64_t[::1] ia = _held(np.empty(asz.n_int, np.int64), hold)
    b.arena = &ar[0]
    b.iarena = &ia[0]

    b.bk_violations = 0
    return params


cdef void _alloc_solve1(sdsge_solve1 *o, object sol_dto, int64_t nd, list hold):
    cdef int64_t n_var = sol_dto.n_var
    cdef int64_t n_state = sol_dto.n_state
    cdef int64_t n_ctrl = sol_dto.n_ctrl
    cdef int64_t n_exog = sol_dto.n_exog

    cdef double[::1] ss = _held(np.empty(n_var, np.float64), hold)
    cdef double[:, ::1] ar = _held(np.empty((n_var, n_var), np.float64), hold)
    cdef double[:, ::1] br = _held(np.empty((n_var, n_var), np.float64), hold)
    cdef double[:, ::1] cr = _held(np.empty((n_var, n_var), np.float64), hold)
    cdef double[:, ::1] dr = _held(np.empty((n_var, n_exog), np.float64), hold)
    cdef double complex[::1, :] sc = _held(
        np.empty((nd, nd), np.complex128, order="F"), hold)
    cdef double complex[::1, :] tc = _held(
        np.empty((nd, nd), np.complex128, order="F"), hold)
    cdef double complex[::1, :] zc = _held(
        np.empty((nd, nd), np.complex128, order="F"), hold)
    cdef double[:, ::1] f = _held(np.empty((n_ctrl, n_state), np.float64), hold)
    cdef double[:, ::1] p = _held(np.empty((n_state, n_state), np.float64), hold)
    cdef double complex[::1] eig = _held(np.empty(nd, np.complex128), hold)
    cdef double[:, ::1] A = _held(np.empty((n_var, n_var), np.float64), hold)
    cdef double[:, ::1] B = _held(np.empty((n_var, n_exog), np.float64), hold)
    cdef int64_t[::1] od = _held(np.empty(n_var, np.int64), hold)

    o.ss = &ss[0]
    o.a_real = &ar[0, 0]
    o.b_real = &br[0, 0]
    o.c_real = &cr[0, 0]
    o.d_real = &dr[0, 0]
    o.s = <c128*>&sc[0, 0]
    o.t = <c128*>&tc[0, 0]
    o.z = <c128*>&zc[0, 0]
    o.f = &f[0, 0]
    o.p = &p[0, 0]
    o.eig = <c128*>&eig[0]
    o.A = &A[0, 0]
    o.B = &B[0, 0]
    o.order = &od[0]


cdef void _alloc_solve2(sdsge_solve2 *o, object sol_dto, list hold):
    cdef int64_t n_var = sol_dto.n_var
    cdef int64_t n_state = sol_dto.n_state
    cdef int64_t n_ctrl = sol_dto.n_ctrl
    cdef int64_t n_exog = sol_dto.n_exog
    cdef int64_t n2 = 3 * n_var + n_exog

    cdef double[:, :, ::1] fxx = _held(
        np.empty((n_var, n2, n2), np.float64), hold)
    cdef double[:, ::1] bx = _held(np.empty((n_state, n_exog), np.float64), hold)
    cdef double[:, :, ::1] gxx = _held(
        np.empty((n_ctrl, n_state, n_state), np.float64), hold)
    cdef double[:, :, ::1] hxx = _held(
        np.empty((n_state, n_state, n_state), np.float64), hold)
    cdef double[:, :, ::1] gxu = _held(
        np.empty((n_ctrl, n_state, n_exog), np.float64), hold)
    cdef double[:, :, ::1] hxu = _held(
        np.empty((n_state, n_state, n_exog), np.float64), hold)
    cdef double[:, :, ::1] guu = _held(
        np.empty((n_ctrl, n_exog, n_exog), np.float64), hold)
    cdef double[:, :, ::1] huu = _held(
        np.empty((n_state, n_exog, n_exog), np.float64), hold)
    cdef double[::1] gss = _held(np.empty(n_ctrl, np.float64), hold)
    cdef double[::1] hss = _held(np.empty(n_state, np.float64), hold)

    o.f_xx = &fxx[0, 0, 0]
    o.bx = &bx[0, 0]
    o.gxx = &gxx[0, 0, 0]
    o.hxx = &hxx[0, 0, 0]
    o.gxu = &gxu[0, 0, 0]
    o.hxu = &hxu[0, 0, 0]
    o.guu = &guu[0, 0, 0]
    o.huu = &huu[0, 0, 0]
    o.gss = &gss[0]
    o.hss = &hss[0]


cdef const double *_alloc_P0(sdsge_obj_common *b, object kf_dto, int64_t fdim,
                             list hold) except NULL:
    cdef double[:, ::1] own
    if kf_dto.P0 is None:
        own = _held(np.zeros((fdim, fdim), np.float64), hold)
        b.derive_P0 = 1
        return &own[0, 0]
    b.derive_P0 = 0
    return _f64p2(kf_dto.P0, hold)


# --- filter input fills -----------------------------------------------------
#
# One per mode, allocating the filter's own buffers and then writing the input
# struct in field order so it reads as the checklist the C header is.
#
# Nothing here is per-draw: every pointer addresses a buffer whose location is
# fixed for the run, and the solve, the measurement build, and the covariance
# build rewrite those buffers in place each draw. MUST run after the mode's solve
# outputs are wired, because it copies their addresses out of the ctx.
#
# `y` and a supplied `P0` are the only DTO-rooted pointers in these structs.

cdef void _alloc_fill_kf(sdsge_linear_ctx *c, object kf_dto, object sol_dto,
                         list hold) except *:
    cdef sdsge_obj_common *b = &c.base
    cdef int64_t n_var = sol_dto.n_var
    cdef int64_t n_obs = kf_dto.n_obs

    c.meas = <meas_fn><void*><size_t>kf_dto.meas_addr
    c.jac = <meas_fn><void*><size_t>kf_dto.jac_addr

    cdef double[:, ::1] C = _held(np.empty((n_obs, n_var), np.float64), hold)
    cdef double[::1] d = _held(np.empty(n_obs, np.float64), hold)
    cdef double[::1] x0 = _held(np.zeros(n_var, np.float64), hold)

    c.kf_ctx.n = n_var
    c.kf_ctx.m = n_obs
    c.kf_ctx.k = <int64_t>sol_dto.n_exog
    c.kf_ctx.T = <int64_t>kf_dto.T
    c.kf_ctx.A = c.solve_out.A
    c.kf_ctx.B = c.solve_out.B
    c.kf_ctx.C = &C[0, 0]
    c.kf_ctx.d = &d[0]
    c.kf_ctx.Q = _cov_ptr(&b.q)
    c.kf_ctx.R = _cov_ptr(&b.r)
    c.kf_ctx.steady_state = c.solve_out.ss
    c.kf_ctx.y = _f64p2(kf_dto.y_reordered, hold)
    c.kf_ctx.x0 = &x0[0]
    c.kf_ctx.P0 = _alloc_P0(b, kf_dto, n_var, hold)
    c.kf_ctx.symmetrize = <int>bool(kf_dto.sym)
    c.kf_ctx.joseph_cov = <int>bool(kf_dto.joseph_cov)
    c.kf_ctx.jitter = <double>kf_dto.jitter
    c.kf_ctx.return_shocks = 0
    c.kf_ctx.store_history = 0


cdef void _alloc_fill_ekf(sdsge_extended_ctx *c, object kf_dto, object sol_dto,
                          list hold) except *:
    cdef sdsge_obj_common *b = &c.base
    cdef int64_t n_var = sol_dto.n_var

    cdef double[::1] x0 = _held(np.zeros(n_var, np.float64), hold)

    c.ekf_ctx.meas = <meas_fn><void*><size_t>kf_dto.meas_addr
    c.ekf_ctx.jac = <meas_fn><void*><size_t>kf_dto.jac_addr
    c.ekf_ctx.A = c.solve_out.A
    c.ekf_ctx.B = c.solve_out.B
    c.ekf_ctx.calib_params = b.params
    c.ekf_ctx.Q = _cov_ptr(&b.q)
    c.ekf_ctx.R = _cov_ptr(&b.r)
    c.ekf_ctx.steady_state = c.solve_out.ss
    c.ekf_ctx.y = _f64p2(kf_dto.y_reordered, hold)
    c.ekf_ctx.x0 = &x0[0]
    c.ekf_ctx.P0 = _alloc_P0(b, kf_dto, n_var, hold)
    c.ekf_ctx.T = <int64_t>kf_dto.T
    c.ekf_ctx.n = n_var
    c.ekf_ctx.m = <int64_t>kf_dto.n_obs
    c.ekf_ctx.k = <int64_t>sol_dto.n_exog
    c.ekf_ctx.n_par = <int64_t>sol_dto.n_par
    c.ekf_ctx.jitter = <double>kf_dto.jitter
    c.ekf_ctx.symmetrize = <int>bool(kf_dto.sym)
    c.ekf_ctx.joseph_cov = <int>bool(kf_dto.joseph_cov)
    c.ekf_ctx.compute_y_filt = 0
    c.ekf_ctx.return_shocks = 0
    c.ekf_ctx.store_history = 0


cdef void _alloc_fill_ukf(sdsge_unscented_ctx *c, object kf_dto,
                          object sol_dto, list hold) except *:
    cdef sdsge_obj_common *b = &c.base
    cdef int64_t n_state = sol_dto.n_state

    c.solve_ctx.bc_residual = <bc_residual_fn><void*><size_t>sol_dto.bc_residual_addr
    c.solve_ctx.Q = _cov_ptr(&b.q)

    cdef double[::1] z0 = _held(np.zeros(2 * n_state, np.float64), hold)

    c.ukf_ctx.meas = <meas_fn><void*><size_t>kf_dto.meas_addr
    c.ukf_ctx.hx = c.solve1_out.p
    c.ukf_ctx.gx = c.solve1_out.f
    c.ukf_ctx.bu = c.solve1_out.B
    c.ukf_ctx.hxx = c.solve2_out.hxx
    c.ukf_ctx.gxx = c.solve2_out.gxx
    c.ukf_ctx.hxu = c.solve2_out.hxu
    c.ukf_ctx.gxu = c.solve2_out.gxu
    c.ukf_ctx.huu = c.solve2_out.huu
    c.ukf_ctx.guu = c.solve2_out.guu
    c.ukf_ctx.hss = c.solve2_out.hss
    c.ukf_ctx.gss = c.solve2_out.gss
    c.ukf_ctx.steady_state = c.solve1_out.ss
    c.ukf_ctx.params = b.params
    c.ukf_ctx.Q = _cov_ptr(&b.q)
    c.ukf_ctx.R = _cov_ptr(&b.r)
    c.ukf_ctx.obs = _f64p2(kf_dto.y_reordered, hold)
    c.ukf_ctx.z0 = &z0[0]
    c.ukf_ctx.P0 = _alloc_P0(b, kf_dto, 2 * n_state, hold)
    c.ukf_ctx.T = <int64_t>kf_dto.T
    c.ukf_ctx.n_state = n_state
    c.ukf_ctx.n_ctrl = <int64_t>sol_dto.n_ctrl
    c.ukf_ctx.n_exog = <int64_t>sol_dto.n_exog
    c.ukf_ctx.n_obs = <int64_t>kf_dto.n_obs
    c.ukf_ctx.n_params = <int64_t>sol_dto.n_par
    c.ukf_ctx.alpha = <double>kf_dto.alpha
    c.ukf_ctx.beta = <double>kf_dto.beta
    c.ukf_ctx.kappa = <double>kf_dto.kappa
    c.ukf_ctx.jitter = <double>kf_dto.jitter
    c.ukf_ctx.symmetrize = <int>bool(kf_dto.sym)
    c.ukf_ctx.store_history = 0


# --- Native MLE/MAP driver over the per-mode objective (issue #330) ----------
#
# numpy tags the BitGenerator capsule with this exact name; a mismatch makes
# PyCapsule_GetPointer reject the pointer, so a foreign capsule can't be
# dereferenced. Mirrors the rng subsystem's own unwrap.
cdef const char *_BITGEN_CAPSULE_NAME = b"BitGenerator"


cdef bitgen_t *_bitgen_ptr(object rng) except NULL:
    """Borrow the ``bitgen_t*`` from a numpy ``Generator``. The caller MUST keep
    ``rng`` alive for the whole native run (the capsule pointer is borrowed)."""
    capsule = rng.bit_generator.capsule
    if not PyCapsule_IsValid(capsule, _BITGEN_CAPSULE_NAME):
        raise ValueError(
            "random_state must resolve to a numpy Generator exposing a valid "
            "BitGenerator capsule."
        )
    return <bitgen_t *>PyCapsule_GetPointer(capsule, _BITGEN_CAPSULE_NAME)


cdef void _check_theta(object ctx_dto, object theta) except *:
    if np.shape(theta)[0] != ctx_dto.n_theta:
        raise ValueError(
            "theta length does not match the estimated parameter count."
        )


# --- actions -----------------------------------------------------------------
#
# What a wired ctx is driven with. These never touch scratch, only `ctxp` and the
# base, so they do not have to live in the frame that owns the allocations: the
# three per-mode builders below each wire their ctx and then hand off here, which
# is what keeps the driver logic from being written once per mode.
#
# `opts` carries the entry point's own arguments. It is read once per call, next
# to a marshal that already dominates it.

cdef int _ACT_ESTIMATE = 0
cdef int _ACT_MCMC = 1
cdef int _ACT_EVAL = 2


cdef void _fill_bounds(object bounds, int64_t d, double[::1] lo, double[::1] hi,
                       int64_t[::1] nbd) except *:
    """scipy's L-BFGS-B convention: none=0, lower=1, both=2, upper=3."""
    cdef int64_t i
    for i in range(d):
        lb, ub = bounds[i]
        has_lo = lb is not None
        has_hi = ub is not None
        if has_lo:
            lo[i] = lb
        if has_hi:
            hi[i] = ub
        nbd[i] = (2 if has_hi else 1) if has_lo else (3 if has_hi else 0)


cdef int _method_code(str method) except -1:
    if method == "L-BFGS-B":
        return 0
    if method == "Nelder-Mead":
        return 1
    raise ValueError(f"unsupported native method {method!r}")


cdef dict _do_estimate(void *ctxp, sdsge_obj_common *b, int filter_mode,
                       dict opts, object params):
    """Native MLE/MAP: minimize -loglik or -logpost from theta0, then resolve
    params and the log-prior at the optimum.
    """
    cdef bint has_priors = opts["has_priors"]
    b.prior.include_logjac = <int>bool(opts["include_logjac"])

    x = np.array(opts["theta0"], dtype=np.float64, copy=True)
    cdef double[::1] xv = x
    cdef int64_t d = xv.shape[0]

    cdef double[::1] lo = np.zeros(d, dtype=np.float64)
    cdef double[::1] hi = np.zeros(d, dtype=np.float64)
    cdef int64_t[::1] nbd = np.zeros(d, dtype=np.int64)
    bounds = opts["bounds"]
    cdef int has_bounds = bounds is not None
    if has_bounds:
        _fill_bounds(bounds, d, lo, hi, nbd)

    cdef sdsge_estimation_options est
    est.filter_mode = filter_mode
    est.method = _method_code(opts["method"])
    est.has_priors = has_priors
    est.lo = &lo[0]
    est.hi = &hi[0]
    est.nbd = &nbd[0] if has_bounds else NULL
    est.optim.m = opts["m"]
    est.optim.maxiter = opts["maxiter"]
    est.optim.maxfun = opts["maxfun"]
    est.optim.maxls = opts["maxls"]
    est.optim.factr = opts["factr"]
    est.optim.pgtol = opts["pgtol"]
    est.optim.fd_step = opts["fd_step"]
    est.optim.xatol = opts["xatol"]
    est.optim.fatol = opts["fatol"]
    cdef bint compute_cov = opts["compute_cov"]
    est.compute_cov = compute_cov
    est.cov_fd_step_scale = opts["cov_fd_step_scale"]
    est.cov_fd_absolute_floor = opts["cov_fd_absolute_floor"]

    vcov = np.empty((d, d), dtype=np.float64)
    se = np.empty(d, dtype=np.float64)
    cdef double[:, ::1] vcovv = vcov
    cdef double[::1] sev = se

    cdef sdsge_estimation_result res
    res.vcov = &vcovv[0, 0]
    res.se = &sev[0]

    cdef double lpr = 0.0
    with nogil:
        sdsge_run_estimation(ctxp, d, &xv[0], &est, &res)
        if has_priors:
            lpr = sdsge_logprior_at(b, &xv[0])

    return {
        "x": x,
        "params": np.array(params, dtype=np.float64, copy=True),
        "fun": res.base.fun,
        "nfev": int(res.base.nfev),
        "nit": int(res.base.nit),
        "success": bool(res.base.success),
        "status": int(res.base.status),
        "message": (
            (<bytes>res.base.message).decode() if res.base.message != NULL else ""
        ),
        "bk_violations": int(b.bk_violations),
        "logprior": float(lpr),
        "vcov": vcov if compute_cov else None,
        "se": se if compute_cov else None,
        "cov_status": int(res.cov_status),
    }


cdef dict _do_eval(void *ctxp, sdsge_obj_common *b, int filter_mode, dict opts):
    """One objective evaluation at a theta. `negate` is 0 here: these report a
    density rather than feeding a minimizer."""
    cdef double[::1] th = np.ascontiguousarray(opts["theta"], dtype=np.float64)
    cdef int has_prior = <int>bool(opts["has_prior"])
    b.prior.include_logjac = <int>bool(opts["jacobian"])
    b.bk_violations = 0
    cdef sdsge_objective_fn fn = sdsge_select_objective(0, has_prior, filter_mode)
    cdef double out
    with nogil:
        out = fn(&th[0], ctxp)
    return {"value": np.float64(out)}


cdef dict _do_mcmc(void *ctxp, sdsge_obj_common *b, int filter_mode, dict opts):
    """Native adaptive random-walk Metropolis.

    `rng` is held by `opts` for the duration, which is what keeps the borrowed
    bitgen_t pointer valid across the nogil chain."""
    b.prior.include_logjac = 1

    rng = opts["rng"]
    cdef bitgen_t *bg = _bitgen_ptr(rng)

    cdef double[::1] th0 = np.ascontiguousarray(opts["theta0"], dtype=np.float64)
    cdef int64_t d = th0.shape[0]
    if d <= 0:
        raise ValueError("No estimated parameters were provided.")

    cdef double[:, ::1] pcov = opts["proposal_cov"]

    cdef int64_t n_draws = opts["n_draws"]
    kept = np.empty((n_draws, d), dtype=np.float64)
    kept_lp = np.empty(n_draws, dtype=np.float64)
    kept_lj = np.empty(n_draws, dtype=np.float64)
    cdef double[:, ::1] keptv = kept
    cdef double[::1] keptlpv = kept_lp
    cdef double[::1] keptljv = kept_lj

    cdef sdsge_mcmc_buffers buf
    buf.kept = &keptv[0, 0]
    buf.kept_lp = &keptlpv[0]
    buf.kept_lj = &keptljv[0]

    # The MAP leg the chain may start from, driven through the same options
    # struct a point estimate uses.
    mo = opts["map_options"]
    cdef double[::1] mlo = np.zeros(d, dtype=np.float64)
    cdef double[::1] mhi = np.zeros(d, dtype=np.float64)
    cdef int64_t[::1] mnbd = np.zeros(d, dtype=np.int64)
    map_bounds = mo.get("bounds")
    cdef int has_map_bounds = map_bounds is not None
    if has_map_bounds:
        _fill_bounds(map_bounds, d, mlo, mhi, mnbd)

    cdef sdsge_estimation_options map_opt
    map_opt.filter_mode = filter_mode
    map_opt.method = _method_code(mo.get("method", "L-BFGS-B"))
    map_opt.has_priors = 1
    map_opt.lo = &mlo[0]
    map_opt.hi = &mhi[0]
    map_opt.nbd = &mnbd[0] if has_map_bounds else NULL
    map_opt.optim.m = mo.get("m", 10)
    map_opt.optim.maxiter = mo.get("maxiter", 15000)
    map_opt.optim.maxfun = mo.get("maxfun", 15000)
    map_opt.optim.maxls = mo.get("maxls", 20)
    map_opt.optim.factr = mo.get("factr", 1e7)
    map_opt.optim.pgtol = mo.get("pgtol", 1e-5)
    map_opt.optim.fd_step = mo.get("fd_step", 0.0)
    map_opt.optim.xatol = mo.get("xatol", 1e-4)
    map_opt.optim.fatol = mo.get("fatol", 1e-4)
    map_opt.compute_cov = 0
    map_opt.cov_fd_step_scale = 1.0
    map_opt.cov_fd_absolute_floor = 0.1

    cdef sdsge_mcmc_options mopt
    mopt.n_draws = n_draws
    mopt.burn_in = opts["burn_in"]
    mopt.thin = opts["thin"]
    mopt.needs_map = <int>bool(opts["compute_map"])
    mopt.adapt = <int>bool(opts["adapt"])
    mopt.adapt_start = opts["adapt_start"]
    mopt.adapt_epsilon = opts["adapt_epsilon"]
    mopt.proposal_scale = opts["proposal_scale"]
    mopt.needs_hessian = <int>bool(opts["needs_hessian"])
    mopt.hessian_fd_step_scale = opts["cov_fd_step_scale"]
    mopt.hessian_fd_absolute_floor = opts["cov_fd_absolute_floor"]

    # +logpost, unnegated: MCMC samples a posterior.
    cdef sdsge_objective_fn logpost = sdsge_select_objective(0, 1, filter_mode)

    cdef sdsge_mcmc_result res
    b.bk_violations = 0
    with nogil:
        sdsge_mcmc_run(logpost, ctxp, bg, &th0[0], d, &pcov[0, 0], &mopt,
                       &map_opt, &buf, &res)

    if res.status != 0:
        raise MemoryError(
            (<bytes>res.message).decode()
            if res.message != NULL
            else "native MCMC run failed"
        )

    return {
        "samples": kept,
        "logpost_trace": kept_lp,
        "logjac_trace": kept_lj,
        "n_accepted": int(res.n_accepted),
        "total_steps": int(res.total_steps),
        "bk_violations": int(b.bk_violations),
    }


cdef dict _run(void *ctxp, sdsge_obj_common *b, int filter_mode, int action,
               dict opts, object params):
    """What a wired ctx gets driven with."""
    if action == _ACT_ESTIMATE:
        return _do_estimate(ctxp, b, filter_mode, opts, params)
    if action == _ACT_MCMC:
        return _do_mcmc(ctxp, b, filter_mode, opts)
    return _do_eval(ctxp, b, filter_mode, opts)


# --- per-mode builders ------------------------------------------------------
#
# One per filter mode, each owning its concretely typed ctx. The ctx is a local
# of THIS frame, which is also the frame the native call runs in, and `hold` is
# what keeps every buffer alive for exactly that long. Moving the ctx or `hold`
# out of one of these would leave the struct pointing at freed memory with no
# diagnostic.

cdef dict _build_and_run_kf(object ctx_dto, int action, dict opts):
    cdef object sol = ctx_dto.solve_ctx
    cdef object kf = ctx_dto.filter_ctx
    cdef list hold = []
    cdef sdsge_linear_ctx c
    cdef sdsge_obj_common *b = &c.base

    cdef const signed char *inc = _i8p(sol.incidence, hold)
    cdef int64_t nd = sdsge_pencil_dim(inc, <int64_t>sol.n_var)
    cdef arena_size asz = sdsge_linear_obj_arena_size(
        sol.n_var, sol.n_state, sol.n_ctrl, sol.n_par, sol.n_exog,
        kf.n_obs, nd)

    params = _alloc_base(b, sol, kf.n_obs, asz, hold)
    _fill_param_map(&b.pmap, ctx_dto.pmap, hold)
    _fill_cov_build(&b.q, ctx_dto.q_spec, hold)
    _fill_cov_build(&b.r, ctx_dto.r_spec, hold)
    _fill_prior(&b.prior, ctx_dto.prior, hold)
    _fill_klein_spec(&c.solve_ctx, sol, hold, b.params, inc)
    _alloc_solve1(&c.solve_out, sol, nd, hold)
    _alloc_fill_kf(&c, kf, sol, hold)
    sdsge_init_params(b.params, b.pmap.base_params, <int64_t>sol.n_par)

    return _run(<void*>&c, b, 0, action, opts, params)


cdef dict _build_and_run_ekf(object ctx_dto, int action, dict opts):
    cdef object sol = ctx_dto.solve_ctx
    cdef object kf = ctx_dto.filter_ctx
    cdef list hold = []
    cdef sdsge_extended_ctx c
    cdef sdsge_obj_common *b = &c.base

    cdef const signed char *inc = _i8p(sol.incidence, hold)
    cdef int64_t nd = sdsge_pencil_dim(inc, <int64_t>sol.n_var)
    cdef arena_size asz = sdsge_extended_obj_arena_size(
        sol.n_var, sol.n_state, sol.n_ctrl, sol.n_par, sol.n_exog,
        kf.n_obs, nd)

    params = _alloc_base(b, sol, kf.n_obs, asz, hold)
    _fill_param_map(&b.pmap, ctx_dto.pmap, hold)
    _fill_cov_build(&b.q, ctx_dto.q_spec, hold)
    _fill_cov_build(&b.r, ctx_dto.r_spec, hold)
    _fill_prior(&b.prior, ctx_dto.prior, hold)
    _fill_klein_spec(&c.solve_ctx, sol, hold, b.params, inc)
    _alloc_solve1(&c.solve_out, sol, nd, hold)
    _alloc_fill_ekf(&c, kf, sol, hold)
    sdsge_init_params(b.params, b.pmap.base_params, <int64_t>sol.n_par)

    return _run(<void*>&c, b, 1, action, opts, params)


cdef dict _build_and_run_ukf(object ctx_dto, int action, dict opts):
    cdef object sol = ctx_dto.solve_ctx
    cdef object kf = ctx_dto.filter_ctx
    cdef list hold = []
    cdef sdsge_unscented_ctx c
    cdef sdsge_obj_common *b = &c.base

    cdef const signed char *inc = _i8p(sol.incidence, hold)
    cdef int64_t nd = sdsge_pencil_dim(inc, <int64_t>sol.n_var)
    cdef arena_size asz = sdsge_unscented_obj_arena_size(
        sol.n_var, sol.n_state, sol.n_ctrl, sol.n_par, sol.n_exog,
        kf.n_obs, nd)

    params = _alloc_base(b, sol, kf.n_obs, asz, hold)
    _fill_param_map(&b.pmap, ctx_dto.pmap, hold)
    _fill_cov_build(&b.q, ctx_dto.q_spec, hold)
    _fill_cov_build(&b.r, ctx_dto.r_spec, hold)
    _fill_prior(&b.prior, ctx_dto.prior, hold)
    _fill_klein_spec(&c.solve_ctx.first, sol, hold, b.params, inc)
    _alloc_solve1(&c.solve1_out, sol, nd, hold)
    _alloc_solve2(&c.solve2_out, sol, hold)
    _alloc_fill_ukf(&c, kf, sol, hold)
    sdsge_init_params(b.params, b.pmap.base_params, <int64_t>sol.n_par)

    return _run(<void*>&c, b, 2, action, opts, params)


cdef dict _dispatch(object ctx_dto, str mode, int action, dict opts):
    """Mode picks the builder, which owns the ctx and its scratch for the call."""
    if mode == "linear":
        return _build_and_run_kf(ctx_dto, action, opts)
    if mode == "extended":
        return _build_and_run_ekf(ctx_dto, action, opts)
    if mode == "unscented":
        return _build_and_run_ukf(ctx_dto, action, opts)
    raise ValueError(f"unsupported native filter mode {mode!r}")


def run_estimation(
    object ctx_dto,
    str method,
    double[::1] theta0,
    bounds=None,
    bint has_priors=False,
    bint include_logjac=False,
    int m=10,
    int maxiter=15000,
    int maxfun=15000,
    int maxls=20,
    double factr=1e7,
    double pgtol=1e-5,
    double fd_step=0.0,
    double xatol=1e-4,
    double fatol=1e-4,
    bint compute_cov=True,
    double cov_fd_step_scale=1.0,
    double cov_fd_absolute_floor=0.1,
):
    """Native MLE/MAP over the linear / extended / unscented objective. Marshal
    the mode's context DTO into its C ctx, then minimize ``-loglik``
    (``has_priors=0``) or ``-logpost`` (``has_priors=1``) with the native
    L-BFGS-B / Nelder-Mead driver. Returns the driver result plus ``params`` (the
    named parameter vector scattered at x_best) and ``logprior`` (MAP), all
    resolved natively with no filter re-eval.

    ``compute_cov`` also returns ``vcov``, the asymptotic covariance at the
    optimum, from a finite-difference Hessian costing
    ``n_theta * (n_theta + 1) + 1`` further objective evaluations. It is the
    covariance of theta, the vector minimized here. A Hessian that is not
    positive definite there leaves NaN throughout and reports it on
    ``cov_status``; the estimate itself is unaffected."""
    cdef str mode = ctx_dto.filter_ctx.mode
    return _dispatch(ctx_dto, mode, _ACT_ESTIMATE, {
        "theta0": theta0,
        "bounds": bounds,
        "method": method,
        "has_priors": has_priors,
        "include_logjac": include_logjac,
        "m": m,
        "maxiter": maxiter,
        "maxfun": maxfun,
        "maxls": maxls,
        "factr": factr,
        "pgtol": pgtol,
        "fd_step": fd_step,
        "xatol": xatol,
        "fatol": fatol,
        "compute_cov": compute_cov,
        "cov_fd_step_scale": cov_fd_step_scale,
        "cov_fd_absolute_floor": cov_fd_absolute_floor,
    })


def run_mcmc(
    object ctx_dto,
    double[::1] theta0,
    object rng,
    int64_t n_draws,
    int64_t burn_in=1000,
    int64_t thin=1,
    bint adapt=True,
    int64_t adapt_start=100,
    double proposal_scale=0.1,
    proposal_cov=None,
    double cov_fd_step_scale=1.0,
    double cov_fd_absolute_floor=0.1,
    double adapt_epsilon=1e-8,
    bint compute_map=True,
    dict map_options=None,
):
    """Native adaptive random-walk Metropolis over the linear / extended /
    unscented +logpost objective. Marshals the mode's context DTO once, borrows
    ``rng``'s PCG64 state, and runs the whole chain in native ``nogil`` code,
    returning the kept draws (in theta space) plus the acceptance / BK counters.
    ``rng`` must be a numpy ``Generator``; it is held for the whole native run
    (the ``bitgen_t*`` is borrowed). Draws stay bit-exact numpy; the proposal
    (Cholesky) and covariance adaptation are native (statistical, not bit,
    equivalence with the numpy chain).

    ``theta0`` is where the chain begins. With ``compute_map`` the MAP is found
    from it first and the chain begins at the mode instead; without, ``theta0``
    is taken to be that mode already and the proposal Hessian is built there."""
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    if cov_fd_step_scale <= 0.0 or cov_fd_absolute_floor <= 0.0:
        raise ValueError("Hessian finite-difference settings must be positive.")
    if map_options is None:
        map_options = {}
    unknown_map_options = set(map_options) - {
        "method", "bounds", "m", "maxiter", "maxfun", "maxls", "factr",
        "pgtol", "fd_step", "xatol", "fatol",
    }
    if unknown_map_options:
        raise ValueError(
            f"unsupported MAP option(s): {sorted(unknown_map_options)!r}"
        )

    cdef int64_t d = theta0.shape[0]
    cdef bint needs_hessian = True
    if proposal_cov is not None:
        needs_hessian = False
        if compute_map:
            raise ValueError(
                    "``compute_map=True`` will overwrite the proposal covariance. "
                    "To manually provide a proposal covariance, "
                    "set ``compute_map=False``."
                    )
        proposal_cov = np.ascontiguousarray(proposal_cov, dtype=np.float64)
        if proposal_cov.shape != (d, d):
            raise ValueError(
                "proposal_cov must be square with elements row and column counts "
                "equal to the number of estimated parameters. "
                f"Expected shape ({d}, {d}), got {proposal_cov.shape}."
            )
    else:
        proposal_cov = np.zeros((d, d), dtype=np.float64)

    cdef str mode = ctx_dto.filter_ctx.mode
    return _dispatch(ctx_dto, mode, _ACT_MCMC, {
        "theta0": theta0,
        "rng": rng,
        "n_draws": n_draws,
        "burn_in": burn_in,
        "thin": thin,
        "adapt": adapt,
        "adapt_start": adapt_start,
        "proposal_scale": proposal_scale,
        "proposal_cov": proposal_cov,
        "cov_fd_step_scale": cov_fd_step_scale,
        "cov_fd_absolute_floor": cov_fd_absolute_floor,
        "adapt_epsilon": adapt_epsilon,
        "compute_map": compute_map,
        "needs_hessian": needs_hessian,
        "map_options": map_options,
    })


# Point objectives at an arbitrary theta. Each call marshals its own context from
# the DTO and drops it on return, so nothing is shared between calls: the scratch
# arena, `include_logjac` and the BK counter are all call-local, and concurrent
# callers cannot reach each other's state.
#
# The rebuild is not an optimization choice to revisit. An `Estimator` is mutable
# between calls (`y`, `priors`, `R`, `P0`, `estimated_params` and `ss_seed` are
# all public and rebindable, and their arrays can be written in place), so a
# cached context would be stale with nothing able to detect it. Any key sound
# enough to trust would have to hash the array contents, which costs what the
# rebuild costs.


def loglik(object ctx_dto, theta not None):
    """Log-likelihood at ``theta`` (the unconstrained vector). The prior is not
    evaluated, so this is the same quantity the MLE objective maximizes."""
    _check_theta(ctx_dto, theta)
    cdef str mode = ctx_dto.filter_ctx.mode
    return _dispatch(ctx_dto, mode, _ACT_EVAL, {
        "theta": theta,
        "has_prior": False,
        "jacobian": False,
    })["value"]


def logpost(object ctx_dto, theta not None, bint jacobian=False):
    """Log-posterior at ``theta``. ``jacobian`` picks the density: with it, the
    density over theta the sampler walks; without, the prior over the parameters
    read at ``theta``. Equals the log-likelihood when the run carries no
    prior."""
    _check_theta(ctx_dto, theta)
    cdef str mode = ctx_dto.filter_ctx.mode
    return _dispatch(ctx_dto, mode, _ACT_EVAL, {
        "theta": theta,
        "has_prior": bool(ctx_dto.prior.has_prior),
        "jacobian": jacobian,
    })["value"]


def logprior(object ctx_dto, theta not None, bint jacobian=False):
    """The packed log-prior at ``theta``, read off the context's prior tables.

    ``jacobian=True`` picks the density over the theta a sampler walks. ``False``
    picks the prior over the parameters theta maps to. A run carrying no prior
    has ``has_prior`` clear and the kernel returns 0.0.

    No solve, no filter and no mode: this wires the base alone, which is all
    ``sdsge_logprior_at`` reads."""
    _check_theta(ctx_dto, theta)
    cdef int64_t n_par = ctx_dto.solve_ctx.n_par
    cdef list hold = []
    cdef sdsge_obj_common b

    cdef double[::1] paramsv = np.empty(n_par, dtype=np.float64)
    b.params = &paramsv[0]
    _fill_param_map(&b.pmap, ctx_dto.pmap, hold)
    _fill_prior(&b.prior, ctx_dto.prior, hold)
    b.prior.include_logjac = <int>bool(jacobian)
    sdsge_init_params(b.params, b.pmap.base_params, n_par)

    cdef double[::1] th = np.ascontiguousarray(theta, dtype=np.float64)
    cdef double out
    with nogil:
        out = sdsge_logprior_at(&b, &th[0])
    return np.float64(out)
