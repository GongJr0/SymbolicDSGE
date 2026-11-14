# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython composer for the native estimation objective.

Marshals Python-side prep (packed arrays + cfunc/LAPACK addresses) into the C
context struct and calls the native objective. Python never assembles the struct
by hand.
"""

from libc.stdint cimport int64_t, uint64_t

from cpython.mem cimport PyMem_Malloc, PyMem_Free

from cpython.pycapsule cimport (
    PyCapsule_GetName,
    PyCapsule_GetPointer,
)

import numpy as np
import scipy.linalg.cython_lapack as _cython_lapack

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
# holds. Appending the result keeps it alive.

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
# Each one allocates a struct's buffers and populates the given struct with pointers
# to them. The arrays are rooted in `hold`, a list the CALLER owns. That way, the
# callsite and lifetime of allocations are separately managed.

cdef void _alloc_base(sdsge_obj_common *b, object sol_dto, int64_t n_obs,
                      arena_size asz, list hold) except *:
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


cdef void _alloc_solve1(sdsge_solve1 *o, object sol_dto, int64_t nd,
                        list hold) except *:
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


cdef void _alloc_solve2(sdsge_solve2 *o, object sol_dto,
                        list hold) except *:
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
    c.kf_ctx.symmetrize = <int>bool(kf_dto.symmetrize)
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
    c.ekf_ctx.symmetrize = <int>bool(kf_dto.symmetrize)
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
    c.ukf_ctx.symmetrize = <int>bool(kf_dto.symmetrize)
    c.ukf_ctx.store_history = 0


# --- objective ctx struct fills -------------------------------------------------
#
# Per-mode fillers receive typed structs from the outer _get_obj_ctx.
# The getter takes one caller owned union slot, fills the arm the mode names,
# and hands it back as the void the native entry points take.

cdef void _fill_obj_ctx_linear(object ctx_dto, sdsge_linear_ctx *c, object sol,
                               object kf, const signed char *inc, int64_t nd,
                               arena_size asz, list hold) except *:
    cdef sdsge_obj_common *b = &c.base
    _alloc_base(b, sol, kf.n_obs, asz, hold)
    _fill_param_map(&b.pmap, ctx_dto.pmap, hold)
    _fill_cov_build(&b.q, ctx_dto.q_spec, hold)
    _fill_cov_build(&b.r, ctx_dto.r_spec, hold)
    _fill_prior(&b.prior, ctx_dto.prior, hold)
    _fill_klein_spec(&c.solve_ctx, sol, hold, b.params, inc)
    _alloc_solve1(&c.solve_out, sol, nd, hold)
    _alloc_fill_kf(c, kf, sol, hold)
    sdsge_init_params(b.params, b.pmap.base_params, <int64_t>sol.n_par)


cdef void _fill_obj_ctx_extended(object ctx_dto, sdsge_extended_ctx *c, object sol,
                                 object kf, const signed char *inc, int64_t nd,
                                 arena_size asz, list hold) except *:
    cdef sdsge_obj_common *b = &c.base
    _alloc_base(b, sol, kf.n_obs, asz, hold)
    _fill_param_map(&b.pmap, ctx_dto.pmap, hold)
    _fill_cov_build(&b.q, ctx_dto.q_spec, hold)
    _fill_cov_build(&b.r, ctx_dto.r_spec, hold)
    _fill_prior(&b.prior, ctx_dto.prior, hold)
    _fill_klein_spec(&c.solve_ctx, sol, hold, b.params, inc)
    _alloc_solve1(&c.solve_out, sol, nd, hold)
    _alloc_fill_ekf(c, kf, sol, hold)
    sdsge_init_params(b.params, b.pmap.base_params, <int64_t>sol.n_par)


cdef void _fill_obj_ctx_unscented(object ctx_dto, sdsge_unscented_ctx *c,
                                  object sol, object kf, const signed char *inc,
                                  int64_t nd, arena_size asz, list hold) except *:
    cdef sdsge_obj_common *b = &c.base
    _alloc_base(b, sol, kf.n_obs, asz, hold)
    _fill_param_map(&b.pmap, ctx_dto.pmap, hold)
    _fill_cov_build(&b.q, ctx_dto.q_spec, hold)
    _fill_cov_build(&b.r, ctx_dto.r_spec, hold)
    _fill_prior(&b.prior, ctx_dto.prior, hold)
    _fill_klein_spec(&c.solve_ctx.first, sol, hold, b.params, inc)
    _alloc_solve1(&c.solve1_out, sol, nd, hold)
    _alloc_solve2(&c.solve2_out, sol, hold)
    _alloc_fill_ukf(c, kf, sol, hold)
    sdsge_init_params(b.params, b.pmap.base_params, <int64_t>sol.n_par)


cdef void *_get_obj_ctx(object ctx_dto, str mode, list hold,
                        sdsge_obj_ctx *storage,
                        int *filter_mode) except NULL:
    cdef object sol = ctx_dto.solve_ctx
    cdef object kf = ctx_dto.filter_ctx
    cdef const signed char *inc = _i8p(sol.incidence, hold)
    cdef int64_t nd = sdsge_pencil_dim(inc, <int64_t>sol.n_var)
    cdef arena_size asz

    if mode == "linear":
        asz = sdsge_linear_obj_arena_size(
            sol.n_var, sol.n_state, sol.n_ctrl, sol.n_par, sol.n_exog,
            kf.n_obs, nd)
        _fill_obj_ctx_linear(
            ctx_dto, &storage.linear, sol, kf, inc, nd, asz, hold)
        filter_mode[0] = 0
    elif mode == "extended":
        asz = sdsge_extended_obj_arena_size(
            sol.n_var, sol.n_state, sol.n_ctrl, sol.n_par, sol.n_exog,
            kf.n_obs, nd)
        _fill_obj_ctx_extended(
            ctx_dto, &storage.extended, sol, kf, inc, nd, asz, hold)
        filter_mode[0] = 1
    elif mode == "unscented":
        asz = sdsge_unscented_obj_arena_size(
            sol.n_var, sol.n_state, sol.n_ctrl, sol.n_par, sol.n_exog,
            kf.n_obs, nd)
        _fill_obj_ctx_unscented(
            ctx_dto, &storage.unscented, sol, kf, inc, nd, asz, hold)
        filter_mode[0] = 2
    else:
        raise ValueError(f"unsupported native filter mode {mode!r}")
    return <void*>storage


# --- argument checks ---------------------------------------------------------

cdef void _check_theta(object ctx_dto, object theta) except *:
    if np.shape(theta)[0] != ctx_dto.n_theta:
        raise ValueError(
            "theta length does not match the estimated parameter count."
        )


# --- options fills -----------------------------------------------------------

cdef void _fill_nbd(double[::1] lo, double[::1] hi, int64_t d,
                    int64_t[::1] nbd) except *:
    """scipy's L-BFGS-B convention: none=0, lower=1, both=2, upper=3.

    An infinity is the box's only spelling of an absent bound, so an
    all-infinite box leaves every code at 0 and the caller hands the driver
    NULL. The comparisons are ``isfinite`` without the numpy call, NaN
    included, which compares false on both sides and reads as absent.
    """
    for i in range(d):
        has_lo = np.isfinite(lo[i])
        has_hi = np.isfinite(hi[i])
        nbd[i] = (2 if has_hi else 1) if has_lo else (3 if has_hi else 0)


cdef int _method_code(str method) except -1:
    if method == "L-BFGS-B":
        return 0
    if method == "Nelder-Mead":
        return 1
    raise ValueError(f"unsupported native method {method!r}")


cdef void _fill_map_options(sdsge_estimation_options *est, list hold,
                            int64_t d, int filter_mode,
                            dict map_options) except *:
    """The MAP leg's options, read off the mapping the public API accepts.

    The fallbacks are the native side's own: an omitted key takes one rather
    than erroring, which is what lets `map_options` be partial. `hold` roots
    the bound arrays, because `est` keeps pointers into them.
    """
    cdef double[::1] lo = _held(
        np.ascontiguousarray(map_options["lo"], dtype=np.float64), hold)
    cdef double[::1] hi = _held(
        np.ascontiguousarray(map_options["hi"], dtype=np.float64), hold)
    if lo.shape[0] != d or hi.shape[0] != d:
        raise ValueError(
            f"MAP lo and hi must run to theta0's length; got {lo.shape[0]} and "
            f"{hi.shape[0]} against {d}."
        )
    cdef int64_t[::1] nbd = _held(np.zeros(d, dtype=np.int64), hold)
    _fill_nbd(lo, hi, d, nbd)

    est.filter_mode = filter_mode
    est.method = _method_code(map_options.get("method", "L-BFGS-B"))
    est.has_priors = 1
    est.lo = &lo[0]
    est.hi = &hi[0]
    est.nbd = &nbd[0] if np.sum(nbd) > 0 else NULL
    est.optim.m = map_options.get("m", 10)
    est.optim.maxiter = map_options.get("maxiter", 15000)
    est.optim.maxfun = map_options.get("maxfun", 15000)
    est.optim.maxls = map_options.get("maxls", 20)
    est.optim.factr = map_options.get("factr", 1e7)
    est.optim.pgtol = map_options.get("pgtol", 1e-5)
    est.optim.fd_step = map_options.get("fd_step", 0.0)
    est.optim.xatol = map_options.get("xatol", 1e-4)
    est.optim.fatol = map_options.get("fatol", 1e-4)
    est.compute_cov = 0
    est.cov_fd_step_scale = 1.0
    est.cov_fd_absolute_floor = 0.1


# --- objective evaluation ----------------------------------------------------
#
# `loglik` and `logpost` share `_do_eval` and decide
# whether the logprior goes into the output.
#
# `sdsge_obj_common` is the first member of all three ctx structs,
# casting the void pointer safely reaches the common fields.


cdef object _do_eval(void *ctxp, int filter_mode, object theta,
                     bint has_prior, bint jacobian):
    """One objective evaluation at a theta. `negate` is 0 here: these report a
    density rather than feeding a minimizer."""
    cdef sdsge_obj_common *b = <sdsge_obj_common*>ctxp
    cdef double[::1] th = np.ascontiguousarray(theta, dtype=np.float64)
    b.prior.include_logjac = <int>jacobian
    b.bk_violations = 0
    cdef sdsge_objective_fn fn = sdsge_select_objective(
        0, <int>has_prior, filter_mode)
    cdef double out
    with nogil:
        out = fn(&th[0], ctxp)
    return np.float64(out)


def run_estimation(
    object ctx_dto,
    str method,
    double[::1] theta0,
    double[::1] lo,
    double[::1] hi,
    bint has_priors,
    bint include_logjac,
    int m,
    int maxiter,
    int maxfun,
    int maxls,
    double factr,
    double pgtol,
    double fd_step,
    double xatol,
    double fatol,
    bint compute_cov,
    double cov_fd_step_scale,
    double cov_fd_absolute_floor,
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
    cdef list hold = []
    cdef sdsge_obj_ctx c
    cdef int filter_mode
    cdef void *ctxp = _get_obj_ctx(ctx_dto, mode, hold, &c, &filter_mode)
    cdef sdsge_obj_common *b = <sdsge_obj_common*>ctxp
    cdef int64_t n_par = ctx_dto.solve_ctx.n_par
    b.prior.include_logjac = <int>include_logjac

    x = np.array(theta0, dtype=np.float64, copy=True)
    cdef double[::1] xv = x
    cdef int64_t d = xv.shape[0]
    if lo.shape[0] != d or hi.shape[0] != d:
        raise ValueError(
            f"lo and hi must run to theta0's length; got {lo.shape[0]} and "
            f"{hi.shape[0]} against {d}."
        )
    cdef int64_t[::1] nbd = np.zeros(d, dtype=np.int64)
    _fill_nbd(lo, hi, d, nbd)

    cdef sdsge_estimation_options est
    est.filter_mode = filter_mode
    est.method = _method_code(method)
    est.has_priors = has_priors
    est.lo = &lo[0]
    est.hi = &hi[0]
    est.nbd = &nbd[0] if np.sum(nbd) > 0 else NULL
    est.optim.m = m
    est.optim.maxiter = maxiter
    est.optim.maxfun = maxfun
    est.optim.maxls = maxls
    est.optim.factr = factr
    est.optim.pgtol = pgtol
    est.optim.fd_step = fd_step
    est.optim.xatol = xatol
    est.optim.fatol = fatol
    est.compute_cov = compute_cov
    est.cov_fd_step_scale = cov_fd_step_scale
    est.cov_fd_absolute_floor = cov_fd_absolute_floor

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
        "params": np.array(<double[:n_par]>b.params, dtype=np.float64, copy=True),
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


def run_mcmc(
    object ctx_dto,
    double[::1] theta0,
    uint64_t seed,
    int64_t n_chains,
    int64_t n_draws,
    int64_t burn_in,
    int64_t thin,
    bint adapt,
    int64_t adapt_start,
    int64_t adapt_end,
    int64_t adapt_interval,
    double adapt_epsilon,
    double init_scale,
    double proposal_scale,
    proposal_cov,
    double cov_fd_step_scale,
    double cov_fd_absolute_floor,
    bint compute_map,
    dict map_options,
):
    """Native adaptive random-walk Metropolis over the linear / extended /
    unscented objective. Returns the kept draws (in parameter space) plus
    the acceptance / BK counters.

    ``theta0`` is where the chain begins. With ``compute_map`` the MAP is found
    from it first and the chain begins at the mode instead; without, ``theta0``
    is taken to be that mode already and the proposal Hessian is built there.

    ``n_chains`` above 1 runs that many chains, each starting from a perturbation
    of chain 0's point."""
    if n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if thin <= 0:
        raise ValueError("thin must be positive.")
    if cov_fd_step_scale <= 0.0 or cov_fd_absolute_floor <= 0.0:
        raise ValueError("Hessian finite-difference settings must be positive.")
    if proposal_scale <= 0.0:
        raise ValueError("proposal_scale must be positive.")
    if adapt:
        if adapt_start <= 0:
            raise ValueError("adapt_start must be positive.")
        if adapt_start > adapt_end:
            raise ValueError(
                f"adapt_start ({adapt_start}) must not exceed adapt_end ({adapt_end})."
            )
        if adapt_interval <= 0:
            raise ValueError("adapt_interval must be positive.")
        if adapt_epsilon < 0.0:
            raise ValueError("adapt_epsilon must be non-negative.")
    if init_scale <= 0.0:
        raise ValueError("init_scale must be positive.")

    map_options = {} if map_options is None else dict(map_options)
    unknown_map_options = set(map_options) - {
        "method", "lo", "hi", "m", "maxiter", "maxfun", "maxls", "factr",
        "pgtol", "fd_step", "xatol", "fatol",
    }
    if unknown_map_options:
        raise ValueError(
            f"unsupported MAP option(s): {sorted(unknown_map_options)!r}"
        )

    cdef int64_t d = theta0.shape[0]
    if d <= 0:
        raise ValueError("No estimated parameters were provided.")
    map_options.setdefault("lo", np.full(d, -np.inf, dtype=np.float64))
    map_options.setdefault("hi", np.full(d, np.inf, dtype=np.float64))

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
        if not np.all(np.isfinite(proposal_cov)):
            raise ValueError("proposal_cov must be finite.")

        sym_tol = float(np.sqrt(np.finfo(np.float64).eps))
        asym = np.abs(proposal_cov - proposal_cov.T).max()
        if asym > sym_tol * np.abs(proposal_cov).max():
            raise ValueError(
                "proposal_cov is not symmetric under the relative tolerance "
                f"{sym_tol:.3e}."
            )
    else:
        proposal_cov = np.zeros((d, d), dtype=np.float64)

    cdef str mode = ctx_dto.filter_ctx.mode
    cdef list hold = []
    cdef int filter_mode
    cdef int64_t i
    cdef sdsge_obj_common *bi
    cdef sdsge_obj_ctx *ctxs = NULL
    cdef sdsge_mcmc_buffers *bufs = NULL
    cdef sdsge_mcmc_result *outs = NULL

    cdef double[:, ::1] pcov = proposal_cov

    # Chain-major even at n_chains == 1, so one fill loop serves both drivers.
    # The single-chain return slices chain 0 back out.
    kept = np.empty((n_chains, n_draws, d), dtype=np.float64)
    kept_lp = np.empty((n_chains, n_draws), dtype=np.float64)
    kept_lj = np.empty((n_chains, n_draws), dtype=np.float64)
    init_theta = np.empty((n_chains, d), dtype=np.float64)
    cdef double[:, :, ::1] keptv = kept
    cdef double[:, ::1] keptlpv = kept_lp
    cdef double[:, ::1] keptljv = kept_lj
    cdef double[:, ::1] thetaiv = init_theta

    cdef arena_size wsize
    cdef double[::1] work
    cdef sdsge_estimation_options map_opt
    cdef sdsge_mcmc_options mopt
    cdef sdsge_objective_fn logpost
    cdef int64_t[::1] accb, accs, nstep, bk, status
    cdef double[::1] tburn, tsamp
    cdef double wall_t, init_t
    cdef int64_t rc

    try:
        ctxs = <sdsge_obj_ctx*>PyMem_Malloc(n_chains * sizeof(sdsge_obj_ctx))
        bufs = <sdsge_mcmc_buffers*>PyMem_Malloc(
            n_chains * sizeof(sdsge_mcmc_buffers))
        outs = <sdsge_mcmc_result*>PyMem_Malloc(
            n_chains * sizeof(sdsge_mcmc_result))
        if ctxs == NULL or bufs == NULL or outs == NULL:
            raise MemoryError("could not allocate MCMC chain scaffolding")

        # One context per chain. Every mutable buffer a context points at is
        # allocated inside the fill, so chains share nothing the kernel writes;
        # what they do share is const and DTO-rooted.
        for i in range(n_chains):
            _get_obj_ctx(ctx_dto, mode, hold, &ctxs[i], &filter_mode)
            bi = <sdsge_obj_common*>&ctxs[i]
            bi.prior.include_logjac = 1
            bufs[i].kept = &keptv[i, 0, 0]
            bufs[i].kept_lp = &keptlpv[i, 0]
            bufs[i].kept_lj = &keptljv[i, 0]
            bufs[i].init_theta = &thetaiv[i, 0]

        # The MAP leg the chain may start from, driven through the same options
        # struct a point estimate uses.
        _fill_map_options(&map_opt, hold, d, filter_mode, map_options)

        mopt.n_draws = n_draws
        mopt.burn_in = burn_in
        mopt.thin = thin
        mopt.needs_map = <int>compute_map
        mopt.adapt = <int>adapt
        mopt.adapt_start = adapt_start
        mopt.adapt_end = adapt_end
        mopt.adapt_interval = adapt_interval
        mopt.adapt_epsilon = adapt_epsilon
        mopt.proposal_scale = proposal_scale
        mopt.needs_hessian = <int>needs_hessian
        mopt.hessian_fd_step_scale = cov_fd_step_scale
        mopt.hessian_fd_absolute_floor = cov_fd_absolute_floor
        mopt.init_scale = init_scale
        mopt.seed = seed
        # +logpost, unnegated: MCMC samples a posterior.
        logpost = sdsge_select_objective(0, 1, filter_mode)

        wsize = sdsge_mcmc_arena_size(d, n_chains, &ctxs[0])
        work = np.empty(wsize.n_float, np.float64)

        with nogil:
            rc = sdsge_mcmc_run(logpost, ctxs, &theta0[0], d, n_chains,
                                &pcov[0, 0], &mopt, &map_opt, bufs, outs,
                                &wall_t, &init_t, &work[0])

        if rc != 0:
            codes = []
            messages = []
            for i in range(n_chains):
                codes.append(outs[i].status)
                messages.append(
                    (<bytes>outs[i].message).decode()
                    if outs[i].message != NULL
                    else "unknown error."
                )
            raise RuntimeError(
                f"Chain(s) {[i for i, c in enumerate(codes) if c != 0]} failed.\n  "
                + "\n  ".join(
                    f"Chain {i}: {m} ({c})"
                    for i, (c, m) in enumerate(zip(codes, messages))
                    if c != 0
                )
            )

        accb = np.empty(n_chains, dtype=np.int64)
        accs = np.empty(n_chains, dtype=np.int64)
        nstep = np.empty(n_chains, dtype=np.int64)
        bk = np.empty(n_chains, dtype=np.int64)
        status = np.empty(n_chains, dtype=np.int64)
        tburn = np.empty(n_chains, dtype=np.float64)
        tsamp = np.empty(n_chains, dtype=np.float64)
        for i in range(n_chains):
            accb[i] = outs[i].accepted_burn_in
            accs[i] = outs[i].accepted_in_sample
            nstep[i] = outs[i].total_steps
            bk[i] = outs[i].bk_violations
            status[i] = outs[i].status
            tburn[i] = outs[i].burn_in_t
            tsamp[i] = outs[i].in_sample_t

        return {
            "samples": kept,
            "logpost_trace": kept_lp,
            "logjac_trace": kept_lj,
            "accepted_burn_in": accb,
            "accepted_in_sample": accs,
            "total_steps": nstep,
            "bk_violations": bk,
            "init_t": init_t,
            "wall_t": wall_t,
            "burn_in_t": tburn,
            "in_sample_t": tsamp,
            "init_theta": init_theta,
            "status": status,
        }
    finally:
        PyMem_Free(ctxs)
        PyMem_Free(bufs)
        PyMem_Free(outs)


def loglik(object ctx_dto, theta not None):
    """Log-likelihood at ``theta`` (the unconstrained vector). The prior is not
    evaluated, so this is the same quantity the MLE objective maximizes."""
    _check_theta(ctx_dto, theta)
    cdef str mode = ctx_dto.filter_ctx.mode
    cdef list hold = []
    cdef sdsge_obj_ctx c
    cdef int filter_mode
    cdef void *ctxp = _get_obj_ctx(ctx_dto, mode, hold, &c, &filter_mode)
    return _do_eval(ctxp, filter_mode, theta, False, False)


def logpost(object ctx_dto, theta not None, bint jacobian=False):
    """Log-posterior at ``theta``. ``jacobian`` picks the density: with it, the
    density over theta the sampler walks; without, the prior over the parameters
    read at ``theta``. Equals the log-likelihood when the run carries no
    prior."""
    _check_theta(ctx_dto, theta)
    cdef str mode = ctx_dto.filter_ctx.mode
    cdef list hold = []
    cdef sdsge_obj_ctx c
    cdef int filter_mode
    cdef void *ctxp = _get_obj_ctx(ctx_dto, mode, hold, &c, &filter_mode)
    return _do_eval(ctxp, filter_mode, theta,
                    bool(ctx_dto.prior.has_prior), jacobian)


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
