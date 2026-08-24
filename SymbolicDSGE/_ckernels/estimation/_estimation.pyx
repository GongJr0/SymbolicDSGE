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
from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
import scipy.linalg.cython_lapack as _cython_lapack

from numpy.random cimport bitgen_t


cdef extern from "../_common/sdsge_complex.h":
    ctypedef struct c128:
        double re
        double im


cdef extern from "optim.h":
    ctypedef double (*sdsge_objective_fn)(const double *x, void *ctx) noexcept nogil

    ctypedef struct sdsge_optim_options:
        int64_t m
        int64_t maxiter
        int64_t maxfun
        int64_t maxls
        double factr
        double pgtol
        double fd_step
        double xatol
        double fatol

    ctypedef struct sdsge_optim_result:
        int64_t status
        int64_t nfev
        int64_t nit
        double fun
        int success
        const char *message

cdef extern from "estimation.h":
    ctypedef void (*sdsge_residual_fn)()
    ctypedef void (*bc_residual_fn)()
    ctypedef void (*klein_zgges_fn)()
    ctypedef void (*sdsge_dgeqrf_fn)()
    ctypedef void (*sdsge_dormqr_fn)()
    int64_t sdsge_pencil_dim(const signed char *incidence, int64_t n_var)
    ctypedef void (*meas_fn)()

    ctypedef struct sdsge_dims:
        int64_t n_theta
        int64_t n_var
        int64_t n_state
        int64_t n_ctrl
        int64_t n_exog
        int64_t n_obs
        int64_t n_par
        int64_t T

    ctypedef struct sdsge_scalar_scatter:
        int64_t theta_idx
        int64_t param_slot
        int64_t transform_code
        double transform_params[3]

    ctypedef struct sdsge_param_map:
        const double *base_params
        const sdsge_scalar_scatter *scalars
        int64_t n_scalars

    ctypedef struct sdsge_cov_spec:
        int is_constant
        const double *constant
        int64_t K
        const int64_t *std_slots
        int corr_from_block
        int64_t block_theta_off
        int64_t block_theta_len
        const int64_t *pair_i
        const int64_t *pair_j
        const int64_t *pair_slot
        int64_t n_pairs

    ctypedef struct sdsge_prior_tables:
        int has_prior
        const int64_t *scalar_indices
        const int64_t *scalar_dist_codes
        const int64_t *scalar_transform_codes
        const double *scalar_dist_params
        const double *scalar_transform_params
        int64_t n_scalar
        const int64_t *matrix_offsets
        const int64_t *matrix_dims
        const int64_t *matrix_lengths
        const double *matrix_etas
        const double *matrix_log_constants
        int64_t n_blocks
        int include_logjac

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

    ctypedef struct sdsge_obj_common:
        sdsge_dims dims
        sdsge_residual_fn residual
        bc_residual_fn bc_residual
        klein_zgges_fn zgges
        sdsge_dgeqrf_fn dgeqrf
        sdsge_dormqr_fn dormqr
        meas_fn meas
        meas_fn jac
        const double *ss_seed
        const signed char *incidence
        const double *y
        double *P0
        const double *x0
        double jitter
        int symmetrize
        int joseph_cov
        int derive_P0
        sdsge_param_map pmap
        sdsge_cov_spec q_spec
        sdsge_cov_spec r_spec
        sdsge_prior_tables prior
        double *params
        double *Q
        double *chol
        double *R
        double *corr_q
        double *corr_r
        double *std_q
        double *std_r
        double *filter_arena
        double *solve_arena
        int64_t *solve_iarena
        int64_t bk_violations

    ctypedef struct sdsge_linear_ctx:
        sdsge_obj_common base
        sdsge_solve1 solve
        double *C
        double *d

    ctypedef struct sdsge_extended_ctx:
        sdsge_obj_common base
        sdsge_solve1 solve

    ctypedef struct sdsge_solve2:
        double *f_xx
        double *bx
        double *gxx
        double *hxx
        double *gxu
        double *hxu
        double *guu
        double *huu
        double *gss
        double *hss

    ctypedef struct sdsge_unscented_ctx:
        sdsge_obj_common base
        sdsge_solve1 solve
        sdsge_solve2 solve2
        double *z0
        double alpha
        double beta
        double kappa

    ctypedef struct sdsge_estimation_options:
        int filter_mode
        int method
        int has_priors
        const double *lo
        const double *hi
        const int64_t *nbd
        sdsge_optim_options optim
        int compute_cov
        double cov_fd_step_scale
        double cov_fd_absolute_floor

    ctypedef struct sdsge_estimation_result:
        sdsge_optim_result base
        double *vcov
        double *se
        int64_t cov_status

    void sdsge_init_params(double *params, const double *base_params,
                           int64_t n_par) nogil
    void sdsge_scatter_params(sdsge_obj_common *base, const double *theta) nogil
    double sdsge_logprior_at(const sdsge_obj_common *base,
                             const double *theta) nogil
    double sdsge_obj_linear(sdsge_linear_ctx *ctx, const double *theta,
                            int has_priors) nogil
    double sdsge_obj_extended(sdsge_extended_ctx *ctx, const double *theta,
                              int has_priors) nogil
    double sdsge_obj_unscented(sdsge_unscented_ctx *ctx, const double *theta,
                               int has_priors) nogil

    sdsge_objective_fn sdsge_select_objective(int negate, int has_priors,
                                              int filter_mode) nogil
    double sdsge_post_linear(const double *x, void *ctx) noexcept nogil
    double sdsge_post_extended(const double *x, void *ctx) noexcept nogil
    double sdsge_post_unscented(const double *x, void *ctx) noexcept nogil

    void sdsge_run_estimation(void *ctx, int64_t n_theta, double *theta,
                              const sdsge_estimation_options *opt,
                              sdsge_estimation_result *out) noexcept nogil

cdef extern from "sdsge_common.h":
    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int

    int SDSGE_OK


cdef extern from "sdsge_linalg.h":
    int sdsge_chol(const double *S, double jitter, double *L, int64_t n) nogil


cdef extern from "../core/klein_solve.h":
    arena_size sdsge_klein_solve1_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t nd) nogil
    arena_size sdsge_sgu_klein_solve2_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t nd) nogil


cdef extern from "../kalman/kalman.h":
    arena_size kf_arena_size(int64_t n, int64_t m, int64_t k) nogil
    arena_size ekf_arena_size(int64_t n, int64_t m, int64_t k) nogil
    arena_size ukf_arena_size(
        int64_t n_state, int64_t n_ctrl, int64_t n_exog, int64_t n_obs,
    ) nogil


cdef extern from "mcmc.h":
    ctypedef struct sdsge_mcmc_options:
        int64_t n_draws
        int64_t burn_in
        int64_t thin
        int needs_map
        int adapt
        int64_t adapt_start
        double adapt_epsilon
        double proposal_scale
        int needs_hessian
        double hessian_fd_step_scale
        double hessian_fd_absolute_floor

    ctypedef struct sdsge_mcmc_buffers:
        double *kept
        double *kept_lp
        double *kept_lj

    ctypedef struct sdsge_mcmc_result:
        int64_t n_accepted
        int64_t total_steps
        int64_t bk_violations
        int64_t status
        const char *message

    int64_t sdsge_mcmc_run(sdsge_objective_fn logpost, void *obj_ctx,
                           bitgen_t *bg, const double *theta0,
                           int64_t d, const double *hessian,
                           const sdsge_mcmc_options *opt,
                           const sdsge_estimation_options *map_opt,
                           sdsge_mcmc_buffers *buf,
                           sdsge_mcmc_result *out) nogil


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


cdef _factor_shock_cov(double[:, ::1] cov, double[:, ::1] out):
    if sdsge_chol(&cov[0, 0], 0.0, &out[0, 0], cov.shape[0]) != SDSGE_OK:
        raise ValueError(
            "The shock covariance is not positive definite, so it has no "
            "Cholesky factor for the solve to load the innovations through."
        )

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


cdef class _NativeCtx:
    """Owns a fully-marshalled native objective context. The C ctx pointers
    (``b`` / ``s1`` / ``ctxp``) are valid only while this holder is alive: it
    retains every backing buffer in ``keep`` and the malloc'd scatter table in
    ``scalars_c`` (freed on dealloc). Shared base for the MLE/MAP driver (#330)
    and the MCMC mainloop (#331); callers add their own working theta, bounds and
    objective wiring on top."""
    cdef sdsge_linear_ctx lctx
    cdef sdsge_extended_ctx ectx
    cdef sdsge_unscented_ctx uctx
    cdef sdsge_obj_common *b
    cdef sdsge_solve1 *s1
    cdef void *ctxp
    cdef sdsge_scalar_scatter *scalars_c
    cdef list keep
    cdef object params
    cdef int has_prior
    cdef int64_t n_theta

    def __cinit__(self):
        self.b = NULL
        self.s1 = NULL
        self.ctxp = NULL
        self.scalars_c = NULL
        self.keep = []
        self.params = None
        self.has_prior = 0
        self.n_theta = 0

    def __dealloc__(self):
        if self.scalars_c != NULL:
            PyMem_Free(self.scalars_c)
            self.scalars_c = NULL


cdef _NativeCtx _build_native_ctx(object ctx_dto, str mode):
    """Marshal a mode's context DTO into its C ctx. Returns a holder that owns the
    ctx and every buffer it points into; the raw pointers stay valid as long as
    the returned holder is referenced. Mode is validated here; the objective
    trampoline is chosen by each caller (minimized for MLE/MAP, +logpost for
    MCMC)."""
    cdef _NativeCtx nc = _NativeCtx()
    cdef object base = ctx_dto.base
    cdef object dims = base.dims
    cdef int64_t n_theta = dims.n_theta
    cdef int64_t n_var = dims.n_var
    cdef int64_t n_state = dims.n_state
    cdef int64_t n_ctrl = dims.n_ctrl
    cdef int64_t n_exog = dims.n_exog
    cdef int64_t n_obs = dims.n_obs
    cdef int64_t n_par = dims.n_par
    cdef int64_t T = dims.T
    cdef int64_t n2 = 3 * n_var + n_exog
    cdef int derive_P0
    nc.n_theta = n_theta

    # Pinned inputs. Python guarantees dtype; C-contiguity is enforced here. Each
    # backing array is retained in nc.keep so the raw pointers below outlive the
    # transient memoryviews.
    _ss_seed = np.ascontiguousarray(base.ss_seed, dtype=np.float64)
    nc.keep.append(_ss_seed)
    cdef double[::1] ssv = _ss_seed
    _incidence = np.ascontiguousarray(base.incidence, dtype=np.int8)
    nc.keep.append(_incidence)
    cdef signed char[::1] incv = _incidence
    cdef int64_t nd = sdsge_pencil_dim(&incv[0], n_var)
    _y = np.ascontiguousarray(base.y, dtype=np.float64)
    nc.keep.append(_y)
    cdef double[:, ::1] yv = _y
    # P0 is square in the *filter's* dimension, not the model's: the unscented
    # filter runs on the augmented state, so its P0 is 2*n_state a side while
    # the other two are n_var. Sizing this from n_var alone hands the UKF a
    # buffer shorter than it reads.
    cdef int64_t n_filter = 2 * n_state if mode == "unscented" else n_var
    if base.P0 is None:
        _P0 = np.zeros((n_filter, n_filter), dtype=np.float64)
        derive_P0 = 1
    else:
        _P0 = np.ascontiguousarray(base.P0, dtype=np.float64)
        derive_P0 = 0
    nc.keep.append(_P0)
    cdef double[:, ::1] P0v = _P0
    _bp = np.ascontiguousarray(base.pmap.base_params, dtype=np.float64)
    nc.keep.append(_bp)
    cdef double[::1] bpv = _bp
    # The linear kf dereferences x0 unconditionally (no NULL guard), so a missing
    # x0 materializes as the zero initial state, matching obj_linear_base.
    cdef double[::1] x0v
    if base.x0 is not None:
        _x0 = np.ascontiguousarray(base.x0, dtype=np.float64)
    else:
        _x0 = np.zeros(n_var, dtype=np.float64)
    nc.keep.append(_x0)
    x0v = _x0

    # Scratch the objective writes (allocated from dims, kept alive in nc.keep).
    params = np.empty(n_par, dtype=np.float64)
    nc.params = params
    nc.keep.append(params)
    cdef double[::1] paramsv = params
    ss = np.empty(n_var, dtype=np.float64)
    nc.keep.append(ss)
    cdef double[::1] ssv2 = ss
    a_real = np.empty((n_var, n_var), dtype=np.float64)
    b_real = np.empty((n_var, n_var), dtype=np.float64)
    c_real = np.empty((n_var, n_var), dtype=np.float64)
    d_real = np.empty((n_var, n_exog), dtype=np.float64)
    order = np.empty(n_var, dtype=np.int64)
    nc.keep.append(a_real)
    nc.keep.append(b_real)
    nc.keep.append(c_real)
    nc.keep.append(d_real)
    nc.keep.append(order)
    cdef double[:, ::1] arv = a_real
    cdef double[:, ::1] brv = b_real
    cdef double[:, ::1] crealv = c_real
    cdef double[:, ::1] drealv = d_real
    cdef int64_t[::1] orderv = order
    s = np.empty((nd, nd), dtype=np.complex128, order="F")
    t = np.empty((nd, nd), dtype=np.complex128, order="F")
    z = np.empty((nd, nd), dtype=np.complex128, order="F")
    nc.keep.append(s)
    nc.keep.append(t)
    nc.keep.append(z)
    cdef double complex[::1, :] sv = s
    cdef double complex[::1, :] tv = t
    cdef double complex[::1, :] zv = z
    f = np.empty((n_ctrl, n_state), dtype=np.float64)
    p = np.empty((n_state, n_state), dtype=np.float64)
    eig = np.empty(nd, dtype=np.complex128)
    nc.keep.append(f)
    nc.keep.append(p)
    nc.keep.append(eig)
    cdef double[:, ::1] fv = f
    cdef double[:, ::1] pv = p
    cdef double complex[::1] eigv = eig
    A = np.empty((n_var, n_var), dtype=np.float64)
    B = np.empty((n_var, n_exog), dtype=np.float64)
    C = np.empty((n_obs, n_var), dtype=np.float64)
    d = np.empty(n_obs, dtype=np.float64)
    nc.keep.append(A)
    nc.keep.append(B)
    nc.keep.append(C)
    nc.keep.append(d)
    cdef double[:, ::1] Av = A
    cdef double[:, ::1] Bv = B
    cdef double[:, ::1] Cv = C
    cdef double[::1] dv = d

    # Unscented-only second-order scratch (allocated in that branch).
    cdef double[:, :, ::1] fxxv
    cdef double[:, ::1] bxv
    cdef double[:, :, ::1] gxxv
    cdef double[:, :, ::1] hxxv
    cdef double[:, :, ::1] gxuv
    cdef double[:, :, ::1] hxuv
    cdef double[:, :, ::1] guuv
    cdef double[:, :, ::1] huuv
    cdef double[::1] gssv
    cdef double[::1] hssv
    cdef double[::1] z0v
    Q = np.empty((n_exog, n_exog), dtype=np.float64)
    R = np.empty((n_obs, n_obs), dtype=np.float64)
    corr_q = np.empty((n_exog, n_exog), dtype=np.float64)
    corr_r = np.empty((n_obs, n_obs), dtype=np.float64)
    std_q = np.empty(n_exog, dtype=np.float64)
    std_r = np.empty(n_obs, dtype=np.float64)
    nc.keep.append(Q)
    nc.keep.append(R)
    nc.keep.append(corr_q)
    nc.keep.append(corr_r)
    nc.keep.append(std_q)
    nc.keep.append(std_r)
    cdef double[:, ::1] Qv = Q
    cdef double[:, ::1] Rv = R
    cdef double[:, ::1] cqv = corr_q
    cdef double[:, ::1] crv = corr_r
    cdef double[::1] sqv = std_q
    cdef double[::1] srv = std_r

    # Scatter array-of-structs (malloc'd; freed by nc.__dealloc__).
    scalars_list = base.pmap.scalars
    cdef int64_t n_scalars = len(scalars_list)
    nc.scalars_c = <sdsge_scalar_scatter *>PyMem_Malloc(
        (n_scalars if n_scalars > 0 else 1) * sizeof(sdsge_scalar_scatter)
    )
    if nc.scalars_c == NULL:
        raise MemoryError()
    cdef sdsge_scalar_scatter *scalars_c = nc.scalars_c
    cdef int64_t si
    cdef object sc
    cdef double[::1] tpv
    for si in range(n_scalars):
        sc = scalars_list[si]
        scalars_c[si].theta_idx = <int64_t>sc.theta_idx
        scalars_c[si].param_slot = <int64_t>sc.param_slot
        scalars_c[si].transform_code = <int64_t>sc.transform_code
        tpv = np.ascontiguousarray(sc.transform_params, dtype=np.float64)
        scalars_c[si].transform_params[0] = tpv[0]
        scalars_c[si].transform_params[1] = tpv[1]
        scalars_c[si].transform_params[2] = tpv[2]

    # Cov-spec pinned arrays (declared up front, assigned per regime).
    cdef object qs = base.q_spec
    cdef object rs = base.r_spec
    cdef double[:, ::1] q_const_v
    cdef double[:, ::1] r_const_v
    cdef int64_t[::1] q_std_v
    cdef int64_t[::1] r_std_v
    cdef int64_t[::1] q_pi_v
    cdef int64_t[::1] q_pj_v
    cdef int64_t[::1] q_ps_v
    cdef int64_t[::1] r_pi_v
    cdef int64_t[::1] r_pj_v
    cdef int64_t[::1] r_ps_v
    cdef int64_t q_np = 0
    cdef int64_t r_np = 0

    # Prior pinned arrays.
    cdef object pr = base.prior
    cdef int has_prior = int(pr.has_prior)
    nc.has_prior = has_prior
    cdef int64_t[::1] p_si_v
    cdef int64_t[::1] p_sdc_v
    cdef int64_t[::1] p_stc_v
    cdef double[:, ::1] p_sdp_v
    cdef double[:, ::1] p_stp_v
    cdef int64_t[::1] p_mo_v
    cdef int64_t[::1] p_md_v
    cdef int64_t[::1] p_ml_v
    cdef double[::1] p_me_v
    cdef double[::1] p_mlc_v
    cdef double[::1] filter_arena_v
    cdef double[::1] solve_arena_v
    cdef int64_t[::1] solve_iarena_v
    cdef arena_size solve_sz
    cdef int64_t p_nscalar = 0
    cdef int64_t p_nblocks = 0

    # Mode dispatch: pick the ctx and its base/solve1 pointers + the void* the
    # driver sees. The objective trampoline is chosen by each caller.
    if mode == "linear":
        nc.b = &nc.lctx.base
        nc.s1 = &nc.lctx.solve
        nc.ctxp = <void*>&nc.lctx
    elif mode == "extended":
        nc.b = &nc.ectx.base
        nc.s1 = &nc.ectx.solve
        nc.ctxp = <void*>&nc.ectx
    elif mode == "unscented":
        nc.b = &nc.uctx.base
        nc.s1 = &nc.uctx.solve
        nc.ctxp = <void*>&nc.uctx
    else:
        raise ValueError(f"unsupported native filter mode {mode!r}")
    cdef sdsge_obj_common *b = nc.b
    cdef sdsge_solve1 *s1 = nc.s1

    b.dims.n_theta = n_theta
    b.dims.n_var = n_var
    b.dims.n_state = n_state
    b.dims.n_ctrl = n_ctrl
    b.dims.n_exog = n_exog
    b.dims.n_obs = n_obs
    b.dims.n_par = n_par
    b.dims.T = T

    b.residual = <sdsge_residual_fn><void*><size_t>base.residual_addr
    b.bc_residual = <bc_residual_fn><void*><size_t>base.bc_residual_addr
    b.zgges = _zgges
    b.dgeqrf = _dgeqrf
    b.dormqr = _dormqr
    b.meas = <meas_fn><void*><size_t>base.meas_addr
    b.jac = <meas_fn><void*><size_t>base.jac_addr

    b.ss_seed = &ssv[0]
    b.incidence = &incv[0]
    b.y = &yv[0, 0]
    b.P0 = &P0v[0, 0]
    b.x0 = &x0v[0]
    b.jitter = base.jitter
    b.symmetrize = int(base.symmetrize)
    b.joseph_cov = int(base.joseph_cov)
    b.derive_P0 = derive_P0

    b.pmap.base_params = &bpv[0]
    b.pmap.scalars = scalars_c if n_scalars > 0 else NULL
    b.pmap.n_scalars = n_scalars

    b.q_spec.is_constant = int(qs.is_constant)
    b.q_spec.K = <int64_t>qs.K
    b.q_spec.corr_from_block = int(qs.corr_from_block)
    b.q_spec.block_theta_off = <int64_t>qs.block_theta_off
    b.q_spec.block_theta_len = <int64_t>qs.block_theta_len
    if qs.is_constant:
        _qc = np.ascontiguousarray(qs.constant, dtype=np.float64)
        nc.keep.append(_qc)
        q_const_v = _qc
        b.q_spec.constant = &q_const_v[0, 0]
        b.q_spec.std_slots = NULL
        b.q_spec.pair_i = NULL
        b.q_spec.pair_j = NULL
        b.q_spec.pair_slot = NULL
        b.q_spec.n_pairs = 0
    else:
        b.q_spec.constant = NULL
        _qstd = np.ascontiguousarray(qs.std_slots, dtype=np.int64)
        nc.keep.append(_qstd)
        q_std_v = _qstd
        b.q_spec.std_slots = &q_std_v[0]
        _qpi = np.ascontiguousarray(qs.pair_i, dtype=np.int64)
        _qpj = np.ascontiguousarray(qs.pair_j, dtype=np.int64)
        _qps = np.ascontiguousarray(qs.pair_slot, dtype=np.int64)
        nc.keep.append(_qpi)
        nc.keep.append(_qpj)
        nc.keep.append(_qps)
        q_pi_v = _qpi
        q_pj_v = _qpj
        q_ps_v = _qps
        q_np = q_pi_v.shape[0]
        b.q_spec.n_pairs = q_np
        b.q_spec.pair_i = &q_pi_v[0] if q_np > 0 else NULL
        b.q_spec.pair_j = &q_pj_v[0] if q_np > 0 else NULL
        b.q_spec.pair_slot = &q_ps_v[0] if q_np > 0 else NULL

    b.r_spec.is_constant = int(rs.is_constant)
    b.r_spec.K = <int64_t>rs.K
    b.r_spec.corr_from_block = int(rs.corr_from_block)
    b.r_spec.block_theta_off = <int64_t>rs.block_theta_off
    b.r_spec.block_theta_len = <int64_t>rs.block_theta_len
    if rs.is_constant:
        _rc = np.ascontiguousarray(rs.constant, dtype=np.float64)
        nc.keep.append(_rc)
        r_const_v = _rc
        b.r_spec.constant = &r_const_v[0, 0]
        b.r_spec.std_slots = NULL
        b.r_spec.pair_i = NULL
        b.r_spec.pair_j = NULL
        b.r_spec.pair_slot = NULL
        b.r_spec.n_pairs = 0
    else:
        b.r_spec.constant = NULL
        _rstd = np.ascontiguousarray(rs.std_slots, dtype=np.int64)
        nc.keep.append(_rstd)
        r_std_v = _rstd
        b.r_spec.std_slots = &r_std_v[0]
        _rpi = np.ascontiguousarray(rs.pair_i, dtype=np.int64)
        _rpj = np.ascontiguousarray(rs.pair_j, dtype=np.int64)
        _rps = np.ascontiguousarray(rs.pair_slot, dtype=np.int64)
        nc.keep.append(_rpi)
        nc.keep.append(_rpj)
        nc.keep.append(_rps)
        r_pi_v = _rpi
        r_pj_v = _rpj
        r_ps_v = _rps
        r_np = r_pi_v.shape[0]
        b.r_spec.n_pairs = r_np
        b.r_spec.pair_i = &r_pi_v[0] if r_np > 0 else NULL
        b.r_spec.pair_j = &r_pj_v[0] if r_np > 0 else NULL
        b.r_spec.pair_slot = &r_ps_v[0] if r_np > 0 else NULL

    b.prior.has_prior = has_prior

    if has_prior:
        _psi = np.ascontiguousarray(pr.scalar_indices, dtype=np.int64)
        _psdc = np.ascontiguousarray(pr.scalar_dist_codes, dtype=np.int64)
        _pstc = np.ascontiguousarray(pr.scalar_transform_codes, dtype=np.int64)
        _psdp = np.ascontiguousarray(pr.scalar_dist_params, dtype=np.float64)
        _pstp = np.ascontiguousarray(pr.scalar_transform_params, dtype=np.float64)
        _pmo = np.ascontiguousarray(pr.matrix_offsets, dtype=np.int64)
        _pmd = np.ascontiguousarray(pr.matrix_dims, dtype=np.int64)
        _pml = np.ascontiguousarray(pr.matrix_lengths, dtype=np.int64)
        _pme = np.ascontiguousarray(pr.matrix_etas, dtype=np.float64)
        _pmlc = np.ascontiguousarray(pr.matrix_log_constants, dtype=np.float64)
        for _a in (_psi, _psdc, _pstc, _psdp, _pstp,
                   _pmo, _pmd, _pml, _pme, _pmlc):
            nc.keep.append(_a)
        p_si_v = _psi
        p_sdc_v = _psdc
        p_stc_v = _pstc
        p_sdp_v = _psdp
        p_stp_v = _pstp
        p_mo_v = _pmo
        p_md_v = _pmd
        p_ml_v = _pml
        p_me_v = _pme
        p_mlc_v = _pmlc
        p_nscalar = p_si_v.shape[0]
        p_nblocks = p_mo_v.shape[0]
        b.prior.scalar_indices = &p_si_v[0] if p_nscalar > 0 else NULL
        b.prior.scalar_dist_codes = &p_sdc_v[0] if p_nscalar > 0 else NULL
        b.prior.scalar_transform_codes = &p_stc_v[0] if p_nscalar > 0 else NULL
        b.prior.scalar_dist_params = &p_sdp_v[0, 0] if p_nscalar > 0 else NULL
        b.prior.scalar_transform_params = &p_stp_v[0, 0] if p_nscalar > 0 else NULL
        b.prior.n_scalar = p_nscalar
        b.prior.matrix_offsets = &p_mo_v[0] if p_nblocks > 0 else NULL
        b.prior.matrix_dims = &p_md_v[0] if p_nblocks > 0 else NULL
        b.prior.matrix_lengths = &p_ml_v[0] if p_nblocks > 0 else NULL
        b.prior.matrix_etas = &p_me_v[0] if p_nblocks > 0 else NULL
        b.prior.matrix_log_constants = &p_mlc_v[0] if p_nblocks > 0 else NULL
        b.prior.n_blocks = p_nblocks
    else:
        b.prior.n_scalar = 0
        b.prior.n_blocks = 0

    b.params = &paramsv[0]
    b.Q = &Qv[0, 0]
    b.chol = NULL
    b.R = &Rv[0, 0]
    b.corr_q = &cqv[0, 0]
    b.corr_r = &crv[0, 0]
    b.std_q = &sqv[0]
    b.std_r = &srv[0]
    b.bk_violations = 0

    if mode == "linear":
        filter_arena = np.empty(
            kf_arena_size(n_var, n_obs, n_exog).n_float, dtype=np.float64
        )
    elif mode == "extended":
        filter_arena = np.empty(
            ekf_arena_size(n_var, n_obs, n_exog).n_float, dtype=np.float64
        )
    else:
        filter_arena = np.empty(
            ukf_arena_size(n_state, n_ctrl, n_exog, n_obs).n_float,
            dtype=np.float64,
        )
    nc.keep.append(filter_arena)
    filter_arena_v = filter_arena
    b.filter_arena = &filter_arena_v[0]

    if mode == "unscented":
        solve_sz = sdsge_sgu_klein_solve2_arena_size(
            n_var, n_state, n_ctrl, n_par, n_exog, nd
        )
    else:
        solve_sz = sdsge_klein_solve1_arena_size(
            n_var, n_state, n_ctrl, n_par, n_exog, nd)
    solve_arena = np.empty(solve_sz.n_float, dtype=np.float64)
    solve_iarena = np.empty(solve_sz.n_int, dtype=np.int64)
    nc.keep.append(solve_arena)
    nc.keep.append(solve_iarena)
    solve_arena_v = solve_arena
    solve_iarena_v = solve_iarena
    b.solve_arena = &solve_arena_v[0]
    b.solve_iarena = &solve_iarena_v[0]

    s1.ss = &ssv2[0]
    s1.a_real = &arv[0, 0]
    s1.b_real = &brv[0, 0]
    s1.c_real = &crealv[0, 0]
    s1.d_real = &drealv[0, 0] if n_exog > 0 else NULL
    s1.s = <c128*>&sv[0, 0]
    s1.t = <c128*>&tv[0, 0]
    s1.z = <c128*>&zv[0, 0]
    s1.f = &fv[0, 0]
    s1.p = &pv[0, 0]
    s1.eig = <c128*>&eigv[0]
    s1.A = &Av[0, 0]
    s1.B = &Bv[0, 0]
    s1.order = &orderv[0]

    # Mode-specific ctx wiring.
    if mode == "linear":
        nc.lctx.C = &Cv[0, 0]
        nc.lctx.d = &dv[0]
    elif mode == "unscented":
        f_xx = np.empty((n_var, n2, n2), dtype=np.float64)
        bx = np.empty((n_state, n_exog), dtype=np.float64)
        gxx = np.empty((n_ctrl, n_state, n_state), dtype=np.float64)
        hxx = np.empty((n_state, n_state, n_state), dtype=np.float64)
        gxu = np.empty((n_ctrl, n_state, n_exog), dtype=np.float64)
        hxu = np.empty((n_state, n_state, n_exog), dtype=np.float64)
        guu = np.empty((n_ctrl, n_exog, n_exog), dtype=np.float64)
        huu = np.empty((n_state, n_exog, n_exog), dtype=np.float64)
        gss = np.empty(n_ctrl, dtype=np.float64)
        hss = np.empty(n_state, dtype=np.float64)
        z0 = np.ascontiguousarray(ctx_dto.z0, dtype=np.float64)
        for _a in (f_xx, bx, gxx, hxx, gxu, hxu, guu, huu, gss, hss, z0):
            nc.keep.append(_a)
        fxxv = f_xx
        bxv = bx
        gxxv = gxx
        hxxv = hxx
        gxuv = gxu
        hxuv = hxu
        guuv = guu
        huuv = huu
        gssv = gss
        hssv = hss
        z0v = z0
        nc.uctx.solve2.f_xx = &fxxv[0, 0, 0]
        nc.uctx.solve2.bx = &bxv[0, 0]
        nc.uctx.solve2.gxx = &gxxv[0, 0, 0]
        nc.uctx.solve2.hxx = &hxxv[0, 0, 0]
        nc.uctx.solve2.gxu = &gxuv[0, 0, 0]
        nc.uctx.solve2.hxu = &hxuv[0, 0, 0]
        nc.uctx.solve2.guu = &guuv[0, 0, 0]
        nc.uctx.solve2.huu = &huuv[0, 0, 0]
        nc.uctx.solve2.gss = &gssv[0]
        nc.uctx.solve2.hss = &hssv[0]
        nc.uctx.z0 = &z0v[0]
        nc.uctx.alpha = ctx_dto.alpha
        nc.uctx.beta = ctx_dto.beta
        nc.uctx.kappa = ctx_dto.kappa

    # Seed the calibrated baseline once; the per-eval fill touches only the
    # estimated slots.
    sdsge_init_params(&paramsv[0], &bpv[0], n_par)
    return nc


def run_estimation(
    object ctx_dto,
    str mode,
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
    resolved natively with no filter re-eval. The base marshaling is shared; only
    the ctx struct, objective, and mode scratch differ per ``mode``.

    ``compute_cov`` also returns ``vcov``, the asymptotic covariance at the
    optimum, from a finite-difference Hessian costing
    ``n_theta * (n_theta + 1) + 1`` further objective evaluations. It is the
    covariance of theta, the vector minimized here. A Hessian that is not
    positive definite there leaves NaN throughout and reports it on
    ``cov_status``; the estimate itself is unaffected."""
    cdef _NativeCtx nc = _build_native_ctx(ctx_dto, mode)
    cdef sdsge_obj_common *b = nc.b
    b.prior.include_logjac = include_logjac

    cdef void *ctxp = nc.ctxp
    cdef int64_t n_theta = nc.n_theta

    # Working theta (the driver mutates it in place).
    x = np.array(theta0, dtype=np.float64, copy=True)
    cdef double[::1] xv = x

    # Bounds -> lo/hi/nbd (scipy map: none=0, lower=1, both=2, upper=3).
    cdef double[::1] lo = np.zeros(n_theta, dtype=np.float64)
    cdef double[::1] hi = np.zeros(n_theta, dtype=np.float64)
    cdef int64_t[::1] nbd = np.zeros(n_theta, dtype=np.int64)
    cdef int has_bounds = bounds is not None
    cdef int64_t bi
    if has_bounds:
        for bi in range(n_theta):
            lb, ub = bounds[bi]
            has_lo = lb is not None
            has_hi = ub is not None
            if has_lo:
                lo[bi] = lb
            if has_hi:
                hi[bi] = ub
            nbd[bi] = (2 if has_hi else 1) if has_lo else (3 if has_hi else 0)
    cdef const int64_t *nbd_ptr = &nbd[0] if has_bounds else NULL

    cdef int filter_mode
    if mode == "linear":
        filter_mode = 0
    elif mode == "extended":
        filter_mode = 1
    else:
        filter_mode = 2

    cdef int estimation_method
    if method == "L-BFGS-B":
        estimation_method = 0
    elif method == "Nelder-Mead":
        estimation_method = 1
    else:
        raise ValueError(f"unsupported native method {method!r}")

    cdef sdsge_estimation_options est_opt
    est_opt.filter_mode = filter_mode
    est_opt.method = estimation_method
    est_opt.has_priors = has_priors
    est_opt.lo = &lo[0]
    est_opt.hi = &hi[0]
    est_opt.nbd = nbd_ptr
    est_opt.optim.m = m
    est_opt.optim.maxiter = maxiter
    est_opt.optim.maxfun = maxfun
    est_opt.optim.maxls = maxls
    est_opt.optim.factr = factr
    est_opt.optim.pgtol = pgtol
    est_opt.optim.fd_step = fd_step
    est_opt.optim.xatol = xatol
    est_opt.optim.fatol = fatol

    est_opt.compute_cov = compute_cov
    est_opt.cov_fd_step_scale = cov_fd_step_scale
    est_opt.cov_fd_absolute_floor = cov_fd_absolute_floor

    vcov = np.empty((n_theta, n_theta), dtype=np.float64)
    se = np.empty(n_theta, dtype=np.float64)
    cdef double[:, ::1] vcovv = vcov
    cdef double[::1] sev = se

    cdef sdsge_estimation_result res
    res.vcov = &vcovv[0, 0]
    res.se = &sev[0]

    cdef double lpr = 0.0

    with nogil:
        sdsge_run_estimation(ctxp, n_theta, &xv[0], &est_opt, &res)
        if has_priors:
            lpr = sdsge_logprior_at(b, &xv[0])
    params_out = np.array(nc.params, dtype=np.float64, copy=True)

    return {
        "x": x,
        "fun": res.base.fun,
        "nfev": int(res.base.nfev),
        "nit": int(res.base.nit),
        "success": bool(res.base.success),
        "status": int(res.base.status),
        "message": (
            (<bytes>res.base.message).decode() if res.base.message != NULL else ""
        ),
        "bk_violations": int(b.bk_violations),
        "params": params_out,
        "logprior": float(lpr),
        "vcov": vcov if compute_cov else None,
        "se": se if compute_cov else None,
        "cov_status": int(res.cov_status),
    }


def run_mcmc(
    object ctx_dto,
    str mode,
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
    unscented +logpost objective (issue #331). Marshals the mode's context DTO
    once, borrows ``rng``'s PCG64 state, and runs the whole chain in native
    ``nogil`` code, returning the kept draws (in theta space) plus the acceptance
    / BK counters. ``rng`` must be a numpy ``Generator``; it is held for the whole
    native run (the ``bitgen_t*`` is borrowed). Draws stay bit-exact numpy; the
    proposal (Cholesky) and covariance adaptation are native (statistical, not
    bit, equivalence with the numpy chain).

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

    cdef _NativeCtx nc = _build_native_ctx(ctx_dto, mode)
    cdef sdsge_obj_common *b = nc.b
    cdef void *ctxp = nc.ctxp
    cdef int64_t d = nc.n_theta
    if d <= 0:
        raise ValueError("No estimated parameters were provided.")

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

    # The chain and the MAP it starts from both walk theta, so both want the
    # change-of-variables density rather than the prior over the parameters.
    b.prior.include_logjac = 1

    # Borrow numpy's PCG64 state. `rng` is an argument, so its reference is held
    # for the whole call, including the nogil run below (the pointer is borrowed).
    cdef bitgen_t *bg = _bitgen_ptr(rng)

    cdef double[::1] th0v = np.ascontiguousarray(theta0, dtype=np.float64)
    cdef double[:, ::1] pcovv = proposal_cov

    cdef str map_method = map_options.get("method", "L-BFGS-B")
    cdef object map_bounds = map_options.get("bounds")
    cdef int64_t map_m = map_options.get("m", 10)
    cdef int64_t map_maxiter = map_options.get("maxiter", 15000)
    cdef int64_t map_maxfun = map_options.get("maxfun", 15000)
    cdef int64_t map_maxls = map_options.get("maxls", 20)
    cdef double map_factr = map_options.get("factr", 1e7)
    cdef double map_pgtol = map_options.get("pgtol", 1e-5)
    cdef double map_fd_step = map_options.get("fd_step", 0.0)
    cdef double map_xatol = map_options.get("xatol", 1e-4)
    cdef double map_fatol = map_options.get("fatol", 1e-4)
    cdef double[::1] map_lo = np.zeros(d, dtype=np.float64)
    cdef double[::1] map_hi = np.zeros(d, dtype=np.float64)
    cdef int64_t[::1] map_nbd = np.zeros(d, dtype=np.int64)
    cdef int has_map_bounds = map_bounds is not None
    cdef int64_t bi
    if has_map_bounds:
        for bi in range(d):
            lb, ub = map_bounds[bi]
            has_lo = lb is not None
            has_hi = ub is not None
            if has_lo:
                map_lo[bi] = lb
            if has_hi:
                map_hi[bi] = ub
            map_nbd[bi] = (2 if has_hi else 1) if has_lo else (3 if has_hi else 0)
    cdef const int64_t *map_nbd_ptr = &map_nbd[0] if has_map_bounds else NULL

    # Output buffers (Python-owned; native fills them).
    kept = np.empty((n_draws, d), dtype=np.float64)
    kept_lp = np.empty(n_draws, dtype=np.float64)
    kept_lj = np.empty(n_draws, dtype=np.float64)
    cdef double[:, ::1] keptv = kept
    cdef double[::1] keptlpv = kept_lp
    cdef double[::1] keptljv = kept_lj

    cdef int filter_mode
    cdef sdsge_objective_fn logpost
    if mode == "linear":
        filter_mode = 0
        logpost = sdsge_post_linear
    elif mode == "extended":
        filter_mode = 1
        logpost = sdsge_post_extended
    else:
        filter_mode = 2
        logpost = sdsge_post_unscented

    cdef int estimation_method
    if map_method == "L-BFGS-B":
        estimation_method = 0
    elif map_method == "Nelder-Mead":
        estimation_method = 1
    else:
        raise ValueError(f"unsupported native method {map_method!r}")

    cdef sdsge_estimation_options map_opt
    map_opt.filter_mode = filter_mode
    map_opt.method = estimation_method
    map_opt.has_priors = 1
    map_opt.lo = &map_lo[0]
    map_opt.hi = &map_hi[0]
    map_opt.nbd = map_nbd_ptr
    map_opt.optim.m = map_m
    map_opt.optim.maxiter = map_maxiter
    map_opt.optim.maxfun = map_maxfun
    map_opt.optim.maxls = map_maxls
    map_opt.optim.factr = map_factr
    map_opt.optim.pgtol = map_pgtol
    map_opt.optim.fd_step = map_fd_step
    map_opt.optim.xatol = map_xatol
    map_opt.optim.fatol = map_fatol

    cdef sdsge_mcmc_options opt
    opt.n_draws = n_draws
    opt.burn_in = burn_in
    opt.thin = thin
    opt.needs_map = compute_map
    opt.adapt = adapt
    opt.adapt_start = adapt_start
    opt.adapt_epsilon = adapt_epsilon
    opt.proposal_scale = proposal_scale
    opt.needs_hessian = needs_hessian
    opt.hessian_fd_step_scale = cov_fd_step_scale
    opt.hessian_fd_absolute_floor = cov_fd_absolute_floor

    cdef sdsge_mcmc_buffers buf
    buf.kept = &keptv[0, 0]
    buf.kept_lp = &keptlpv[0]
    buf.kept_lj = &keptljv[0]

    cdef sdsge_mcmc_result res
    b.bk_violations = 0
    with nogil:
        sdsge_mcmc_run(logpost, ctxp, bg, &th0v[0], d, &pcovv[0, 0], &opt, &map_opt,
                       &buf, &res)

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


cdef int _filter_mode_code(str mode):
    if mode == "linear":
        return 0
    if mode == "extended":
        return 1
    return 2

# Point objectives at an arbitrary theta. Each call marshals its own context
# from the DTO and drops it on return, so nothing is shared between calls: the
# scratch arenas, `include_logjac` and the BK counter are all call-local, and
# concurrent callers cannot reach each other's state. The cost is one full
# marshal per evaluation, which is the trade for having no lifetime to manage.


def loglik(object ctx_dto, str mode, theta not None):
    """Log-likelihood at ``theta`` (the unconstrained vector). The prior is not
    evaluated, so this is the same quantity the MLE objective maximizes."""
    cdef _NativeCtx nc = _build_native_ctx(ctx_dto, mode)
    cdef double[::1] thetav = np.ascontiguousarray(theta, dtype=np.float64)

    if thetav.shape[0] != nc.n_theta:
        raise ValueError(
            "theta length does not match the estimated parameter count."
        )
    cdef void *ctxp = nc.ctxp
    cdef sdsge_objective_fn fn = sdsge_select_objective(
        0, 0, _filter_mode_code(mode)
    )
    cdef double out
    nc.b.bk_violations = 0
    with nogil:
        out = fn(&thetav[0], ctxp)
    return np.float64(out)


def logprior(object ctx_dto, theta not None, bint jacobian=False):
    """The packed log-prior at ``theta``, read off an ``sdsge_prior_tables``
    mirror rather than eleven loose arrays. ``tables.has_prior`` is not consulted
    here: a disabled table carries zero-length columns, so the kernel sums
    nothing and returns 0.0."""

    cdef _NativeCtx nc = _build_native_ctx(ctx_dto, "linear")  # mode is irrelevant
    cdef double[::1] thetav = np.ascontiguousarray(theta, dtype=np.float64)

    if thetav.shape[0] != nc.n_theta:
        raise ValueError(
            "theta length does not match the estimated parameter count."
        )

    cdef double out
    nc.b.bk_violations = 0
    nc.b.prior.include_logjac = jacobian

    with nogil:
        sdsge_logprior_at(nc.b, &thetav[0])
    return np.float64(out)


def logpost(object ctx_dto, str mode, theta not None, bint jacobian=False):
    """Log-posterior at ``theta``. ``include_logjac`` picks the density: with
    it, the density over theta the sampler walks; without, the prior over the
    parameters read at ``theta``. Equals the log-likelihood when the run carries
    no prior."""
    cdef _NativeCtx nc = _build_native_ctx(ctx_dto, mode)
    cdef double[::1] thetav = np.ascontiguousarray(theta, dtype=np.float64)

    if thetav.shape[0] != nc.n_theta:
        raise ValueError(
            "theta length does not match the estimated parameter count."
        )
    cdef void *ctxp = nc.ctxp
    cdef sdsge_objective_fn fn = sdsge_select_objective(
        0, nc.has_prior, _filter_mode_code(mode)
    )
    cdef double out
    nc.b.bk_violations = 0
    nc.b.prior.include_logjac = jacobian
    with nogil:
        out = fn(&thetav[0], ctxp)
    return np.float64(out)
