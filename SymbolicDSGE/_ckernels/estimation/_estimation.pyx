# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython composer for the native estimation objective.

Marshals Python-side prep (packed arrays + cfunc/LAPACK addresses) into the C
context struct and calls the native objective. Python never assembles the struct
by hand.
"""

from libc.stdint cimport int64_t

from cpython.pycapsule cimport PyCapsule_GetName, PyCapsule_GetPointer
from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
import scipy.linalg.cython_lapack as _cython_lapack


cdef extern from "../_common/sdsge_complex.h":
    ctypedef struct c128:
        double re
        double im


cdef extern from "estimation.h":
    ctypedef void (*sdsge_residual_fn)()
    ctypedef void (*bc_residual_fn)()
    ctypedef void (*klein_zgges_fn)()
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

    ctypedef struct sdsge_solve1:
        double *ss
        double *a_real
        double *b_real
        c128 *s
        c128 *t
        c128 *z
        c128 *f
        c128 *p
        c128 *eig
        int64_t stab
        double *A
        double *B

    ctypedef struct sdsge_obj_common:
        sdsge_dims dims
        sdsge_residual_fn residual
        bc_residual_fn bc_residual
        klein_zgges_fn zgges
        meas_fn meas
        meas_fn jac
        const double *ss_seed
        int log_linear
        const double *y
        const double *P0
        const double *x0
        double jitter
        int symmetrize
        sdsge_param_map pmap
        sdsge_cov_spec q_spec
        sdsge_cov_spec r_spec
        sdsge_prior_tables prior
        double *params
        double *Q
        double *R
        double *corr_q
        double *corr_r
        double *std_q
        double *std_r
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
        double *hx_real
        double *gx_real
        double *bx
        double *eta
        double *gxx
        double *hxx
        double *gss
        double *hss
        double *steady_state

    ctypedef struct sdsge_unscented_ctx:
        sdsge_obj_common base
        sdsge_solve1 solve
        sdsge_solve2 solve2
        double *z0
        double alpha
        double beta
        double kappa

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


cdef extern from "optim.h":
    ctypedef double (*sdsge_objective_fn)(const double *x, void *ctx) noexcept nogil

    ctypedef struct sdsge_lbfgsb_options:
        int64_t m
        int64_t maxiter
        int64_t maxfun
        int64_t maxls
        double factr
        double pgtol
        double fd_step

    ctypedef struct sdsge_lbfgsb_result:
        int64_t status
        int64_t nfev
        int64_t nit
        double fun
        int success
        const char *message

    int64_t sdsge_lbfgsb(sdsge_objective_fn obj, void *obj_ctx, int64_t n,
                         double *x, const double *lo, const double *hi,
                         const int64_t *nbd, const sdsge_lbfgsb_options *opt,
                         sdsge_lbfgsb_result *out) nogil


cdef extern from "nelder_mead.h":
    ctypedef struct sdsge_neldermead_options:
        int64_t maxiter
        int64_t maxfun
        double xatol
        double fatol

    ctypedef struct sdsge_neldermead_result:
        int64_t status
        int64_t nfev
        int64_t nit
        double fun
        int success
        const char *message

    int64_t sdsge_neldermead(sdsge_objective_fn obj, void *obj_ctx, int64_t n,
                             double *x, const double *lo, const double *hi,
                             const int64_t *nbd,
                             const sdsge_neldermead_options *opt,
                             sdsge_neldermead_result *out) nogil


cdef object _zgges_capsule = _cython_lapack.__pyx_capi__["zgges"]
cdef klein_zgges_fn _zgges = <klein_zgges_fn>PyCapsule_GetPointer(
    _zgges_capsule, PyCapsule_GetName(_zgges_capsule)
)


def obj_linear_base(
    size_t residual_addr,
    size_t meas_addr,
    size_t jac_addr,
    int n_state,
    int n_exog,
    int n_obs,
    int log_linear,
    double[::1] ss_seed,        # n_var (Newton seed for the steady state)
    double[::1] base_calib,     # n_par
    double[:, ::1] Q,           # n_exog*n_exog constant
    double[:, ::1] R,           # n_obs*n_obs constant
    double[:, ::1] y,           # T*n_obs
    double[:, ::1] P0,          # n_var*n_var
    double jitter,
    int symmetrize,
):
    """Evaluate the native linear objective at base calibration (n_theta == 0,
    constant Q/R, no prior). Returns loglik. Composer for the first parity."""
    cdef int64_t n_var = ss_seed.shape[0]
    cdef int64_t n_par = base_calib.shape[0]
    cdef int64_t T = y.shape[0]
    cdef int64_t n_ctrl = n_var - n_state

    # Preallocated scratch (kept alive for the whole call).
    params = np.empty(n_par, dtype=np.float64)
    ss = np.empty(n_var, dtype=np.float64)
    a_real = np.empty((n_var, n_var), dtype=np.float64)
    b_real = np.empty((n_var, n_var), dtype=np.float64)
    s = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    t = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    z = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    f = np.empty((n_ctrl, n_state), dtype=np.complex128)
    p = np.empty((n_state, n_state), dtype=np.complex128)
    eig = np.empty(n_var, dtype=np.complex128)
    x0 = np.zeros(n_var, dtype=np.float64)
    A = np.empty((n_var, n_var), dtype=np.float64)
    B = np.empty((n_var, n_exog), dtype=np.float64)
    C = np.empty((n_obs, n_var), dtype=np.float64)
    d = np.empty(n_obs, dtype=np.float64)

    cdef double[::1] paramsv = params
    cdef double[::1] ssv = ss
    cdef double[:, ::1] arv = a_real
    cdef double[:, ::1] brv = b_real
    cdef double complex[::1, :] sv = s
    cdef double complex[::1, :] tv = t
    cdef double complex[::1, :] zv = z
    cdef double complex[:, ::1] fv = f
    cdef double complex[:, ::1] pv = p
    cdef double complex[::1] eigv = eig
    cdef double[::1] x0v = x0
    cdef double[:, ::1] Av = A
    cdef double[:, ::1] Bv = B
    cdef double[:, ::1] Cv = C
    cdef double[::1] dv = d

    cdef sdsge_linear_ctx ctx
    cdef sdsge_obj_common *b = &ctx.base

    b.dims.n_theta = 0
    b.dims.n_var = n_var
    b.dims.n_state = n_state
    b.dims.n_ctrl = n_ctrl
    b.dims.n_exog = n_exog
    b.dims.n_obs = n_obs
    b.dims.n_par = n_par
    b.dims.T = T

    b.residual = <sdsge_residual_fn><void*>residual_addr
    b.zgges = _zgges
    b.meas = <meas_fn><void*>meas_addr
    b.jac = <meas_fn><void*>jac_addr

    b.ss_seed = &ss_seed[0]
    b.log_linear = log_linear
    b.y = &y[0, 0]
    b.P0 = &P0[0, 0]
    b.x0 = &x0v[0]
    b.jitter = jitter
    b.symmetrize = symmetrize

    b.pmap.base_params = &base_calib[0]
    b.pmap.scalars = NULL
    b.pmap.n_scalars = 0

    b.q_spec.is_constant = 1
    b.q_spec.constant = &Q[0, 0]
    b.q_spec.K = n_exog
    b.q_spec.corr_from_block = 0
    b.q_spec.n_pairs = 0

    b.r_spec.is_constant = 1
    b.r_spec.constant = &R[0, 0]
    b.r_spec.K = n_obs
    b.r_spec.corr_from_block = 0
    b.r_spec.n_pairs = 0

    b.prior.has_prior = 0

    b.params = &paramsv[0]
    b.Q = NULL
    b.R = NULL
    b.corr_q = NULL
    b.corr_r = NULL
    b.std_q = NULL
    b.std_r = NULL
    b.bk_violations = 0

    ctx.solve.ss = &ssv[0]
    ctx.solve.a_real = &arv[0, 0]
    ctx.solve.b_real = &brv[0, 0]
    ctx.solve.s = <c128*>&sv[0, 0]
    ctx.solve.t = <c128*>&tv[0, 0]
    ctx.solve.z = <c128*>&zv[0, 0]
    ctx.solve.f = <c128*>&fv[0, 0]
    ctx.solve.p = <c128*>&pv[0, 0]
    ctx.solve.eig = <c128*>&eigv[0]
    ctx.solve.A = &Av[0, 0]
    ctx.solve.B = &Bv[0, 0]

    ctx.C = &Cv[0, 0]
    ctx.d = &dv[0]

    # One-time construction seed: fill the calibrated baseline (the per-eval fill
    # only touches estimated slots, of which there are none here).
    sdsge_init_params(&paramsv[0], &base_calib[0], n_par)

    cdef double ll
    with nogil:
        ll = sdsge_obj_linear(&ctx, NULL, 0)
    return ll, int(b.bk_violations)


def obj_extended_base(
    size_t residual_addr,
    size_t meas_addr,
    size_t jac_addr,
    int n_state,
    int n_exog,
    int n_obs,
    int log_linear,
    double[::1] ss_seed,        # n_var (Newton seed for the steady state)
    double[::1] base_calib,     # n_par
    double[:, ::1] Q,           # n_exog*n_exog constant
    double[:, ::1] R,           # n_obs*n_obs constant
    double[:, ::1] y,           # T*n_obs
    double[:, ::1] P0,          # n_var*n_var
    double jitter,
    int symmetrize,
):
    """Evaluate the native extended objective at base calibration (n_theta == 0,
    constant Q/R, no prior). Returns loglik. Same as ``obj_linear_base`` minus the
    (C, d) buffers: the EKF relinearizes the measurement each step via the meas /
    jac cfuncs at the running state estimate."""
    cdef int64_t n_var = ss_seed.shape[0]
    cdef int64_t n_par = base_calib.shape[0]
    cdef int64_t T = y.shape[0]
    cdef int64_t n_ctrl = n_var - n_state

    # Preallocated scratch (kept alive for the whole call).
    params = np.empty(n_par, dtype=np.float64)
    ss = np.empty(n_var, dtype=np.float64)
    a_real = np.empty((n_var, n_var), dtype=np.float64)
    b_real = np.empty((n_var, n_var), dtype=np.float64)
    s = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    t = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    z = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    f = np.empty((n_ctrl, n_state), dtype=np.complex128)
    p = np.empty((n_state, n_state), dtype=np.complex128)
    eig = np.empty(n_var, dtype=np.complex128)
    x0 = np.zeros(n_var, dtype=np.float64)
    A = np.empty((n_var, n_var), dtype=np.float64)
    B = np.empty((n_var, n_exog), dtype=np.float64)

    cdef double[::1] paramsv = params
    cdef double[::1] ssv = ss
    cdef double[:, ::1] arv = a_real
    cdef double[:, ::1] brv = b_real
    cdef double complex[::1, :] sv = s
    cdef double complex[::1, :] tv = t
    cdef double complex[::1, :] zv = z
    cdef double complex[:, ::1] fv = f
    cdef double complex[:, ::1] pv = p
    cdef double complex[::1] eigv = eig
    cdef double[::1] x0v = x0
    cdef double[:, ::1] Av = A
    cdef double[:, ::1] Bv = B

    cdef sdsge_extended_ctx ctx
    cdef sdsge_obj_common *b = &ctx.base

    b.dims.n_theta = 0
    b.dims.n_var = n_var
    b.dims.n_state = n_state
    b.dims.n_ctrl = n_ctrl
    b.dims.n_exog = n_exog
    b.dims.n_obs = n_obs
    b.dims.n_par = n_par
    b.dims.T = T

    b.residual = <sdsge_residual_fn><void*>residual_addr
    b.zgges = _zgges
    b.meas = <meas_fn><void*>meas_addr
    b.jac = <meas_fn><void*>jac_addr

    b.ss_seed = &ss_seed[0]
    b.log_linear = log_linear
    b.y = &y[0, 0]
    b.P0 = &P0[0, 0]
    b.x0 = &x0v[0]
    b.jitter = jitter
    b.symmetrize = symmetrize

    b.pmap.base_params = &base_calib[0]
    b.pmap.scalars = NULL
    b.pmap.n_scalars = 0

    b.q_spec.is_constant = 1
    b.q_spec.constant = &Q[0, 0]
    b.q_spec.K = n_exog
    b.q_spec.corr_from_block = 0
    b.q_spec.n_pairs = 0

    b.r_spec.is_constant = 1
    b.r_spec.constant = &R[0, 0]
    b.r_spec.K = n_obs
    b.r_spec.corr_from_block = 0
    b.r_spec.n_pairs = 0

    b.prior.has_prior = 0

    b.params = &paramsv[0]
    b.Q = NULL
    b.R = NULL
    b.corr_q = NULL
    b.corr_r = NULL
    b.std_q = NULL
    b.std_r = NULL
    b.bk_violations = 0

    ctx.solve.ss = &ssv[0]
    ctx.solve.a_real = &arv[0, 0]
    ctx.solve.b_real = &brv[0, 0]
    ctx.solve.s = <c128*>&sv[0, 0]
    ctx.solve.t = <c128*>&tv[0, 0]
    ctx.solve.z = <c128*>&zv[0, 0]
    ctx.solve.f = <c128*>&fv[0, 0]
    ctx.solve.p = <c128*>&pv[0, 0]
    ctx.solve.eig = <c128*>&eigv[0]
    ctx.solve.A = &Av[0, 0]
    ctx.solve.B = &Bv[0, 0]

    # One-time construction seed (see obj_linear_base).
    sdsge_init_params(&paramsv[0], &base_calib[0], n_par)

    cdef double ll
    with nogil:
        ll = sdsge_obj_extended(&ctx, NULL, 0)
    return ll, int(b.bk_violations)


def obj_unscented_base(
    size_t residual_addr,
    size_t bc_residual_addr,
    size_t meas_addr,
    int n_state,
    int n_exog,
    int n_obs,
    double[::1] ss_seed,        # n_var (Newton seed for the steady state)
    double[::1] base_calib,     # n_par
    double[:, ::1] Q,           # n_exog*n_exog constant
    double[:, ::1] R,           # n_obs*n_obs constant
    double[:, ::1] y,           # T*n_obs
    double[:, ::1] P0,          # 2*n_state x 2*n_state (UKF)
    double jitter,
    int symmetrize,
    double alpha=1.0,
    double beta=2.0,
    double kappa=1.0,
):
    """Evaluate the native unscented objective at base calibration (n_theta == 0,
    constant Q/R, no prior). Returns loglik. Order-2 is levels-only, so
    ``log_linear`` is forced to 0 and the solve1 pencil (a_real/b_real) is reused
    by the second-order kernels. Companion to ``obj_linear_base``."""
    cdef int64_t n_var = ss_seed.shape[0]
    cdef int64_t n_par = base_calib.shape[0]
    cdef int64_t T = y.shape[0]
    cdef int64_t n_ctrl = n_var - n_state
    cdef int64_t n2 = 2 * n_var

    # Preallocated scratch (kept alive for the whole call).
    params = np.empty(n_par, dtype=np.float64)
    ss = np.empty(n_var, dtype=np.float64)
    a_real = np.empty((n_var, n_var), dtype=np.float64)
    b_real = np.empty((n_var, n_var), dtype=np.float64)
    s = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    t = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    z = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    f = np.empty((n_ctrl, n_state), dtype=np.complex128)
    p = np.empty((n_state, n_state), dtype=np.complex128)
    eig = np.empty(n_var, dtype=np.complex128)
    x0 = np.zeros(n_var, dtype=np.float64)
    A = np.empty((n_var, n_var), dtype=np.float64)
    B = np.empty((n_var, n_exog), dtype=np.float64)

    # Second-order scratch.
    f_xx = np.empty((n_var, n2, n2), dtype=np.float64)
    hx_real = np.empty((n_state, n_state), dtype=np.float64)
    gx_real = np.empty((n_ctrl, n_state), dtype=np.float64)
    bx = np.empty((n_state, n_exog), dtype=np.float64)
    gxx = np.empty((n_ctrl, n_state, n_state), dtype=np.float64)
    hxx = np.empty((n_state, n_state, n_state), dtype=np.float64)
    gss = np.empty(n_ctrl, dtype=np.float64)
    hss = np.empty(n_state, dtype=np.float64)
    steady2 = np.empty(n_var, dtype=np.float64)

    # eta (n_state x n_exog), padding rows zeroed once. Q is constant here, so the
    # objective's runtime guard skips its own Cholesky and reads this precomputed
    # factor; matches DSGESolver._build_eta (np.linalg.cholesky, no jitter).
    eta = np.zeros((n_state, n_exog), dtype=np.float64)
    if n_exog > 0:
        eta[:n_exog, :] = np.linalg.cholesky(np.asarray(Q))

    # z0 = [x0[:n_state]; 0] (2*n_state); x0 is zero at base.
    z0 = np.zeros(2 * n_state, dtype=np.float64)
    z0[:n_state] = x0[:n_state]

    cdef double[::1] paramsv = params
    cdef double[::1] ssv = ss
    cdef double[:, ::1] arv = a_real
    cdef double[:, ::1] brv = b_real
    cdef double complex[::1, :] sv = s
    cdef double complex[::1, :] tv = t
    cdef double complex[::1, :] zv = z
    cdef double complex[:, ::1] fv = f
    cdef double complex[:, ::1] pv = p
    cdef double complex[::1] eigv = eig
    cdef double[::1] x0v = x0
    cdef double[:, ::1] Av = A
    cdef double[:, ::1] Bv = B

    cdef double[:, :, ::1] fxxv = f_xx
    cdef double[:, ::1] hxrv = hx_real
    cdef double[:, ::1] gxrv = gx_real
    cdef double[:, ::1] bxv = bx
    cdef double[:, ::1] etav = eta
    cdef double[:, :, ::1] gxxv = gxx
    cdef double[:, :, ::1] hxxv = hxx
    cdef double[::1] gssv = gss
    cdef double[::1] hssv = hss
    cdef double[::1] steady2v = steady2
    cdef double[::1] z0v = z0

    cdef sdsge_unscented_ctx ctx
    cdef sdsge_obj_common *b = &ctx.base

    b.dims.n_theta = 0
    b.dims.n_var = n_var
    b.dims.n_state = n_state
    b.dims.n_ctrl = n_ctrl
    b.dims.n_exog = n_exog
    b.dims.n_obs = n_obs
    b.dims.n_par = n_par
    b.dims.T = T

    b.residual = <sdsge_residual_fn><void*>residual_addr
    b.bc_residual = <bc_residual_fn><void*>bc_residual_addr
    b.zgges = _zgges
    b.meas = <meas_fn><void*>meas_addr
    b.jac = NULL

    b.ss_seed = &ss_seed[0]
    b.log_linear = 0
    b.y = &y[0, 0]
    b.P0 = &P0[0, 0]
    b.x0 = &x0v[0]
    b.jitter = jitter
    b.symmetrize = symmetrize

    b.pmap.base_params = &base_calib[0]
    b.pmap.scalars = NULL
    b.pmap.n_scalars = 0

    b.q_spec.is_constant = 1
    b.q_spec.constant = &Q[0, 0]
    b.q_spec.K = n_exog
    b.q_spec.corr_from_block = 0
    b.q_spec.n_pairs = 0

    b.r_spec.is_constant = 1
    b.r_spec.constant = &R[0, 0]
    b.r_spec.K = n_obs
    b.r_spec.corr_from_block = 0
    b.r_spec.n_pairs = 0

    b.prior.has_prior = 0

    b.params = &paramsv[0]
    b.Q = NULL
    b.R = NULL
    b.corr_q = NULL
    b.corr_r = NULL
    b.std_q = NULL
    b.std_r = NULL
    b.bk_violations = 0

    ctx.solve.ss = &ssv[0]
    ctx.solve.a_real = &arv[0, 0]
    ctx.solve.b_real = &brv[0, 0]
    ctx.solve.s = <c128*>&sv[0, 0]
    ctx.solve.t = <c128*>&tv[0, 0]
    ctx.solve.z = <c128*>&zv[0, 0]
    ctx.solve.f = <c128*>&fv[0, 0]
    ctx.solve.p = <c128*>&pv[0, 0]
    ctx.solve.eig = <c128*>&eigv[0]
    ctx.solve.A = &Av[0, 0]
    ctx.solve.B = &Bv[0, 0]

    ctx.solve2.f_xx = &fxxv[0, 0, 0]
    ctx.solve2.hx_real = &hxrv[0, 0]
    ctx.solve2.gx_real = &gxrv[0, 0]
    ctx.solve2.bx = &bxv[0, 0]
    ctx.solve2.eta = &etav[0, 0]
    ctx.solve2.gxx = &gxxv[0, 0, 0]
    ctx.solve2.hxx = &hxxv[0, 0, 0]
    ctx.solve2.gss = &gssv[0]
    ctx.solve2.hss = &hssv[0]
    ctx.solve2.steady_state = &steady2v[0]

    ctx.z0 = &z0v[0]
    ctx.alpha = alpha
    ctx.beta = beta
    ctx.kappa = kappa

    # One-time construction seed (see obj_linear_base).
    sdsge_init_params(&paramsv[0], &base_calib[0], n_par)

    cdef double ll
    with nogil:
        ll = sdsge_obj_unscented(&ctx, NULL, 0)
    return ll, int(b.bk_violations)


# --- Native MLE/MAP driver over the per-mode objective (issue #330) ----------
#
# Trampolines adapt each mode's objective (ctx, theta, has_priors) ABI to the
# optimizer's (x, void*ctx) ABI, negating loglik/logpost for minimization. The
# BK/NaN sentinel (-inf loglik) maps to +inf, which the drivers tolerate.

cdef double _obj_lin_ll(const double *x, void *ctx) noexcept nogil:
    return -sdsge_obj_linear(<sdsge_linear_ctx*>ctx, x, 0)
cdef double _obj_lin_lp(const double *x, void *ctx) noexcept nogil:
    return -sdsge_obj_linear(<sdsge_linear_ctx*>ctx, x, 1)
cdef double _obj_ext_ll(const double *x, void *ctx) noexcept nogil:
    return -sdsge_obj_extended(<sdsge_extended_ctx*>ctx, x, 0)
cdef double _obj_ext_lp(const double *x, void *ctx) noexcept nogil:
    return -sdsge_obj_extended(<sdsge_extended_ctx*>ctx, x, 1)
cdef double _obj_unsc_ll(const double *x, void *ctx) noexcept nogil:
    return -sdsge_obj_unscented(<sdsge_unscented_ctx*>ctx, x, 0)
cdef double _obj_unsc_lp(const double *x, void *ctx) noexcept nogil:
    return -sdsge_obj_unscented(<sdsge_unscented_ctx*>ctx, x, 1)


def run_estimation(
    object ctx_dto,
    str mode,
    str method,
    double[::1] theta0,
    bounds=None,
    int has_priors=0,
    int m=10,
    int maxiter=15000,
    int maxfun=15000,
    int maxls=20,
    double factr=1e7,
    double pgtol=1e-5,
    double fd_step=0.0,
    double xatol=1e-4,
    double fatol=1e-4,
):
    """Native MLE/MAP over the linear / extended / unscented objective. Marshal
    the mode's context DTO into its C ctx, then minimize ``-loglik``
    (``has_priors=0``) or ``-logpost`` (``has_priors=1``) with the native
    L-BFGS-B / Nelder-Mead driver. Returns the driver result plus ``params`` (the
    named parameter vector scattered at x_best) and ``logprior`` (MAP), all
    resolved natively with no filter re-eval. The base marshaling is shared; only
    the ctx struct, objective, and mode scratch differ per ``mode``."""
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
    cdef int64_t n2 = 2 * n_var

    # Working theta (the driver mutates it in place).
    x = np.array(theta0, dtype=np.float64, copy=True)
    cdef double[::1] xv = x

    # Pinned inputs. Python guarantees dtype; C-contiguity is enforced here.
    cdef double[::1] ssv = np.ascontiguousarray(base.ss_seed, dtype=np.float64)
    cdef double[:, ::1] yv = np.ascontiguousarray(base.y, dtype=np.float64)
    cdef double[:, ::1] P0v = np.ascontiguousarray(base.P0, dtype=np.float64)
    cdef double[::1] bpv = np.ascontiguousarray(
        base.pmap.base_params, dtype=np.float64
    )
    # The linear kf dereferences x0 unconditionally (no NULL guard), so a
    # missing x0 materializes as the zero initial state, matching obj_linear_base.
    cdef double[::1] x0v
    if base.x0 is not None:
        x0v = np.ascontiguousarray(base.x0, dtype=np.float64)
    else:
        x0v = np.zeros(n_var, dtype=np.float64)

    # Scratch the objective writes (allocated from dims, kept alive here).
    params = np.empty(n_par, dtype=np.float64)
    cdef double[::1] paramsv = params
    ss = np.empty(n_var, dtype=np.float64)
    cdef double[::1] ssv2 = ss
    a_real = np.empty((n_var, n_var), dtype=np.float64)
    b_real = np.empty((n_var, n_var), dtype=np.float64)
    cdef double[:, ::1] arv = a_real
    cdef double[:, ::1] brv = b_real
    s = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    t = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    z = np.empty((n_var, n_var), dtype=np.complex128, order="F")
    cdef double complex[::1, :] sv = s
    cdef double complex[::1, :] tv = t
    cdef double complex[::1, :] zv = z
    f = np.empty((n_ctrl, n_state), dtype=np.complex128)
    p = np.empty((n_state, n_state), dtype=np.complex128)
    eig = np.empty(n_var, dtype=np.complex128)
    cdef double complex[:, ::1] fv = f
    cdef double complex[:, ::1] pv = p
    cdef double complex[::1] eigv = eig
    A = np.empty((n_var, n_var), dtype=np.float64)
    B = np.empty((n_var, n_exog), dtype=np.float64)
    C = np.empty((n_obs, n_var), dtype=np.float64)
    d = np.empty(n_obs, dtype=np.float64)
    cdef double[:, ::1] Av = A
    cdef double[:, ::1] Bv = B
    cdef double[:, ::1] Cv = C  # linear measurement scratch (wired only for linear)
    cdef double[::1] dv = d

    # Unscented-only second-order scratch (allocated in that branch).
    cdef double[:, :, ::1] fxxv
    cdef double[:, ::1] hxrv
    cdef double[:, ::1] gxrv
    cdef double[:, ::1] bxv
    cdef double[:, :, ::1] gxxv
    cdef double[:, :, ::1] hxxv
    cdef double[::1] gssv
    cdef double[::1] hssv
    cdef double[::1] st2v
    cdef double[:, ::1] etav
    cdef double[::1] z0v
    Q = np.empty((n_exog, n_exog), dtype=np.float64)
    R = np.empty((n_obs, n_obs), dtype=np.float64)
    corr_q = np.empty((n_exog, n_exog), dtype=np.float64)
    corr_r = np.empty((n_obs, n_obs), dtype=np.float64)
    std_q = np.empty(n_exog, dtype=np.float64)
    std_r = np.empty(n_obs, dtype=np.float64)
    cdef double[:, ::1] Qv = Q
    cdef double[:, ::1] Rv = R
    cdef double[:, ::1] cqv = corr_q
    cdef double[:, ::1] crv = corr_r
    cdef double[::1] sqv = std_q
    cdef double[::1] srv = std_r

    # Scatter array-of-structs (malloc'd; freed after the driver returns).
    scalars_list = base.pmap.scalars
    cdef int64_t n_scalars = len(scalars_list)
    cdef sdsge_scalar_scatter *scalars_c = <sdsge_scalar_scatter *>PyMem_Malloc(
        (n_scalars if n_scalars > 0 else 1) * sizeof(sdsge_scalar_scatter)
    )
    if scalars_c == NULL:
        raise MemoryError()
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
    cdef int64_t p_nscalar = 0
    cdef int64_t p_nblocks = 0

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

    # Mode dispatch: pick the ctx, its base/solve1 pointers, the void* the driver
    # sees, and the objective trampoline. Only one ctx is used per call.
    cdef sdsge_linear_ctx lctx
    cdef sdsge_extended_ctx ectx
    cdef sdsge_unscented_ctx uctx
    cdef sdsge_obj_common *b
    cdef sdsge_solve1 *s1
    cdef void *ctxp
    cdef sdsge_objective_fn obj
    if mode == "linear":
        b = &lctx.base
        s1 = &lctx.solve
        ctxp = <void*>&lctx
        obj = _obj_lin_lp if has_priors else _obj_lin_ll
    elif mode == "extended":
        b = &ectx.base
        s1 = &ectx.solve
        ctxp = <void*>&ectx
        obj = _obj_ext_lp if has_priors else _obj_ext_ll
    elif mode == "unscented":
        b = &uctx.base
        s1 = &uctx.solve
        ctxp = <void*>&uctx
        obj = _obj_unsc_lp if has_priors else _obj_unsc_ll
    else:
        PyMem_Free(scalars_c)
        raise ValueError(f"unsupported native filter mode {mode!r}")

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
    b.meas = <meas_fn><void*><size_t>base.meas_addr
    b.jac = <meas_fn><void*><size_t>base.jac_addr

    b.ss_seed = &ssv[0]
    b.log_linear = int(base.log_linear)
    b.y = &yv[0, 0]
    b.P0 = &P0v[0, 0]
    b.x0 = &x0v[0]
    b.jitter = base.jitter
    b.symmetrize = int(base.symmetrize)

    b.pmap.base_params = &bpv[0]
    b.pmap.scalars = scalars_c if n_scalars > 0 else NULL
    b.pmap.n_scalars = n_scalars

    b.q_spec.is_constant = int(qs.is_constant)
    b.q_spec.K = <int64_t>qs.K
    b.q_spec.corr_from_block = int(qs.corr_from_block)
    b.q_spec.block_theta_off = <int64_t>qs.block_theta_off
    b.q_spec.block_theta_len = <int64_t>qs.block_theta_len
    if qs.is_constant:
        q_const_v = np.ascontiguousarray(qs.constant, dtype=np.float64)
        b.q_spec.constant = &q_const_v[0, 0]
        b.q_spec.std_slots = NULL
        b.q_spec.pair_i = NULL
        b.q_spec.pair_j = NULL
        b.q_spec.pair_slot = NULL
        b.q_spec.n_pairs = 0
    else:
        b.q_spec.constant = NULL
        q_std_v = np.ascontiguousarray(qs.std_slots, dtype=np.int64)
        b.q_spec.std_slots = &q_std_v[0]
        q_pi_v = np.ascontiguousarray(qs.pair_i, dtype=np.int64)
        q_pj_v = np.ascontiguousarray(qs.pair_j, dtype=np.int64)
        q_ps_v = np.ascontiguousarray(qs.pair_slot, dtype=np.int64)
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
        r_const_v = np.ascontiguousarray(rs.constant, dtype=np.float64)
        b.r_spec.constant = &r_const_v[0, 0]
        b.r_spec.std_slots = NULL
        b.r_spec.pair_i = NULL
        b.r_spec.pair_j = NULL
        b.r_spec.pair_slot = NULL
        b.r_spec.n_pairs = 0
    else:
        b.r_spec.constant = NULL
        r_std_v = np.ascontiguousarray(rs.std_slots, dtype=np.int64)
        b.r_spec.std_slots = &r_std_v[0]
        r_pi_v = np.ascontiguousarray(rs.pair_i, dtype=np.int64)
        r_pj_v = np.ascontiguousarray(rs.pair_j, dtype=np.int64)
        r_ps_v = np.ascontiguousarray(rs.pair_slot, dtype=np.int64)
        r_np = r_pi_v.shape[0]
        b.r_spec.n_pairs = r_np
        b.r_spec.pair_i = &r_pi_v[0] if r_np > 0 else NULL
        b.r_spec.pair_j = &r_pj_v[0] if r_np > 0 else NULL
        b.r_spec.pair_slot = &r_ps_v[0] if r_np > 0 else NULL

    b.prior.has_prior = has_prior
    if has_prior:
        p_si_v = np.ascontiguousarray(pr.scalar_indices, dtype=np.int64)
        p_sdc_v = np.ascontiguousarray(pr.scalar_dist_codes, dtype=np.int64)
        p_stc_v = np.ascontiguousarray(pr.scalar_transform_codes, dtype=np.int64)
        p_sdp_v = np.ascontiguousarray(pr.scalar_dist_params, dtype=np.float64)
        p_stp_v = np.ascontiguousarray(pr.scalar_transform_params, dtype=np.float64)
        p_mo_v = np.ascontiguousarray(pr.matrix_offsets, dtype=np.int64)
        p_md_v = np.ascontiguousarray(pr.matrix_dims, dtype=np.int64)
        p_ml_v = np.ascontiguousarray(pr.matrix_lengths, dtype=np.int64)
        p_me_v = np.ascontiguousarray(pr.matrix_etas, dtype=np.float64)
        p_mlc_v = np.ascontiguousarray(pr.matrix_log_constants, dtype=np.float64)
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
    b.R = &Rv[0, 0]
    b.corr_q = &cqv[0, 0]
    b.corr_r = &crv[0, 0]
    b.std_q = &sqv[0]
    b.std_r = &srv[0]
    b.bk_violations = 0

    s1.ss = &ssv2[0]
    s1.a_real = &arv[0, 0]
    s1.b_real = &brv[0, 0]
    s1.s = <c128*>&sv[0, 0]
    s1.t = <c128*>&tv[0, 0]
    s1.z = <c128*>&zv[0, 0]
    s1.f = <c128*>&fv[0, 0]
    s1.p = <c128*>&pv[0, 0]
    s1.eig = <c128*>&eigv[0]
    s1.A = &Av[0, 0]
    s1.B = &Bv[0, 0]

    # Mode-specific ctx wiring.
    if mode == "linear":
        lctx.C = &Cv[0, 0]
        lctx.d = &dv[0]
    elif mode == "unscented":
        f_xx = np.empty((n_var, n2, n2), dtype=np.float64)
        hx_real = np.empty((n_state, n_state), dtype=np.float64)
        gx_real = np.empty((n_ctrl, n_state), dtype=np.float64)
        bx = np.empty((n_state, n_exog), dtype=np.float64)
        gxx = np.empty((n_ctrl, n_state, n_state), dtype=np.float64)
        hxx = np.empty((n_state, n_state, n_state), dtype=np.float64)
        gss = np.empty(n_ctrl, dtype=np.float64)
        hss = np.empty(n_state, dtype=np.float64)
        steady2 = np.empty(n_var, dtype=np.float64)
        # eta (n_state x n_exog): the objective recomputes chol(Q) per eval when Q
        # varies; for constant Q it reads this precomputed factor.
        eta = np.zeros((n_state, n_exog), dtype=np.float64)
        if qs.is_constant and n_exog > 0:
            eta[:n_exog, :] = np.linalg.cholesky(
                np.asarray(qs.constant, dtype=np.float64).reshape(n_exog, n_exog)
            )
        z0 = np.ascontiguousarray(ctx_dto.z0, dtype=np.float64)
        fxxv = f_xx
        hxrv = hx_real
        gxrv = gx_real
        bxv = bx
        gxxv = gxx
        hxxv = hxx
        gssv = gss
        hssv = hss
        st2v = steady2
        etav = eta
        z0v = z0
        uctx.solve2.f_xx = &fxxv[0, 0, 0]
        uctx.solve2.hx_real = &hxrv[0, 0]
        uctx.solve2.gx_real = &gxrv[0, 0]
        uctx.solve2.bx = &bxv[0, 0]
        uctx.solve2.eta = &etav[0, 0]
        uctx.solve2.gxx = &gxxv[0, 0, 0]
        uctx.solve2.hxx = &hxxv[0, 0, 0]
        uctx.solve2.gss = &gssv[0]
        uctx.solve2.hss = &hssv[0]
        uctx.solve2.steady_state = &st2v[0]
        uctx.z0 = &z0v[0]
        uctx.alpha = ctx_dto.alpha
        uctx.beta = ctx_dto.beta
        uctx.kappa = ctx_dto.kappa

    # Seed the calibrated baseline once; the per-eval fill touches only the
    # estimated slots.
    sdsge_init_params(&paramsv[0], &bpv[0], n_par)

    cdef sdsge_lbfgsb_options lopt
    cdef sdsge_lbfgsb_result lres
    cdef sdsge_neldermead_options nopt
    cdef sdsge_neldermead_result nres
    cdef int64_t status = 0
    cdef int64_t nfev = 0
    cdef int64_t nit = 0
    cdef double fun = 0.0
    cdef int success = 0
    cdef bytes message = b""
    cdef double lpr = 0.0
    params_out = params

    try:
        if method == "L-BFGS-B":
            lopt.m = m
            lopt.maxiter = maxiter
            lopt.maxfun = maxfun
            lopt.maxls = maxls
            lopt.factr = factr
            lopt.pgtol = pgtol
            lopt.fd_step = fd_step
            with nogil:
                sdsge_lbfgsb(obj, ctxp, n_theta, &xv[0], &lo[0], &hi[0],
                             nbd_ptr, &lopt, &lres)
            status = lres.status
            nfev = lres.nfev
            nit = lres.nit
            fun = lres.fun
            success = lres.success
            message = (<bytes>lres.message) if lres.message != NULL else b""
        elif method == "Nelder-Mead":
            nopt.maxiter = maxiter
            nopt.maxfun = maxfun
            nopt.xatol = xatol
            nopt.fatol = fatol
            with nogil:
                sdsge_neldermead(obj, ctxp, n_theta, &xv[0], &lo[0],
                                 &hi[0], nbd_ptr, &nopt, &nres)
            status = nres.status
            nfev = nres.nfev
            nit = nres.nit
            fun = nres.fun
            success = nres.success
            message = (<bytes>nres.message) if nres.message != NULL else b""
        else:
            raise ValueError(f"unsupported native method {method!r}")
        # Resolve the named params (scatter x_best -> params) and, for MAP, the
        # log-prior at x_best. Scatter / prior only, no filter. scalars_c must
        # still be alive here (the scatter reads it), so this runs before free.
        with nogil:
            sdsge_scatter_params(b, &xv[0])
            if has_priors:
                lpr = sdsge_logprior_at(b, &xv[0])
        params_out = np.array(params, dtype=np.float64, copy=True)
    finally:
        PyMem_Free(scalars_c)

    return {
        "x": x,
        "fun": fun,
        "nfev": int(nfev),
        "nit": int(nit),
        "success": bool(success),
        "status": int(status),
        "message": message.decode(),
        "bk_violations": int(b.bk_violations),
        "params": params_out,
        "logprior": float(lpr),
    }
