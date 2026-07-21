# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython composer for the native estimation objective.

Marshals Python-side prep (packed arrays + cfunc/LAPACK addresses) into the C
context struct and calls the native objective. Python never assembles the struct
by hand.
"""

from libc.stdint cimport int64_t

from cpython.pycapsule cimport PyCapsule_GetName, PyCapsule_GetPointer

import numpy as np
import scipy.linalg.cython_lapack as _cython_lapack


cdef extern from "../_common/sdsge_complex.h":
    ctypedef struct c128:
        double re
        double im


cdef extern from "estimation.h":
    ctypedef void (*sdsge_residual_fn)()
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
        int64_t n_params
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
        const int64_t *calib_gather
        const int64_t *calib_upd
        int64_t n_calib_upd

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

    ctypedef struct sdsge_solve1:
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
        klein_zgges_fn zgges
        meas_fn meas
        meas_fn jac
        const double *steady_state
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
        double *calib_vec
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
        const double *zero_state
        double *C
        double *d

    void sdsge_init_params(double *params, const double *base_params,
                           int64_t n_params) nogil
    void sdsge_init_calib(double *calib_vec, const double *params,
                          const int64_t *calib_gather, int64_t n_par) nogil
    double sdsge_obj_linear(sdsge_linear_ctx *ctx, const double *theta,
                            int has_priors) nogil


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
    double[::1] steady_state,   # n_var
    double[::1] base_calib,     # n_par (== n_params here)
    double[:, ::1] Q,           # n_exog*n_exog constant
    double[:, ::1] R,           # n_obs*n_obs constant
    double[:, ::1] y,           # T*n_obs
    double[:, ::1] P0,          # n_var*n_var
    double[::1] zero_state,     # n_var
    double jitter,
    int symmetrize,
):
    """Evaluate the native linear objective at base calibration (n_theta == 0,
    constant Q/R, no prior). Returns loglik. Composer for the first parity."""
    cdef int64_t n_var = steady_state.shape[0]
    cdef int64_t n_par = base_calib.shape[0]
    cdef int64_t T = y.shape[0]
    cdef int64_t n_ctrl = n_var - n_state

    # Preallocated scratch (kept alive for the whole call).
    params = np.empty(n_par, dtype=np.float64)
    calib_vec = np.empty(n_par, dtype=np.float64)
    calib_gather = np.arange(n_par, dtype=np.int64)
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
    cdef double[::1] calibv = calib_vec
    cdef int64_t[::1] cgv = calib_gather
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
    b.dims.n_params = n_par
    b.dims.T = T

    b.residual = <sdsge_residual_fn><void*>residual_addr
    b.zgges = _zgges
    b.meas = <meas_fn><void*>meas_addr
    b.jac = <meas_fn><void*>jac_addr

    b.steady_state = &steady_state[0]
    b.log_linear = log_linear
    b.y = &y[0, 0]
    b.P0 = &P0[0, 0]
    b.x0 = &x0v[0]
    b.jitter = jitter
    b.symmetrize = symmetrize

    b.pmap.base_params = &base_calib[0]
    b.pmap.scalars = NULL
    b.pmap.n_scalars = 0
    b.pmap.calib_gather = &cgv[0]
    b.pmap.calib_upd = NULL
    b.pmap.n_calib_upd = 0

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
    b.calib_vec = &calibv[0]
    b.Q = NULL
    b.R = NULL
    b.corr_q = NULL
    b.corr_r = NULL
    b.std_q = NULL
    b.std_r = NULL
    b.bk_violations = 0

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

    ctx.zero_state = &zero_state[0]
    ctx.C = &Cv[0, 0]
    ctx.d = &dv[0]

    # One-time construction seeds: fill the calibrated baseline and its calib
    # image (the per-eval fill only touches estimated slots, of which there are
    # none here).
    sdsge_init_params(&paramsv[0], &base_calib[0], n_par)
    sdsge_init_calib(&calibv[0], &paramsv[0], &cgv[0], n_par)

    cdef double ll
    with nogil:
        ll = sdsge_obj_linear(&ctx, NULL, 0)
    return ll, int(b.bk_violations)
