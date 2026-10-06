"""Native declarations for the estimation composer.

One ``cdef extern`` block per header, in dependency order, each typedef under
the header that declares it rather than under whichever one happens to reach it.
The context structs embed the solve specs and the filter input structs by value,
so those have to be declared before ``estimation.h``.

Same stem as ``_estimation.pyx``, so these are visible there with no cimport.
"""

from libc.stdint cimport int64_t

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

cdef extern from "sdsge_common.h":
    ctypedef struct arena_size:
        int64_t n_float
        int64_t n_int


cdef extern from "../core/klein_solve.h":
    ctypedef void (*sdsge_residual_fn)()
    ctypedef void (*bc_residual_fn)()
    ctypedef void (*klein_zgges_fn)()
    ctypedef void (*sdsge_dgeqrf_fn)()
    ctypedef void (*sdsge_dormqr_fn)()
    int64_t sdsge_pencil_dim(const signed char *incidence, int64_t n_var)
    ctypedef void (*sdsge_ztgexc_fn)()

    ctypedef struct klein_spec:
        sdsge_residual_fn residual
        klein_zgges_fn zgges
        sdsge_dgeqrf_fn dgeqrf
        sdsge_dormqr_fn dormqr
        sdsge_ztgexc_fn ztgexc
        const double *ss_seed
        const double *params
        const signed char *incidence
        int64_t n_var
        int64_t n_state
        int64_t n_ctrl
        int64_t n_exog
        int64_t n_par

    ctypedef struct sgu_klein_spec:
        klein_spec first
        bc_residual_fn bc_residual
        const double *Q

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

    arena_size sdsge_klein_solve1_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t nd) nogil
    arena_size sdsge_sgu_klein_solve2_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t nd) nogil


cdef extern from "../kalman/kalman.h":
    ctypedef void (*meas_fn)()

    ctypedef struct kf_inputs:
        int64_t n
        int64_t m
        int64_t k
        int64_t T
        const double *A
        const double *B
        const double *C
        const double *d
        const double *Q
        const double *R
        const double *steady_state
        const double *y
        const double *x0
        const double *P0
        int symmetrize
        int joseph_cov
        double jitter
        int return_shocks
        int store_history

    ctypedef struct ekf_inputs:
        meas_fn meas
        meas_fn jac
        const double *A
        const double *B
        const double *calib_params
        const double *Q
        const double *R
        const double *steady_state
        const double *y
        const double *x0
        const double *P0
        int64_t T
        int64_t n
        int64_t m
        int64_t k
        int64_t n_par
        double jitter
        int symmetrize
        int joseph_cov
        int compute_y_filt
        int return_shocks
        int store_history

    ctypedef struct ukf_inputs:
        meas_fn meas
        const double *hx
        const double *gx
        const double *bu
        const double *hxx
        const double *gxx
        const double *hxu
        const double *gxu
        const double *huu
        const double *guu
        const double *hss
        const double *gss
        const double *steady_state
        const double *params
        const double *Q
        const double *R
        const double *obs
        const double *z0
        const double *P0
        int64_t T
        int64_t n_state
        int64_t n_ctrl
        int64_t n_exog
        int64_t n_obs
        int64_t n_params
        double alpha
        double beta
        double kappa
        double jitter
        int symmetrize
        int store_history

    arena_size kf_arena_size(int64_t n, int64_t m, int64_t k) nogil
    arena_size ekf_arena_size(int64_t n, int64_t m, int64_t k) nogil
    arena_size ukf_arena_size(
        int64_t n_state, int64_t n_ctrl, int64_t n_exog, int64_t n_obs,
    ) nogil


cdef extern from "estimation.h":
    ctypedef struct sdsge_param_map:
        const double *base_params
        const int64_t *theta_idx
        const int64_t *param_slot
        const int64_t *transform_code
        const double *transform_params
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

    ctypedef struct sdsge_cov_build:
        sdsge_cov_spec spec
        double *out
        double *corr
        double *std

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

    ctypedef struct sdsge_obj_common:
        sdsge_param_map pmap
        sdsge_cov_build q
        sdsge_cov_build r
        sdsge_prior_tables prior
        int derive_P0
        double *params
        double *arena
        int64_t *iarena
        int64_t bk_violations

    ctypedef struct sdsge_linear_ctx:
        sdsge_obj_common base
        klein_spec solve_ctx
        kf_inputs kf_ctx
        meas_fn meas
        meas_fn jac
        sdsge_solve1 solve_out

    ctypedef struct sdsge_extended_ctx:
        sdsge_obj_common base
        klein_spec solve_ctx
        ekf_inputs ekf_ctx
        sdsge_solve1 solve_out

    ctypedef struct sdsge_unscented_ctx:
        sdsge_obj_common base
        sgu_klein_spec solve_ctx
        ukf_inputs ukf_ctx
        sdsge_solve1 solve1_out
        sdsge_solve2 solve2_out

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

    arena_size sdsge_linear_obj_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t n_obs, int64_t nd) nogil
    arena_size sdsge_extended_obj_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t n_obs, int64_t nd) nogil
    arena_size sdsge_unscented_obj_arena_size(
        int64_t n_var, int64_t n_state, int64_t n_ctrl, int64_t n_par,
        int64_t n_exog, int64_t n_obs, int64_t nd) nogil

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
