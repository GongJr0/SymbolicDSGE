#ifndef SDSGE_ESTIMATION_H
#define SDSGE_ESTIMATION_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include "../core/bicomplex_hessian.h" /* bc_residual_fn */
#include "../core/klein_preproc.h"     /* sdsge_residual_fn */
#include "../core/klein_qz.h"          /* klein_zgges_fn */
#include "../core/klein_solve.h"       /* klein specs, sdsge_solve1/2 */
#include "../kalman/kalman.h"          /* meas_fn */
#include "../optim/optim.h"            /* sdsge_objective_fn */
#include "prior_program.h"             /* transform codes, dispatch */

/* Native estimation objective context and theta-fill (issue #327). */

/* Filter mode. */
typedef enum {
  FILTER_LINEAR = 0,
  FILTER_EXTENDED = 1,
  FILTER_UNSCENTED = 2
} sdsge_filter_mode;

/* Optimizer selection shared by point estimation and MCMC's MAP setup. */
typedef enum {
  ESTIMATION_LBFGSB = 0,
  ESTIMATION_NELDER_MEAD = 1
} sdsge_estimation_method;

/* Native counterpart of run_estimation's optimizer arguments. `nbd == NULL`
 * denotes unbounded optimization; otherwise it uses the L-BFGS-B convention
 * {0 none, 1 lower, 2 both, 3 upper}. The caller owns theta and all bound
 * buffers for the duration of the call. */
typedef struct {
  sdsge_filter_mode filter_mode;
  sdsge_estimation_method method;
  int has_priors;
  const f64 *lo;
  const f64 *hi;
  const i64 *nbd;
  sdsge_optim_options optim;
  int compute_cov;       /* fill the result's vcov / se */
  f64 cov_fd_step_scale; /* as sdsge_estimation_cov_factor's fd_step_scale */
  f64 cov_fd_absolute_floor; /* as its fd_absolute_floor */
} sdsge_estimation_options;

/* Q or R covariance build spec. */
typedef struct {
  int is_constant;
  const f64 *constant;  /* K*K, or NULL */
  i64 K;                /* n_exog (Q) or n_obs (R) */
  const i64 *std_slots; /* K */
  int corr_from_block;
  i64 block_theta_off;
  i64 block_theta_len;
  const i64 *pair_i;    /* n_pairs */
  const i64 *pair_j;    /* n_pairs */
  const i64 *pair_slot; /* n_pairs */
  i64 n_pairs;
} sdsge_cov_spec;

/* A spec whose correlation is a live CPC block: the only shape that owns a run
 * of theta, and so the only one a theta -> parameters map has work to do on. */
static inline int sdsge_spec_has_block(const sdsge_cov_spec *sp) {
  return !sp->is_constant && sp->corr_from_block;
}

/* One estimated scalar's theta -> params scatter. */
typedef struct {
  i64 theta_idx;
  i64 param_slot;
  i64 transform_code;
  f64 transform_params[SDSGE_N_TRANSFORM_PARAMS];
} sdsge_scalar_scatter;

/* Packed log-prior program arguments. */
typedef struct {
  int has_prior;
  const i64 *scalar_indices;          /* n_scalar */
  const i64 *scalar_dist_codes;       /* n_scalar */
  const i64 *scalar_transform_codes;  /* n_scalar */
  const f64 *scalar_dist_params;      /* n_scalar*5 */
  const f64 *scalar_transform_params; /* n_scalar*3 */
  i64 n_scalar;
  const i64 *matrix_offsets;       /* n_blocks */
  const i64 *matrix_dims;          /* n_blocks */
  const i64 *matrix_lengths;       /* n_blocks */
  const f64 *matrix_etas;          /* n_blocks */
  const f64 *matrix_log_constants; /* n_blocks */
  i64 n_blocks;
  /* Which density the prior evaluates to; see sdsge_logprior_program. Set per
   * entry point: the sampler walks theta and takes it, a maximizer reporting a
   * parameter value does not, because the jacobian moves the mode. */
  int include_logjac;
} sdsge_prior_tables;

/* Model and data dimensions. */
typedef struct {
  i64 n_theta; /* estimated params */
  i64 n_var;   /* nx + ny (pencil / filter dim) */
  i64 n_state; /* nx */
  i64 n_ctrl;  /* ny */
  i64 n_exog;  /* k */
  i64 n_obs;   /* m */
  i64 n_par;   /* calib params */
  i64 T;       /* observations */
} sdsge_dims;

/* theta -> params resolution tables. base_params and every slot index
 * (scalars' param_slot, cov std_slots/pair_slot) are in calib_params order, so
 * params doubles as the residual/measurement argument vector: no gather. */
typedef struct {
  const f64 *base_params;              /* n_par */
  const sdsge_scalar_scatter *scalars; /* n_scalars */
  i64 n_scalars;
} sdsge_param_map;

/* Mode-independent objective context. */
typedef struct {
  sdsge_dims dims;

  sdsge_residual_fn residual;
  bc_residual_fn bc_residual;

  klein_zgges_fn zgges;
  sdsge_dgeqrf_fn dgeqrf;
  sdsge_dormqr_fn dormqr;
  meas_fn meas;
  meas_fn jac;

  const f64 *ss_seed;  /* n_var: Newton seed for the steady state */
  const i8 *incidence; /* n_var: SDSGE_INC_* bits, unioned over the regimes */
  const f64 *y;        /* T*n_obs */
  f64 *P0;       /* explicit n_var*n_var prior; NULL derives it after solve */
  const f64 *x0; /* n_var, or NULL */
  f64 jitter;
  int symmetrize;
  int joseph_cov;
  int derive_P0; /* if P0 is NULL, derive it from the stationary covariance */

  sdsge_param_map pmap;
  sdsge_cov_spec q_spec;
  sdsge_cov_spec r_spec;
  sdsge_prior_tables prior;

  f64 *params; /* n_par; calib_params order, residual/meas argument vector */
  f64 *Q;      /* n_exog*n_exog */
  f64 *chol;   /* n_exog*n_exog: chol(Q), refactored only when Q moves */
  f64 *R;      /* n_obs*n_obs */
  f64 *corr_q; /* n_exog*n_exog */
  f64 *corr_r; /* n_obs*n_obs */
  f64 *std_q;  /* n_exog */
  f64 *std_r;  /* n_obs */

  f64 *filter_arena; /* scratch for the filter sizeof(f64)*<filter>_arena_size()
                        reused for the P0 == NULL case, which occurs before the
                        filter.
                      */

  /* Scratch for the per-draw solve, sized by sdsge_klein_solve1_arena_size or
   * sdsge_sgu_klein_solve2_arena_size. Held for the run so no draw allocates.
   */
  f64 *solve_arena;
  i64 *solve_iarena;

  i64 bk_violations;
} sdsge_obj_common;

/* Linear-filter objective context. */
typedef struct {
  sdsge_obj_common base;
  sdsge_solve1 solve;
  f64 *C; /* n_obs*n_var */
  f64 *d; /* n_obs */
} sdsge_linear_ctx;

/* Extended-filter objective context. */
typedef struct {
  sdsge_obj_common base;
  sdsge_solve1 solve;
} sdsge_extended_ctx;

/* Unscented-filter objective context. */
typedef struct {
  sdsge_obj_common base;
  sdsge_solve1 solve;
  sdsge_solve2 solve2;
  f64 *z0; /* 2*n_state */
  f64 alpha;
  f64 beta;
  f64 kappa;
} sdsge_unscented_ctx;

/* One-time construction seeds (called once, from the ctx composer). */
void sdsge_init_params(f64 *SDSGE_RESTRICT params,
                       const f64 *SDSGE_RESTRICT base_params, i64 n_par);

/* Post-loop resolution at a theta (e.g. x_best): scatter into params, and the
 * log-prior from the packed tables. Both are scatter / prior only, no filter,
 * so they are cheap to call once after the optimizer returns. Shared by every
 * mode (they operate on the common base). */
void sdsge_scatter_params(sdsge_obj_common *SDSGE_RESTRICT base,
                          const f64 *SDSGE_RESTRICT theta);
f64 sdsge_logprior_at(const sdsge_obj_common *SDSGE_RESTRICT base,
                      const f64 *SDSGE_RESTRICT theta);

/* Status codes for the covariance, reported separately from the optimizer's. */
#define SDSGE_ESTIMATION_OK 0
#define SDSGE_ESTIMATION_EALLOC (-1800)
#define SDSGE_ESTIMATION_EHESSIAN (-1801)
#define SDSGE_ESTIMATION_ENOTSPD (-1802)

/* Covariance of the estimate at `theta`, in factored form: `factor` satisfies
 * factor * factor^T = H^-1, where H = -d^2 logpost(theta). `fd_step_scale` and
 * `fd_absolute_floor` define h_i = max(abs(theta_i), fd_absolute_floor) *
 * DBL_EPSILON^(1/6) * fd_step_scale. The caller owns the d*d row-major
 * `factor` output and supplies `work`, at least 4*d + 2*d*d f64 of scratch. */
i64 sdsge_estimation_cov_factor(sdsge_objective_fn logpost, void *obj_ctx,
                                const f64 *SDSGE_RESTRICT theta, i64 d,
                                f64 fd_step_scale, f64 fd_absolute_floor,
                                f64 *SDSGE_RESTRICT factor,
                                f64 *SDSGE_RESTRICT work);

/* What an estimation call returns: the optimizer's own result, plus the
 * asymptotic covariance of the point it found. `vcov` (n_theta*n_theta,
 * row-major) is caller-owned and may be NULL, which reads the same as
 * opt->compute_cov == 0. It is the covariance of theta, the vector the
 * optimizer moves, which is also the space the sampler's proposal wants;
 * carrying it into the caller's own parameter space is the caller's job.
 * `cov_status` is deliberately not the optimizer's: a Hessian that is not
 * positive definite at the optimum leaves NaN behind and says so there, while
 * the estimate itself stands. */
typedef struct {
  sdsge_optim_result base;
  f64 *vcov;
  f64 *se;
  i64 cov_status;
} sdsge_estimation_result;

/* Minimize the configured likelihood or negative log posterior in place.
 * `ctx` must point to the filter-mode context selected by opt->filter_mode;
 * theta has length n_theta. `has_priors` selects the posterior objective.
 * The caller owns theta, options, and all bound buffers for the call. */
void sdsge_run_estimation(void *ctx, i64 n_theta, f64 *SDSGE_RESTRICT theta,
                          const sdsge_estimation_options *opt,
                          sdsge_estimation_result *out);

/* Per-flavor objective: theta -> loglik (+ logprior if has_priors). */
f64 sdsge_obj_linear(sdsge_linear_ctx *ctx, const f64 *SDSGE_RESTRICT theta,
                     int has_priors);
f64 sdsge_obj_extended(sdsge_extended_ctx *ctx, const f64 *SDSGE_RESTRICT theta,
                       int has_priors);

f64 sdsge_obj_unscented(sdsge_unscented_ctx *ctx,
                        const f64 *SDSGE_RESTRICT theta, int has_priors);

/* The objectives above as `sdsge_objective_fn`: one closure ABI the optimizer
 * and MCMC drivers can hold, with the mode's ctx recovered from `void *`.
 *
 * `sdsge_min_*` negate, because the drivers minimize; the -inf a rejected draw
 * returns becomes +inf, which their line search reads as no decrease. `_ll` is
 * the likelihood alone and `_lp` folds the log-prior in.
 *
 * `sdsge_post_*` are +logpost and do not negate: MCMC samples a posterior, so
 * priors are always on and the -inf flows through to auto-reject the draw. */
f64 sdsge_min_linear_ll(const f64 *SDSGE_RESTRICT x, void *ctx);
f64 sdsge_min_linear_lp(const f64 *SDSGE_RESTRICT x, void *ctx);
f64 sdsge_min_extended_ll(const f64 *SDSGE_RESTRICT x, void *ctx);
f64 sdsge_min_extended_lp(const f64 *SDSGE_RESTRICT x, void *ctx);
f64 sdsge_min_unscented_ll(const f64 *SDSGE_RESTRICT x, void *ctx);
f64 sdsge_min_unscented_lp(const f64 *SDSGE_RESTRICT x, void *ctx);
f64 sdsge_post_linear(const f64 *SDSGE_RESTRICT x, void *ctx);
f64 sdsge_post_extended(const f64 *SDSGE_RESTRICT x, void *ctx);
f64 sdsge_post_unscented(const f64 *SDSGE_RESTRICT x, void *ctx);

/* Objective for one (sign, priors, mode) triple. `negate` selects the drivers'
 * minimized form over the +value one; `filter_mode` is an `sdsge_filter_mode`.
 * The +value likelihood row is file-static, so this is the only way to it. */
sdsge_objective_fn sdsge_select_objective(int negate, int has_priors,
                                          int filter_mode);

#endif /* SDSGE_ESTIMATION_H */
