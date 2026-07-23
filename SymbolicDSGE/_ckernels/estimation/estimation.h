#ifndef SDSGE_ESTIMATION_H
#define SDSGE_ESTIMATION_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include "../core/bicomplex_hessian.h" /* bc_residual_fn */
#include "../core/klein_preproc.h"     /* sdsge_residual_fn */
#include "../core/klein_qz.h"          /* klein_zgges_fn */
#include "../kalman/kalman.h"          /* meas_fn */
#include "prior_program.h"             /* transform codes, dispatch */

/* Native estimation objective context and theta-fill (issue #327). */

/* Filter mode. */
typedef enum {
  SDSGE_FILTER_LINEAR = 0,
  SDSGE_FILTER_EXTENDED = 1,
  SDSGE_FILTER_UNSCENTED = 2
} sdsge_filter_mode;

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
} sdsge_prior_tables;

/* Model and data dimensions. */
typedef struct {
  i64 n_theta;  /* estimated params */
  i64 n_var;    /* nx + ny (pencil / filter dim) */
  i64 n_state;  /* nx */
  i64 n_ctrl;   /* ny */
  i64 n_exog;   /* k */
  i64 n_obs;    /* m */
  i64 n_par; /* calib params */
  i64 T;     /* observations */
} sdsge_dims;

/* theta -> params resolution tables. base_params and every slot index
 * (scalars' param_slot, cov std_slots/pair_slot) are in calib_params order, so
 * params doubles as the residual/measurement argument vector: no gather. */
typedef struct {
  const f64 *base_params;              /* n_par */
  const sdsge_scalar_scatter *scalars; /* n_scalars */
  i64 n_scalars;
} sdsge_param_map;

/* First-order Klein solve outputs. */
typedef struct {
  f64 *ss;     /* n_var: Newton-resolved steady state (from ss_seed) */
  f64 *a_real; /* n_var*n_var */
  f64 *b_real; /* n_var*n_var */
  c128 *s;     /* n_var*n_var */
  c128 *t;     /* n_var*n_var */
  c128 *z;     /* n_var*n_var */
  c128 *f;     /* n_ctrl*n_state */
  c128 *p;     /* n_state*n_state */
  c128 *eig;   /* n_var */
  i64 stab;
  f64 *A; /* n_var*n_var */
  f64 *B; /* n_var*n_exog */
} sdsge_solve1;

/* Second-order (SGU) solve outputs. */
typedef struct {
  f64 *f_xx;         /* n_var*(2*n_var)*(2*n_var) */
  f64 *hx_real;      /* n_state*n_state */
  f64 *gx_real;      /* n_ctrl*n_state */
  f64 *bx;           /* n_state*n_exog */
  f64 *eta;          /* n_state*n_exog */
  f64 *gxx;          /* n_ctrl*n_state*n_state */
  f64 *hxx;          /* n_state*n_state*n_state */
  f64 *gss;          /* n_ctrl */
  f64 *hss;          /* n_state */
  f64 *steady_state; /* n_var */
} sdsge_solve2;

/* Mode-independent objective context. */
typedef struct {
  sdsge_dims dims;

  sdsge_residual_fn residual;
  bc_residual_fn bc_residual;

  klein_zgges_fn zgges;
  meas_fn meas;
  meas_fn jac;

  const f64 *ss_seed; /* n_var: Newton seed for the steady state */
  int log_linear;

  const f64 *y;  /* T*n_obs */
  const f64 *P0; /* n_var*n_var; UKF 2n_state*2n_state */
  const f64 *x0; /* n_var, or NULL */
  f64 jitter;
  int symmetrize;

  sdsge_param_map pmap;
  sdsge_cov_spec q_spec;
  sdsge_cov_spec r_spec;
  sdsge_prior_tables prior;

  f64 *params;    /* n_par; calib_params order, residual/meas argument vector */
  f64 *Q;         /* n_exog*n_exog */
  f64 *R;         /* n_obs*n_obs */
  f64 *corr_q;    /* n_exog*n_exog */
  f64 *corr_r;    /* n_obs*n_obs */
  f64 *std_q;     /* n_exog */
  f64 *std_r;     /* n_obs */

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
 * log-prior from the packed tables. Both are scatter / prior only, no filter, so
 * they are cheap to call once after the optimizer returns. Shared by every mode
 * (they operate on the common base). */
void sdsge_scatter_params(sdsge_obj_common *SDSGE_RESTRICT base,
                          const f64 *SDSGE_RESTRICT theta);
f64 sdsge_logprior_at(const sdsge_obj_common *SDSGE_RESTRICT base,
                      const f64 *SDSGE_RESTRICT theta);

/* Per-flavor objective: theta -> loglik (+ logprior if has_priors). */
f64 sdsge_obj_linear(sdsge_linear_ctx *ctx, const f64 *SDSGE_RESTRICT theta,
                     int has_priors);
f64 sdsge_obj_extended(sdsge_extended_ctx *ctx, const f64 *SDSGE_RESTRICT theta,
                       int has_priors);

f64 sdsge_obj_unscented(sdsge_unscented_ctx *ctx,
                        const f64 *SDSGE_RESTRICT theta, int has_priors);

/* Hessian Step Constant */
#define SDSGE_HESSIAN_STEP 1e-4

#endif /* SDSGE_ESTIMATION_H */
