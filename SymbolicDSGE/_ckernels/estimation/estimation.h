#ifndef SDSGE_ESTIMATION_H
#define SDSGE_ESTIMATION_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include "../core/klein_preproc.h" /* sdsge_residual_fn */
#include "../core/klein_qz.h"      /* klein_zgges_fn */
#include "../kalman/kalman.h"      /* meas_fn */
#include "prior_program.h"         /* transform codes, dispatch */

/* Native estimation objective context and theta-fill (issue #327). */

/* Filter mode. */
typedef enum {
  SDSGE_FILTER_LINEAR = 0,
  SDSGE_FILTER_EXTENDED = 1,
  SDSGE_FILTER_UNSCENTED = 2
} sdsge_filter_mode;

/* corr(K*K) := I, then off-diagonal pairs corr[i,j]=corr[j,i]=params[slot]. */
static inline void sdsge_assemble_corr(const i64 *SDSGE_RESTRICT pair_i,
                                       const i64 *SDSGE_RESTRICT pair_j,
                                       const i64 *SDSGE_RESTRICT pair_slot,
                                       i64 n_pairs,
                                       const f64 *SDSGE_RESTRICT params, i64 K,
                                       f64 *SDSGE_RESTRICT corr) {
  for (i64 r = 0; r < K; ++r) {
    for (i64 c = 0; c < K; ++c) {
      corr[r * K + c] = (r == c) ? 1.0 : 0.0;
    }
  }
  for (i64 p = 0; p < n_pairs; ++p) {
    const i64 i = pair_i[p];
    const i64 j = pair_j[p];
    const f64 v = params[pair_slot[p]];
    corr[i * K + j] = v;
    corr[j * K + i] = v;
  }
}

/* out(K*K) := outer(std, std) * corr, with std[k] = params[std_slots[k]]. */
static inline void sdsge_cov_from_std_corr(const i64 *SDSGE_RESTRICT std_slots,
                                           const f64 *SDSGE_RESTRICT params,
                                           const f64 *SDSGE_RESTRICT corr,
                                           i64 K, f64 *SDSGE_RESTRICT out) {
  for (i64 i = 0; i < K; ++i) {
    const f64 si = params[std_slots[i]];
    for (i64 j = 0; j < K; ++j) {
      const f64 sj = params[std_slots[j]];
      out[i * K + j] = si * sj * corr[i * K + j];
    }
  }
}

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
  i64 n_par;    /* calib subvector length */
  i64 n_params; /* total param slots */
  i64 T;        /* observations */
} sdsge_dims;

/* theta -> params resolution tables. */
typedef struct {
  const f64 *base_params;              /* n_params */
  const sdsge_scalar_scatter *scalars; /* n_scalars */
  i64 n_scalars;
  const i64 *calib_gather; /* n_par */
  const i64 *calib_upd;    /* n_calib_upd */
  i64 n_calib_upd;
} sdsge_param_map;

/* First-order Klein solve outputs. */
typedef struct {
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
  klein_zgges_fn zgges;
  meas_fn meas;
  meas_fn jac;

  const f64 *steady_state; /* n_var */
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

  f64 *params;    /* n_params */
  f64 *calib_vec; /* n_par */
  f64 *Q;         /* n_exog*n_exog */
  f64 *R;         /* n_obs*n_obs */
  f64 *corr_q;    /* n_exog*n_exog */
  f64 *corr_r;    /* n_obs*n_obs */

  i64 bk_violations;
} sdsge_obj_common;

/* Seed params(n_params) with the calibrated baseline (once, at construction). */
static inline void sdsge_init_params(f64 *SDSGE_RESTRICT params,
                                     const f64 *SDSGE_RESTRICT base_params,
                                     i64 n_params) {
  for (i64 i = 0; i < n_params; ++i) {
    params[i] = base_params[i];
  }
}

/* Seed calib_vec(n_par) from params (once, after sdsge_init_params). */
static inline void sdsge_init_calib(f64 *SDSGE_RESTRICT calib_vec,
                                    const f64 *SDSGE_RESTRICT params,
                                    const i64 *SDSGE_RESTRICT calib_gather,
                                    i64 n_par) {
  for (i64 i = 0; i < n_par; ++i) {
    calib_vec[i] = params[calib_gather[i]];
  }
}

/* Per-eval: scatter estimated scalars into params, refresh estimated calib_vec. */
static inline void sdsge_fill_params(sdsge_obj_common *base,
                                     const f64 *SDSGE_RESTRICT theta) {
  f64 x, logjac;
  for (i64 s = 0; s < base->pmap.n_scalars; ++s) {
    const sdsge_scalar_scatter *sc = &base->pmap.scalars[s];
    sdsge_transform_inverse_and_logjac(sc->transform_code,
                                       (f64 *)sc->transform_params,
                                       theta[sc->theta_idx], &x, &logjac);
    base->params[sc->param_slot] = x;
  }
  for (i64 u = 0; u < base->pmap.n_calib_upd; ++u) {
    const i64 pos = base->pmap.calib_upd[u];
    base->calib_vec[pos] = base->params[base->pmap.calib_gather[pos]];
  }
}

/* Linear-filter objective context. */
typedef struct {
  sdsge_obj_common base;
  sdsge_solve1 solve;
  const f64 *zero_state; /* n_var */
  f64 *C;                /* n_obs*n_var */
  f64 *d;                /* n_obs */
} sdsge_obj_linear;

/* Extended-filter objective context. */
typedef struct {
  sdsge_obj_common base;
  sdsge_solve1 solve;
} sdsge_obj_extended;

/* Unscented-filter objective context. */
typedef struct {
  sdsge_obj_common base;
  sdsge_solve1 solve;
  sdsge_solve2 solve2;
  f64 *z0; /* 2*n_state */
  f64 alpha;
  f64 beta;
  f64 kappa;
} sdsge_obj_unscented;

/* Per-flavor objective: theta -> loglik (+ logprior if has_priors). */
f64 sdsge_objective_linear(sdsge_obj_linear *ctx,
                           const f64 *SDSGE_RESTRICT theta, int has_priors);
f64 sdsge_objective_extended(sdsge_obj_extended *ctx,
                             const f64 *SDSGE_RESTRICT theta, int has_priors);
f64 sdsge_objective_unscented(sdsge_obj_unscented *ctx,
                              const f64 *SDSGE_RESTRICT theta, int has_priors);

#endif /* SDSGE_ESTIMATION_H */
