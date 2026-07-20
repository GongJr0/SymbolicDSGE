#ifndef SDSGE_ESTIMATION_H
#define SDSGE_ESTIMATION_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

/* Type providers for the context struct's runtime-address fields. Included for
 * their typedefs only; nothing here is called at this stage, so no link
 * dependency is introduced yet (relative paths resolve against this header's
 * directory regardless of the -I search path). */
#include "../core/klein_preproc.h" /* sdsge_residual_fn */
#include "../core/klein_qz.h"      /* klein_zgges_fn    */
#include "../kalman/kalman.h"      /* meas_fn           */
#include "prior_program.h" /* SdsgeTransformCode, SDSGE_N_TRANSFORM_PARAMS */

/* ========================================================================== *
 * Native estimation objective (issue #327), context + builders.
 *
 * The whole per-eval objective runs in C: theta arrives from the optimizer or
 * sampler, and one scalar (loglik, or loglik + logprior) comes back. Everything
 * loop-invariant is resolved once by a Python prep step into the context struct
 * below; every name to index lookup that dies at the C boundary becomes a
 * packed integer table here. All external routines (model residual, measurement
 * and Jacobian cfuncs, LAPACK zgges) arrive as runtime function-pointer
 * addresses, so this translation unit links against nothing new.
 *
 * PARAMETER MODEL. There is one flat resolved-value vector `params` of length
 * `n_params`, keyed by a global slot index assigned in prep to every named
 * parameter (model calibration, shock std/corr, measurement-noise std/corr).
 * Each eval starts from `base_params` (the calibrated baseline) and the theta
 * scatter overwrites the estimated slots. Every `*_slot` field below indexes
 * this vector. The model solve and measurement cfuncs consume a calibration
 * subvector gathered from `params` via `calib_gather`.
 * ========================================================================== */

/* Filter mode. Values are independent of the Python FilterMode enum; the prep
 * step maps "linear"/"extended"/"unscented" onto these. The first native
 * milestone implements SDSGE_FILTER_LINEAR only. */
typedef enum {
  SDSGE_FILTER_LINEAR = 0,
  SDSGE_FILTER_EXTENDED = 1,
  SDSGE_FILTER_UNSCENTED = 2
} sdsge_filter_mode;

/* -------------------------------------------------------------------------- *
 * Inlined covariance builders (Q and R share the same shape).
 *
 * Q = outer(std, std) * corr with K = n_exog; the R config-parameter regime is
 * identical with K = n_obs. Row-major K*K throughout. The `R_override` and
 * fixed-slice R regimes, and a Q with no estimated std/corr, are loop-invariant
 * and baked to a constant in prep (see sdsge_cov_spec.is_constant), so these
 * builders run only for the genuinely per-eval regime.
 * -------------------------------------------------------------------------- */

/* corr := I_K, then corr[i,j] = corr[j,i] = params[pair_slot[p]] for each of the
 * `n_pairs` packed off-diagonal correlation entries. The named/frozenset pair
 * lookups from build_Q are resolved to (i, j, slot) triples in prep. */
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

/* out[i,j] := std[i] * std[j] * corr[i,j], with std[k] = params[std_slots[k]].
 * `corr` is either the identity-plus-pairs matrix from sdsge_assemble_corr or a
 * correlation materialized by the theta-resolution step for a CholeskyCorr
 * block. */
static inline void sdsge_cov_from_std_corr(const i64 *SDSGE_RESTRICT std_slots,
                                           const f64 *SDSGE_RESTRICT params,
                                           const f64 *SDSGE_RESTRICT corr, i64 K,
                                           f64 *SDSGE_RESTRICT out) {
  for (i64 i = 0; i < K; ++i) {
    const f64 si = params[std_slots[i]];
    for (i64 j = 0; j < K; ++j) {
      const f64 sj = params[std_slots[j]];
      out[i * K + j] = si * sj * corr[i * K + j];
    }
  }
}

/* -------------------------------------------------------------------------- *
 * Context sub-structs.
 * -------------------------------------------------------------------------- */

/* How one covariance (Q or R) is produced each eval. */
typedef struct {
  int is_constant;     /* 1: copy `constant` as-is (baked in prep). */
  const f64 *constant; /* K*K row-major, valid when is_constant; else NULL. */
  i64 K;               /* matrix dimension (n_exog for Q, n_obs for R). */
  const i64 *std_slots; /* K param slots for the diagonal stds. */
  /* Correlation source when not constant. */
  int corr_from_block;  /* 1: correlation is materialized into workspace by a
                           CholeskyCorr block; 0: identity + packed pairs. */
  const i64 *pair_i;    /* n_pairs off-diagonal entries (row). */
  const i64 *pair_j;    /* n_pairs (col). */
  const i64 *pair_slot; /* n_pairs param slots. */
  i64 n_pairs;
} sdsge_cov_spec;

/* One non-block scalar: params[param_slot] = inverse_transform(theta[theta_idx]).
 * The transform dispatch uses sdsge_transform_inverse_and_logjac (prior_program).
 * The estimation objective needs only the inverted value; the log-jacobian is a
 * prior-side concern handled by sdsge_logprior_program. */
typedef struct {
  i64 theta_idx;
  i64 param_slot;
  i64 transform_code; /* SdsgeTransformCode */
  f64 transform_params[SDSGE_N_TRANSFORM_PARAMS];
} sdsge_scalar_scatter;

/* One CholeskyCorr matrix block: theta[theta_off .. theta_off + theta_len]
 * reparameterizes a K*K correlation. Its members scatter into `params` at
 * `member_slots` (positioned by (pos_row, pos_col) in the block), and the block
 * correlation feeds whichever cov_spec has corr_from_block set. The exact
 * correlation-from-theta call firms up with the theta-resolution TU. */
typedef struct {
  i64 theta_off;
  i64 theta_len;
  i64 K;
  const i64 *member_slots; /* n_members param slots. */
  const i64 *pos_row;      /* n_members block rows. */
  const i64 *pos_col;      /* n_members block cols. */
  i64 n_members;
} sdsge_block_scatter;

/* Packed log-prior program, the argument bundle for sdsge_logprior_program. */
typedef struct {
  int has_prior; /* 0: flat prior, logprior contributes 0. */
  const i64 *scalar_indices;
  const i64 *scalar_dist_codes;
  const i64 *scalar_transform_codes;
  const f64 *scalar_dist_params;
  const f64 *scalar_transform_params;
  i64 n_scalar;
  const i64 *matrix_offsets;
  const i64 *matrix_dims;
  const i64 *matrix_lengths;
  const f64 *matrix_etas;
  const f64 *matrix_log_constants;
  i64 n_blocks;
} sdsge_prior_tables;

/* Dimensions. Shared by every flavor. `n_var` is the square QZ pencil dimension
 * and, because the first-order state space is [states; controls], also the
 * filter state dimension: A is (n_var, n_var), B is (n_var, n_exog), and the
 * linear C is (n_obs, n_var). n_ctrl = n_var - n_state. */
typedef struct {
  i64 n_theta;  /* estimated parameter count (length of theta). */
  i64 n_var;    /* pencil / state-space dimension (nx + ny). */
  i64 n_state;  /* nx: predetermined states. */
  i64 n_ctrl;   /* ny: controls / jumps. */
  i64 n_exog;   /* k: exogenous shocks. */
  i64 n_obs;    /* m: observables. */
  i64 n_par;    /* calibration subvector length (residual / meas cfunc params). */
  i64 n_params; /* total named-parameter slots in the flat `params` vector. */
  i64 T;        /* observation count. */
} sdsge_dims;

/* theta -> params resolution: the flat baseline plus the scatter tables, and
 * the gather that forms the calibration subvector the cfuncs consume. */
typedef struct {
  const f64 *base_params; /* n_params calibrated baseline (memcpy'd per eval). */
  const sdsge_scalar_scatter *scalars;
  i64 n_scalars;
  const sdsge_block_scatter *blocks;
  i64 n_blocks;
  const i64 *calib_gather; /* n_par slots into `params` -> calib subvector. */
} sdsge_param_map;

/* First-order Klein solve outputs, shared by all three flavors (the linear /
 * extended filters use A, B; the unscented filter is second-order but still
 * needs the first-order p, f, B). All scratch is preallocated in prep. */
typedef struct {
  f64 *a_real;  /* n_var*n_var: klein_preproc A pencil. */
  f64 *b_real;  /* n_var*n_var: klein_preproc B pencil. */
  c128 *s;      /* n_var*n_var: QZ Schur factor S (pencil in, overwritten). */
  c128 *t;      /* n_var*n_var: QZ Schur factor T. */
  c128 *z;      /* n_var*n_var: QZ right Schur vectors. */
  c128 *f;      /* n_ctrl*n_state: policy g_x (klein_postproc f). */
  c128 *p;      /* n_state*n_state: policy h_x (klein_postproc p). */
  c128 *eig;    /* n_var: generalized eigenvalues. */
  i64 stab;     /* klein_postproc stability code (0 == ok). */
  f64 *A;       /* n_var*n_var: real state transition for the filter. */
  f64 *B;       /* n_var*n_exog: real shock loading. */
} sdsge_solve1;

/* Second-order (SGU) additions, unscented flavor only. Builds on sdsge_solve1
 * (real h_x = Re(p), g_x = Re(f), and bx = B[:n_state]). */
typedef struct {
  f64 *f_xx;         /* n_var*(2 n_var)*(2 n_var): residual Hessian (bicomplex). */
  f64 *hx_real;      /* n_state*n_state: Re(p) for the UKF. */
  f64 *gx_real;      /* n_ctrl*n_state: Re(f) for the UKF. */
  f64 *bx;           /* n_state*n_exog: B[:n_state, :]. */
  f64 *eta;          /* n_state*n_exog: shock loading (for the risk correction). */
  f64 *gxx;          /* n_ctrl*n_state*n_state. */
  f64 *hxx;          /* n_state*n_state*n_state. */
  f64 *gss;          /* n_ctrl: sigma^2 risk correction. */
  f64 *hss;          /* n_state. */
  f64 *steady_state; /* n_var: nonlinear expansion point for the tensors. */
} sdsge_solve2;

/* -------------------------------------------------------------------------- *
 * Mode-independent I/O, embedded by value in every flavor context.
 * -------------------------------------------------------------------------- */
typedef struct {
  sdsge_dims dims;

  /* Runtime addresses (all handed in from Python). */
  sdsge_residual_fn residual; /* klein_preproc model residual cfunc. */
  klein_zgges_fn zgges;       /* LAPACK zgges pointer. */
  meas_fn meas;               /* measurement cfunc. */
  meas_fn jac;                /* observable-Jacobian cfunc. */

  /* Solve inputs. */
  const f64 *steady_state; /* n_var. */
  int log_linear;

  /* Filter inputs shared across modes. */
  const f64 *y;  /* T*n_obs observations, row-major. */
  const f64 *P0; /* initial state covariance (flavor-sized). */
  const f64 *x0; /* initial state mean, or NULL. */
  f64 jitter;
  int symmetrize;

  /* theta -> params and the covariance / prior specs. */
  sdsge_param_map pmap;
  sdsge_cov_spec q_spec;
  sdsge_cov_spec r_spec;
  sdsge_prior_tables prior;

  /* Mode-independent scratch (allocated once). */
  f64 *params;    /* n_params: base_params + theta scatter. */
  f64 *calib_vec; /* n_par: params gathered via calib_gather. */
  f64 *Q;         /* n_exog*n_exog. */
  f64 *R;         /* n_obs*n_obs. */
  f64 *corr_q;    /* n_exog*n_exog: assembled or block correlation for Q. */
  f64 *corr_r;    /* n_obs*n_obs: assembled or block correlation for R. */

  /* Running BK (Blanchard-Kahn) violation tally, from the native solve status
   * (folds in the retired #326). The objective increments this and returns the
   * +-inf sentinel when the solve is non-finite / unstable. */
  i64 bk_violations;
} sdsge_obj_common;

/* -------------------------------------------------------------------------- *
 * Flavor contexts. Order and mode are resolved once in prep, which builds the
 * matching context and exposes the matching objective address; there is no
 * per-eval order branch and no optional / unused field.
 * -------------------------------------------------------------------------- */

/* Linear filter (first order). Precomputes the linear measurement (C, d) once
 * per eval from the meas / jac cfuncs at the linearization point. */
typedef struct {
  sdsge_obj_common base;
  sdsge_solve1 solve;
  const f64 *zero_state; /* n_var: linearization point for C, d. */
  f64 *C;                /* n_obs*n_var: linear measurement. */
  f64 *d;                /* n_obs: measurement intercept. */
} sdsge_obj_linear;

/* Extended filter (first order). Relinearizes via the meas / jac cfuncs each
 * step, so it needs no precomputed (C, d): base + first-order solve is the whole
 * I/O surface. */
typedef struct {
  sdsge_obj_common base;
  sdsge_solve1 solve;
} sdsge_obj_extended;

/* Unscented filter (second order). First-order solve is a prerequisite for the
 * second-order tensors and the bx slice. */
typedef struct {
  sdsge_obj_common base;
  sdsge_solve1 solve;
  sdsge_solve2 solve2;
  f64 *z0;    /* 2*n_state: augmented initial mean. */
  f64 alpha;  /* UKF sigma-point spread. */
  f64 beta;   /* prior-knowledge term. */
  f64 kappa;  /* secondary scaling. */
} sdsge_obj_unscented;

/* Per-flavor scalar objectives: theta -> loglik (want_prior == 0) or
 * loglik + logprior (want_prior != 0). Each writes its scratch and
 * base.bk_violations, and returns the +-inf sentinel on a BK violation or a
 * non-finite intermediate. Defined in estimation.c (orchestrator, forthcoming);
 * prep exposes exactly one address per constructed estimator. */
f64 sdsge_objective_linear(sdsge_obj_linear *ctx,
                           const f64 *SDSGE_RESTRICT theta, int want_prior);
f64 sdsge_objective_extended(sdsge_obj_extended *ctx,
                             const f64 *SDSGE_RESTRICT theta, int want_prior);
f64 sdsge_objective_unscented(sdsge_obj_unscented *ctx,
                              const f64 *SDSGE_RESTRICT theta, int want_prior);

#endif /* SDSGE_ESTIMATION_H */
