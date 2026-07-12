#ifndef SDSGE_PRIOR_PROGRAM_H
#define SDSGE_PRIOR_PROGRAM_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"

/* Integer dispatch codes for the packed log-prior kernel.
 *
 * These MUST stay in lockstep with the DistCode / TransformCode IntEnums in
 * prior_program.py (same names, same values): the Python side packs every prior
 * into int64 code arrays keyed by these values, and the native kernel switches
 * on them. SDSGE_N_*_PARAMS are the packed-row strides (N_DIST_PARAMS /
 * N_TRANSFORM_PARAMS in that module). Plain integer enums -- no CPython, no
 * NumPy -- so the kernel can include this header directly. */

typedef enum {
  SDSGE_DIST_NORMAL = 1,
  SDSGE_DIST_LOG_NORMAL = 2,
  SDSGE_DIST_HALF_NORMAL = 3,
  SDSGE_DIST_TRUNC_NORMAL = 4,
  SDSGE_DIST_HALF_CAUCHY = 5,
  SDSGE_DIST_BETA = 6,
  SDSGE_DIST_GAMMA = 7,
  SDSGE_DIST_INV_GAMMA = 8,
  SDSGE_DIST_UNIFORM = 9
} SdsgeDistCode;

typedef enum {
  SDSGE_TRANSFORM_IDENTITY = 1,
  SDSGE_TRANSFORM_LOG = 2,
  SDSGE_TRANSFORM_SOFTPLUS = 3,
  SDSGE_TRANSFORM_LOGIT = 4,
  SDSGE_TRANSFORM_PROBIT = 5,
  SDSGE_TRANSFORM_AFFINE_LOGIT = 6,
  SDSGE_TRANSFORM_AFFINE_PROBIT = 7,
  SDSGE_TRANSFORM_LOWER_BOUNDED = 8,
  SDSGE_TRANSFORM_UPPER_BOUNDED = 9
} SdsgeTransformCode;

/* Packed-row strides (mirror N_DIST_PARAMS / N_TRANSFORM_PARAMS). */
#define SDSGE_N_DIST_PARAMS 5
#define SDSGE_N_TRANSFORM_PARAMS 3

/* Scalar helpers (exposed so the parity tests can hit them directly). */
f64 sdsge_softplus_scalar(f64 x);
f64 sdsge_log_sigmoid_scalar(f64 x);
f64 sdsge_sigmoid_scalar(f64 x);
f64 sdsge_std_norm_cdf(f64 x);
f64 sdsge_std_norm_logpdf(f64 x);

/* Leaf dispatchers. `code` is an i64 (the value arrives from an int64 array);
 * it is matched against the Sdsge*Code enum constants. Out-of-support / unknown
 * codes write NaN, the agreed "fall back to the Python path" sentinel. */
void sdsge_transform_inverse_and_logjac(i64 code, f64 *SDSGE_RESTRICT params,
                                        f64 z, f64 *SDSGE_RESTRICT out_x,
                                        f64 *SDSGE_RESTRICT out_logjac);
void sdsge_dist_logpdf(i64 code, f64 *SDSGE_RESTRICT params, f64 x,
                       f64 *SDSGE_RESTRICT out_logpdf);

/* LKJ-Cholesky log-jacobian (out-param and returning variants) and log-density
 * over the `len` packed lower-triangle entries of a `dim`x`dim` block. */
void sdsge_lkj_chol_logjac(f64 *SDSGE_RESTRICT z, i64 dim, i64 len,
                           f64 *SDSGE_RESTRICT out_logjac);
f64 sdsge_lkj_chol_logjac_return(f64 *SDSGE_RESTRICT z, i64 dim, i64 len);
void sdsge_lkj_chol_logpdf_from_z(f64 *SDSGE_RESTRICT z, i64 dim, i64 len,
                                  f64 eta, f64 log_const,
                                  f64 *SDSGE_RESTRICT out_logpdf);

/* Full packed log-prior (per-replication hot path). scalar_*_params are
 * row-major n_scalar x stride; matrix_indices is row-major n_blocks x
 * max_matrix_len. Returns the scalar logprior, or NaN if any term is NaN. */
f64 sdsge_logprior_program(
    f64 *SDSGE_RESTRICT theta, i64 *SDSGE_RESTRICT scalar_indices,
    i64 *SDSGE_RESTRICT scalar_dist_codes,
    i64 *SDSGE_RESTRICT scalar_transform_codes,
    f64 *SDSGE_RESTRICT scalar_dist_params,
    f64 *SDSGE_RESTRICT scalar_transform_params, i64 n_scalar,
    i64 *SDSGE_RESTRICT matrix_indices, i64 *SDSGE_RESTRICT matrix_dims,
    i64 *SDSGE_RESTRICT matrix_lengths, f64 *SDSGE_RESTRICT matrix_etas,
    f64 *SDSGE_RESTRICT matrix_log_constants, i64 n_blocks, i64 max_matrix_len);

/* Unconstrained (z, std) -> full covariance via the correlation Cholesky factor.
 * scratch_M is K*K workspace for L; out receives the K*K covariance (row-major). */
void sdsge_cov_from_unconstrained(const f64 *SDSGE_RESTRICT z,
                                  const f64 *SDSGE_RESTRICT std, const i64 K,
                                  f64 *SDSGE_RESTRICT scratch_M,
                                  f64 *SDSGE_RESTRICT out);

/* Inverse of the Cholesky stage: correlation Cholesky factor L (K*K, row-major)
 * -> unconstrained CPC values out_z (length K(K-1)/2), via the stick-breaking
 * remainder and atanh. */
void sdsge_unconstrained_from_corr_chol(const f64 *SDSGE_RESTRICT L, const i64 K,
                                        f64 *SDSGE_RESTRICT out_z);

#endif /* SDSGE_PRIOR_PROGRAM_H */
