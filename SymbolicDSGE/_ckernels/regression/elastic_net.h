#ifndef SDSGE_ELASTIC_NET_H
#define SDSGE_ELASTIC_NET_H

#include "../_common/sdsge_common.h"
#include "regression.h" /* RegressionStatus codes + sdsge_smooth_threshold */

/* Elastic-net kernels, mirroring the numba helpers in
 * SymbolicDSGE/regression/elastic_net. The numba reference is plain @njit (NOT
 * fastmath), so the strict-IEEE C is bit-parity up to libm. */

/* Coordinate descent with an L2-augmented diagonal and L1 soft-threshold,
 * warm-started from beta0(k). coef(k) is the output, Gcoef(k) caller scratch.
 * Returns REGRESSION_OK / REGRESSION_NON_CONVERGENT. */
i64 sdsge_en_gram_cd(const f64 *G, const f64 *g, i64 k, f64 alpha_l1, f64 alpha_l2,
                     const f64 *beta0, i64 max_iter, f64 tol, f64 *coef,
                     f64 *Gcoef);

/* Warm-start path over alpha_grid(n_alpha): solves descending in alpha, reusing
 * the previous solution as the warm start. coefs(n_alpha*k) and statuses(n_alpha)
 * are outputs; Gcoef(k) and beta(k) are caller scratch. */
void sdsge_en_gram_cd_path(const f64 *G, const f64 *g, i64 k,
                           const f64 *alpha_grid, i64 n_alpha, f64 l1_ratio,
                           i64 max_iter, f64 tol, f64 *coefs, i64 *statuses,
                           f64 *Gcoef, f64 *beta);

/* Effective dof on the active set: trace(penalized^-1 @ G_active) + intercept,
 * where penalized = G_active + alpha_l2*I over the support of beta. Active-set
 * scratch is malloc'd. */
f64 sdsge_en_active_dof(const f64 *G, const f64 *beta, i64 k, f64 alpha_l2,
                        i64 intercept, f64 atol);

#endif /* SDSGE_ELASTIC_NET_H */
