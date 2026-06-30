#include "elastic_net.h"
#include <math.h>
#include <stdlib.h>

/* alpha -> (l1, l2) penalty split (mirrors numba split_penalty). */
static inline void sdsge_split_penalty(f64 alpha, f64 l1_ratio, f64 *a1, f64 *a2) {
  *a1 = alpha * l1_ratio;
  *a2 = alpha * (1.0 - l1_ratio);
}

i64 sdsge_en_gram_cd(const f64 *SDSGE_RESTRICT G, const f64 *SDSGE_RESTRICT g,
                     i64 k, f64 alpha_l1, f64 alpha_l2,
                     const f64 *SDSGE_RESTRICT beta0, i64 max_iter, f64 tol,
                     f64 *SDSGE_RESTRICT coef, f64 *SDSGE_RESTRICT Gcoef) {
  for (i64 i = 0; i < k; ++i)
    coef[i] = beta0[i];

  /* Gcoef = (G + alpha_l2 I) @ coef, maintained incrementally below. */
  for (i64 i = 0; i < k; ++i) {
    f64 acc = 0.0;
    for (i64 j = 0; j < k; ++j)
      acc += G[i * k + j] * coef[j];
    Gcoef[i] = acc + alpha_l2 * coef[i];
  }

  for (i64 it = 0; it < max_iter; ++it) {
    f64 max_delta = 0.0;
    for (i64 j = 0; j < k; ++j) {
      const f64 diag = G[j * k + j] + alpha_l2;
      if (diag <= 0.0)
        continue;
      const f64 z = g[j] - Gcoef[j] + diag * coef[j];
      const f64 new_coef_j = sdsge_smooth_threshold(z, alpha_l1) / diag;
      const f64 delta = new_coef_j - coef[j];
      const f64 abs_delta = fabs(delta);
      if (abs_delta > tol) {
        for (i64 i = 0; i < k; ++i)
          Gcoef[i] += G[i * k + j] * delta;
        Gcoef[j] += alpha_l2 * delta;
        coef[j] = new_coef_j;
        if (abs_delta > max_delta)
          max_delta = abs_delta;
      }
    }
    if (max_delta < tol)
      return REGRESSION_OK;
  }
  return REGRESSION_NON_CONVERGENT;
}

void sdsge_en_gram_cd_path(const f64 *SDSGE_RESTRICT G, const f64 *SDSGE_RESTRICT g,
                           i64 k, const f64 *SDSGE_RESTRICT alpha_grid, i64 n_alpha,
                           f64 l1_ratio, i64 max_iter, f64 tol,
                           f64 *SDSGE_RESTRICT coefs, i64 *SDSGE_RESTRICT statuses,
                           f64 *SDSGE_RESTRICT Gcoef, f64 *SDSGE_RESTRICT beta) {
  for (i64 i = 0; i < k; ++i)
    beta[i] = 0.0;

  /* numba walks the grid from large alpha to small for a better warm start: if
   * the grid is ascending, iterate it in reverse; otherwise iterate as given. */
  const i64 ascending = (n_alpha > 1 && alpha_grid[0] < alpha_grid[n_alpha - 1]);

  for (i64 pos = 0; pos < n_alpha; ++pos) {
    const i64 idx = ascending ? (n_alpha - pos - 1) : pos;
    f64 a1, a2;
    sdsge_split_penalty(alpha_grid[idx], l1_ratio, &a1, &a2);
    f64 *coef_idx = coefs + idx * k;
    statuses[idx] =
        sdsge_en_gram_cd(G, g, k, a1, a2, beta, max_iter, tol, coef_idx, Gcoef);
    for (i64 i = 0; i < k; ++i)
      beta[i] = coef_idx[i]; /* warm start for the next alpha */
  }
}

f64 sdsge_en_active_dof(const f64 *SDSGE_RESTRICT G, const f64 *SDSGE_RESTRICT beta,
                        i64 k, f64 alpha_l2, i64 intercept, f64 atol) {
  const f64 ic = (f64)(intercept ? 1 : 0);

  i64 na = 0;
  for (i64 j = 0; j < k; ++j)
    if (fabs(beta[j]) > atol)
      na += 1;
  if (na == 0)
    return ic;

  i64 *active = (i64 *)malloc((size_t)na * sizeof(i64));
  f64 *G_active = (f64 *)malloc((size_t)(na * na) * sizeof(f64));
  f64 *pen = (f64 *)malloc((size_t)(na * na) * sizeof(f64));
  f64 *L = (f64 *)malloc((size_t)(na * na) * sizeof(f64));
  f64 *col = (f64 *)malloc((size_t)na * sizeof(f64));
  if (active == NULL || G_active == NULL || pen == NULL || L == NULL ||
      col == NULL) {
    free(active);
    free(G_active);
    free(pen);
    free(L);
    free(col);
    return (f64)na + ic; /* numba's except branch returns n_active */
  }

  i64 cur = 0;
  for (i64 j = 0; j < k; ++j)
    if (fabs(beta[j]) > atol)
      active[cur++] = j;

  for (i64 i = 0; i < na; ++i) {
    const i64 row = active[i];
    for (i64 j = 0; j < na; ++j) {
      const f64 val = G[row * k + active[j]];
      G_active[i * na + j] = val;
      pen[i * na + j] = val;
    }
    pen[i * na + i] += alpha_l2;
  }

  f64 dof;
  if (sdsge_chol(pen, 0.0, L, na) != SDSGE_OK) {
    dof = (f64)na; /* singular penalized Gram -> numba's except fallback */
  } else {
    /* trace(penalized^-1 @ G_active) via per-column solves reusing L. */
    dof = 0.0;
    for (i64 c = 0; c < na; ++c) {
      for (i64 r = 0; r < na; ++r)
        col[r] = G_active[r * na + c];
      sdsge_forward_subst(L, col, col, na);
      sdsge_backward_subst_chol_t(L, col, col, na);
      dof += col[c];
    }
  }

  free(active);
  free(G_active);
  free(pen);
  free(L);
  free(col);
  return dof + ic;
}
