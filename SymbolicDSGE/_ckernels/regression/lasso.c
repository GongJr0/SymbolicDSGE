#include "lasso.h"
#include <math.h>
#include <stdlib.h>

/* Soft-thresholding operator (mirrors numba smooth_threshold). */
static inline f64 sdsge_smooth_threshold(f64 z, f64 gamma) {
  if (z > gamma)
    return z - gamma;
  if (z < -gamma)
    return z + gamma;
  return 0.0;
}

i64 sdsge_lasso_gram_cd(const f64 *SDSGE_RESTRICT G, const f64 *SDSGE_RESTRICT g,
                        i64 k, f64 alpha, i64 max_iter, f64 tol,
                        f64 *SDSGE_RESTRICT coef, f64 *SDSGE_RESTRICT Gcoef) {
  for (i64 i = 0; i < k; ++i) {
    coef[i] = 0.0;
    Gcoef[i] = 0.0;
  }

  for (i64 it = 0; it < max_iter; ++it) {
    f64 max_delta = 0.0;
    for (i64 j = 0; j < k; ++j) {
      const f64 Gjj = G[j * k + j];
      if (Gjj <= 0.0)
        continue;
      const f64 z = g[j] - Gcoef[j] + Gjj * coef[j];
      const f64 new_coef_j = sdsge_smooth_threshold(z, alpha) / Gjj;
      const f64 delta = new_coef_j - coef[j];
      /* numba: not np.isclose(delta, 0, atol=tol) -> |delta| > tol. */
      if (fabs(delta) > tol) {
        for (i64 i = 0; i < k; ++i)
          Gcoef[i] += G[i * k + j] * delta;
        coef[j] = new_coef_j;
        if (fabs(delta) > max_delta)
          max_delta = fabs(delta);
      }
    }
    if (max_delta < tol)
      return REGRESSION_OK;
  }
  return REGRESSION_NON_CONVERGENT;
}

/* In-place Gaussian elimination with partial pivoting for a tiny square system
 * (mirrors numba solve_small). DESTROYS A and b; writes the solution into x. */
static void sdsge_solve_small(f64 *A, f64 *b, i64 n, f64 *x) {
  for (i64 col = 0; col < n; ++col) {
    i64 max_row = col;
    for (i64 row = col + 1; row < n; ++row)
      if (fabs(A[row * n + col]) > fabs(A[max_row * n + col]))
        max_row = row;
    if (max_row != col) {
      for (i64 c = 0; c < n; ++c) {
        const f64 t = A[col * n + c];
        A[col * n + c] = A[max_row * n + c];
        A[max_row * n + c] = t;
      }
      const f64 tb = b[col];
      b[col] = b[max_row];
      b[max_row] = tb;
    }
    const f64 piv = A[col * n + col];
    for (i64 row = col + 1; row < n; ++row) {
      const f64 f = A[row * n + col] / piv;
      b[row] -= f * b[col];
      for (i64 c = col; c < n; ++c)
        A[row * n + c] -= f * A[col * n + c];
    }
  }
  for (i64 i = n - 1; i >= 0; --i) {
    x[i] = b[i];
    for (i64 j = i + 1; j < n; ++j)
      x[i] -= A[i * n + j] * x[j];
    x[i] /= A[i * n + i];
  }
}

i64 sdsge_lars_lasso_gram(const f64 *SDSGE_RESTRICT G, const f64 *SDSGE_RESTRICT c,
                          i64 k, i64 max_iter, f64 tol,
                          f64 *SDSGE_RESTRICT lam_path,
                          f64 *SDSGE_RESTRICT beta_path,
                          i64 *SDSGE_RESTRICT n_knots) {
  /* One f64 block (7 vectors of k + the k*k active Gram) and one i64 block
   * (active flags + active indices). */
  f64 *fbuf = (f64 *)malloc((size_t)(7 * k + k * k) * sizeof(f64));
  i64 *ibuf = (i64 *)malloc((size_t)(2 * k) * sizeof(i64));
  if (fbuf == NULL || ibuf == NULL) {
    free(fbuf);
    free(ibuf);
    *n_knots = 0;
    return REGRESSION_NON_CONVERGENT;
  }
  f64 *beta = fbuf;
  f64 *signs = fbuf + k;
  f64 *r = fbuf + 2 * k;
  f64 *s_A = fbuf + 3 * k;
  f64 *w = fbuf + 4 * k;
  f64 *d = fbuf + 5 * k;
  f64 *Gd = fbuf + 6 * k;
  f64 *G_AA = fbuf + 7 * k;
  i64 *active = ibuf;
  i64 *act_idx = ibuf + k;

  for (i64 j = 0; j < k; ++j) {
    beta[j] = 0.0;
    signs[j] = 0.0;
    active[j] = 0;
    r[j] = c[j]; /* residual correlations r = c - G@beta, beta == 0 */
  }
  i64 n_active = 0;
  i64 drop = 0;

  f64 lam = 0.0;
  for (i64 j = 0; j < k; ++j)
    if (fabs(r[j]) > lam)
      lam = fabs(r[j]);

  lam_path[0] = lam;
  for (i64 j = 0; j < k; ++j)
    beta_path[j] = 0.0;
  i64 knot = 1;
  i64 status = REGRESSION_NON_CONVERGENT;

  if (lam <= tol) {
    status = REGRESSION_OK;
    goto done;
  }

  for (i64 iter = 0; iter < max_iter; ++iter) {
    if (!drop) {
      /* Add the inactive variable with maximal absolute correlation. */
      i64 new_var = -1;
      f64 new_lam = -1.0;
      for (i64 j = 0; j < k; ++j) {
        if (!active[j]) {
          const f64 abs_r = fabs(r[j]);
          if (abs_r > new_lam) {
            new_lam = abs_r;
            new_var = j;
          }
        }
      }
      if (new_var >= 0) {
        lam = new_lam;
        active[new_var] = 1;
        signs[new_var] = (r[new_var] > 0.0) ? 1.0 : -1.0;
        n_active += 1;
      } else if (n_active == 0) {
        status = REGRESSION_OK;
        goto done;
      }
    }
    drop = 0;

    /* Collect active indices and build (G_AA, s_A). */
    i64 cnt = 0;
    for (i64 j = 0; j < k; ++j)
      if (active[j])
        act_idx[cnt++] = j;
    for (i64 a = 0; a < n_active; ++a) {
      s_A[a] = signs[act_idx[a]];
      for (i64 b = 0; b < n_active; ++b)
        G_AA[a * n_active + b] = G[act_idx[a] * k + act_idx[b]];
    }

    /* Equiangular direction: solve G_AA w = s_A (s_A is destroyed). */
    sdsge_solve_small(G_AA, s_A, n_active, w);
    f64 sw = 0.0;
    for (i64 a = 0; a < n_active; ++a)
      sw += signs[act_idx[a]] * w[a];
    if (sw <= 0.0 || !isfinite(sw)) {
      status = REGRESSION_NON_CONVERGENT;
      goto done;
    }
    const f64 A_scalar = 1.0 / sqrt(sw);

    for (i64 j = 0; j < k; ++j)
      d[j] = 0.0;
    for (i64 a = 0; a < n_active; ++a)
      d[act_idx[a]] = A_scalar * w[a];

    for (i64 j = 0; j < k; ++j) {
      Gd[j] = 0.0;
      for (i64 a = 0; a < n_active; ++a)
        Gd[j] += G[j * k + act_idx[a]] * d[act_idx[a]];
    }

    f64 step = lam / A_scalar;
    i64 drop_var = -1;

    for (i64 j = 0; j < k; ++j) {
      if (!active[j]) {
        const f64 denom1 = A_scalar - Gd[j];
        const f64 denom2 = A_scalar + Gd[j];
        if (denom1 > 1e-14) {
          const f64 t = (lam - r[j]) / denom1;
          if (0.0 < t && t < step)
            step = t;
        }
        if (denom2 > 1e-14) {
          const f64 t = (lam + r[j]) / denom2;
          if (0.0 < t && t < step)
            step = t;
        }
      }
    }

    /* Lasso drop: an active coefficient crosses zero. */
    for (i64 a = 0; a < n_active; ++a) {
      const i64 j = act_idx[a];
      if (d[j] != 0.0) {
        const f64 t = -beta[j] / d[j];
        if (0.0 < t && t < step) {
          step = t;
          drop_var = j;
        }
      }
    }

    for (i64 j = 0; j < k; ++j) {
      beta[j] += step * d[j];
      r[j] -= step * Gd[j];
    }
    lam -= step * A_scalar;
    if (lam < 0.0 && lam > -tol)
      lam = 0.0;

    if (drop_var >= 0) {
      active[drop_var] = 0;
      beta[drop_var] = 0.0;
      n_active -= 1;
      drop = 1;
    }

    lam_path[knot] = lam;
    for (i64 j = 0; j < k; ++j)
      beta_path[knot * k + j] = beta[j];
    knot += 1;

    if (lam <= tol) {
      status = REGRESSION_OK;
      goto done;
    }
  }

done:
  *n_knots = knot;
  free(fbuf);
  free(ibuf);
  return status;
}

void sdsge_lasso_path_eval(const f64 *SDSGE_RESTRICT lam_path,
                           const f64 *SDSGE_RESTRICT beta_path, i64 n_knots,
                           i64 k, const f64 *SDSGE_RESTRICT lam_grid, i64 n_grid,
                           f64 *SDSGE_RESTRICT out) {
  for (i64 gi = 0; gi < n_grid; ++gi) {
    for (i64 j = 0; j < k; ++j)
      out[gi * k + j] = 0.0;

    const f64 lam = lam_grid[gi];
    if (lam >= lam_path[0]) /* above the first knot -> all zero */
      continue;
    if (lam <= lam_path[n_knots - 1]) { /* below the last knot -> last beta */
      for (i64 j = 0; j < k; ++j)
        out[gi * k + j] = beta_path[(n_knots - 1) * k + j];
      continue;
    }

    i64 lo = 0;
    i64 hi = n_knots - 1;
    while (hi - lo > 1) {
      const i64 mid = (lo + hi) / 2;
      if (lam_path[mid] >= lam)
        lo = mid;
      else
        hi = mid;
    }
    const f64 t = (lam - lam_path[lo]) / (lam_path[hi] - lam_path[lo]);
    for (i64 j = 0; j < k; ++j)
      out[gi * k + j] =
          beta_path[lo * k + j] + t * (beta_path[hi * k + j] - beta_path[lo * k + j]);
  }
}
