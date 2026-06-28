#include "diag.h"
#include <math.h>
#include <stdlib.h>

/* Breusch-Godfrey: regress eps on [1 | X | lagged eps], statistic n * R^2 (no
 * intercept removal -- TSS is sum(eps^2), matching the numba kernel). */
int sdsge_bg_stat(const f64 *SDSGE_RESTRICT eps, const f64 *SDSGE_RESTRICT X,
                  i64 n, i64 K, i64 lags, f64 *SDSGE_RESTRICT stat_out) {
  if (n <= lags) {
    *stat_out = NAN;
    return DIAG_INSUFFICIENT_SAMPLES;
  }

  const i64 p = K + lags + 1; /* intercept + regressors + lagged residuals */

  /* arena: design(n*p) + G(p*p) + L(p*p) + g(p) + coef(p) */
  const i64 total = n * p + 2 * p * p + 2 * p;
  f64 *arena = (f64 *)malloc((size_t)total * sizeof(f64));
  if (arena == NULL)
    return DIAG_LINALG;
  f64 *ptr = arena;
  f64 *design = ptr;
  ptr += n * p;
  f64 *G = ptr;
  ptr += p * p;
  f64 *L = ptr;
  ptr += p * p;
  f64 *g = ptr;
  ptr += p;
  f64 *coef = ptr;
  ptr += p;

  /* Build the design row by row: [1, X[r, :], eps[r-1], ..., eps[r-lags]]. */
  for (i64 r = 0; r < n; ++r) {
    f64 *row = design + r * p;
    row[0] = 1.0;
    for (i64 j = 0; j < K; ++j)
      row[1 + j] = X[r * K + j];
    for (i64 lag = 1; lag <= lags; ++lag)
      row[K + lag] = (r >= lag) ? eps[r - lag] : 0.0;
  }

  sdsge_gram(design, G, n, p);
  sdsge_gram_rhs(design, eps, g, n, p);
  if (sdsge_chol_solve(G, g, coef, L, p) != SDSGE_OK) {
    free(arena);
    return DIAG_FALLBACK;
  }

  f64 rss = 0.0, tss = 0.0;
  for (i64 r = 0; r < n; ++r) {
    const f64 *row = design + r * p;
    f64 fit = 0.0;
    for (i64 j = 0; j < p; ++j)
      fit += row[j] * coef[j];
    f64 resid = eps[r] - fit;
    rss += resid * resid;
    tss += eps[r] * eps[r];
  }

  *stat_out = (tss > 0.0) ? (f64)n * (1.0 - rss / tss) : 0.0;
  free(arena);
  return DIAG_OK;
}

/* Breusch-Pagan auxiliary regression of the scaled squared residuals on X_aug.
 * Returns the fit's RSS and centered TSS; the caller shapes them into bp_stat /
 * robust_bp_stat. */
int sdsge_bp_aux(const f64 *SDSGE_RESTRICT eps, const f64 *SDSGE_RESTRICT X_aug,
                 i64 n, i64 p, f64 *SDSGE_RESTRICT rss_out,
                 f64 *SDSGE_RESTRICT tss_out) {
  if (n == 0)
    return DIAG_INSUFFICIENT_SAMPLES;

  /* sigma2 = mean(eps^2); g = eps^2 / sigma2. */
  f64 sum_sq = 0.0;
  for (i64 r = 0; r < n; ++r)
    sum_sq += eps[r] * eps[r];
  f64 sigma2 = sum_sq / (f64)n;
  if (!isfinite(sigma2) || sigma2 <= 0.0)
    return DIAG_UDEF_VARIANCE;

  /* arena: g(n) + G(p*p) + L(p*p) + rhs(p) + coef(p) */
  const i64 total = n + 2 * p * p + 2 * p;
  f64 *arena = (f64 *)malloc((size_t)total * sizeof(f64));
  if (arena == NULL)
    return DIAG_LINALG;
  f64 *ptr = arena;
  f64 *gvec = ptr;
  ptr += n;
  f64 *G = ptr;
  ptr += p * p;
  f64 *L = ptr;
  ptr += p * p;
  f64 *rhs = ptr;
  ptr += p;
  f64 *coef = ptr;
  ptr += p;

  for (i64 r = 0; r < n; ++r)
    gvec[r] = (eps[r] * eps[r]) / sigma2;

  sdsge_gram(X_aug, G, n, p);
  sdsge_gram_rhs(X_aug, gvec, rhs, n, p);
  if (sdsge_chol_solve(G, rhs, coef, L, p) != SDSGE_OK) {
    free(arena);
    return DIAG_FALLBACK;
  }

  f64 g_sum = 0.0;
  for (i64 r = 0; r < n; ++r)
    g_sum += gvec[r];
  f64 g_mean = g_sum / (f64)n;

  f64 rss = 0.0, tss = 0.0;
  for (i64 r = 0; r < n; ++r) {
    const f64 *row = X_aug + r * p;
    f64 fit = 0.0;
    for (i64 j = 0; j < p; ++j)
      fit += row[j] * coef[j];
    f64 resid = gvec[r] - fit;
    rss += resid * resid;
    f64 centered = gvec[r] - g_mean;
    tss += centered * centered;
  }

  *rss_out = rss;
  *tss_out = tss;
  free(arena);
  return DIAG_OK;
}

/* Residual sum of squares of the OLS fit of y_seg(rows) on X_seg(rows, p), via
 * the normal equations. Returns SDSGE_OK / SDSGE_NOT_PD. Scratch G/L/g/coef are
 * caller-provided (all length p / p*p). */
static int chow_segment_rss(const f64 *SDSGE_RESTRICT y_seg,
                            const f64 *SDSGE_RESTRICT X_seg, i64 rows, i64 p,
                            f64 *SDSGE_RESTRICT G, f64 *SDSGE_RESTRICT L,
                            f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT coef,
                            f64 *SDSGE_RESTRICT rss_out) {
  sdsge_gram(X_seg, G, rows, p);
  sdsge_gram_rhs(X_seg, y_seg, g, rows, p);
  int status = sdsge_chol_solve(G, g, coef, L, p);
  if (status != SDSGE_OK)
    return status;

  f64 rss = 0.0;
  for (i64 r = 0; r < rows; ++r) {
    const f64 *row = X_seg + r * p;
    f64 fit = 0.0;
    for (i64 j = 0; j < p; ++j)
      fit += row[j] * coef[j];
    f64 resid = y_seg[r] - fit;
    rss += resid * resid;
  }
  *rss_out = rss;
  return SDSGE_OK;
}

int sdsge_chow_stat(const f64 *SDSGE_RESTRICT y, const f64 *SDSGE_RESTRICT X,
                    i64 T, i64 p, i64 t_break, f64 *SDSGE_RESTRICT stat_out) {
  if (T <= 2 * p) {
    *stat_out = NAN;
    return DIAG_INSUFFICIENT_SAMPLES;
  }
  if (t_break <= 0 || t_break >= T) {
    *stat_out = NAN;
    return DIAG_BAD_PARAMETER;
  }

  /* arena: G(p*p) + L(p*p) + g(p) + coef(p), reused across the three fits. */
  const i64 total = 2 * p * p + 2 * p;
  f64 *arena = (f64 *)malloc((size_t)total * sizeof(f64));
  if (arena == NULL)
    return DIAG_LINALG;
  f64 *ptr = arena;
  f64 *G = ptr;
  ptr += p * p;
  f64 *L = ptr;
  ptr += p * p;
  f64 *g = ptr;
  ptr += p;
  f64 *coef = ptr;
  ptr += p;

  const i64 n1 = t_break;
  const i64 n2 = T - t_break;
  f64 rss_c = 0.0, rss_1 = 0.0, rss_2 = 0.0;

  if (chow_segment_rss(y, X, T, p, G, L, g, coef, &rss_c) != SDSGE_OK ||
      chow_segment_rss(y, X, n1, p, G, L, g, coef, &rss_1) != SDSGE_OK ||
      chow_segment_rss(y + n1, X + n1 * p, n2, p, G, L, g, coef, &rss_2) !=
          SDSGE_OK) {
    free(arena);
    return DIAG_FALLBACK;
  }

  f64 num = (rss_c - (rss_1 + rss_2)) / (f64)p;
  f64 denom = (rss_1 + rss_2) / (f64)(T - 2 * p);
  *stat_out = num / denom;
  free(arena);
  return DIAG_OK;
}

/* Full-sample OLS residual std (sqrt(SSR / (T-p))) via the normal equations.
 * G/L/g/coef are caller scratch (length p / p*p). Returns SDSGE_OK / NOT_PD. */
static int ols_residual_sigma(const f64 *SDSGE_RESTRICT y,
                              const f64 *SDSGE_RESTRICT X, i64 T, i64 p,
                              f64 *SDSGE_RESTRICT G, f64 *SDSGE_RESTRICT L,
                              f64 *SDSGE_RESTRICT g, f64 *SDSGE_RESTRICT coef,
                              f64 *SDSGE_RESTRICT sigma_out) {
  sdsge_gram(X, G, T, p);
  sdsge_gram_rhs(X, y, g, T, p);
  int status = sdsge_chol_solve(G, g, coef, L, p);
  if (status != SDSGE_OK)
    return status;

  f64 ssr = 0.0;
  for (i64 r = 0; r < T; ++r) {
    const f64 *row = X + r * p;
    f64 fit = 0.0;
    for (i64 j = 0; j < p; ++j)
      fit += row[j] * coef[j];
    f64 resid = y[r] - fit;
    ssr += resid * resid;
  }
  *sigma_out = sqrt(ssr / (f64)(T - p));
  return SDSGE_OK;
}

int sdsge_cusum_series(const f64 *SDSGE_RESTRICT y, const f64 *SDSGE_RESTRICT X,
                       i64 T, i64 p, f64 *SDSGE_RESTRICT series_out) {
  if (T == 0 || T <= p)
    return DIAG_INSUFFICIENT_SAMPLES;

  /* arena: w(T-p) + G(p*p) + L(p*p) + g(p) + coef(p) */
  const i64 nrec = T - p;
  const i64 total = nrec + 2 * p * p + 2 * p;
  f64 *arena = (f64 *)malloc((size_t)total * sizeof(f64));
  if (arena == NULL)
    return DIAG_LINALG;
  f64 *ptr = arena;
  f64 *w = ptr;
  ptr += nrec;
  f64 *G = ptr;
  ptr += p * p;
  f64 *L = ptr;
  ptr += p * p;
  f64 *g = ptr;
  ptr += p;
  f64 *coef = ptr;
  ptr += p;

  int status = sdsge_recursive_residuals(y, X, T, p, w);
  if (status != DIAG_OK) {
    free(arena);
    return status; /* INSUFFICIENT_SAMPLES or FALLBACK */
  }

  f64 sigma = 0.0;
  if (ols_residual_sigma(y, X, T, p, G, L, g, coef, &sigma) != SDSGE_OK) {
    free(arena);
    return DIAG_FALLBACK;
  }

  f64 acc = 0.0;
  for (i64 i = 0; i < nrec; ++i) {
    acc += w[i];
    series_out[i] = acc / sigma;
  }
  free(arena);
  return DIAG_OK;
}

int sdsge_cusum_stat(const f64 *SDSGE_RESTRICT y, const f64 *SDSGE_RESTRICT X,
                     i64 T, i64 p, f64 *SDSGE_RESTRICT stat_out) {
  if (T == 0 || T <= p) {
    *stat_out = NAN;
    return DIAG_INSUFFICIENT_SAMPLES;
  }

  const i64 nrec = T - p;
  f64 *series = (f64 *)malloc((size_t)nrec * sizeof(f64));
  if (series == NULL)
    return DIAG_LINALG;

  int status = sdsge_cusum_series(y, X, T, p, series);
  if (status != DIAG_OK) {
    *stat_out = NAN;
    free(series);
    return status;
  }

  const f64 sqrt_Tp = sqrt((f64)nrec);
  f64 best = 0.0;
  for (i64 i = 0; i < nrec; ++i) {
    f64 denom = sqrt_Tp + (2.0 * (f64)i / sqrt_Tp);
    f64 val = fabs(series[i]) / denom;
    if (i == 0 || val > best)
      best = val;
  }
  *stat_out = best;
  free(series);
  return DIAG_OK;
}

int sdsge_cusumsq_stat(const f64 *SDSGE_RESTRICT y, const f64 *SDSGE_RESTRICT X,
                       i64 T, i64 p, i64 *SDSGE_RESTRICT n_out,
                       f64 *SDSGE_RESTRICT stat_out) {
  const i64 nrec = (T > p) ? (T - p) : 0;
  *n_out = nrec;
  if (T == 0 || T <= p) {
    *stat_out = NAN;
    return DIAG_INSUFFICIENT_SAMPLES;
  }

  f64 *w = (f64 *)malloc((size_t)nrec * sizeof(f64));
  if (w == NULL)
    return DIAG_LINALG;

  int status = sdsge_recursive_residuals(y, X, T, p, w);
  if (status != DIAG_OK) {
    *stat_out = NAN;
    free(w);
    return status;
  }

  f64 total_sq = 0.0;
  for (i64 i = 0; i < nrec; ++i)
    total_sq += w[i] * w[i];

  f64 acc = 0.0, best = 0.0;
  for (i64 i = 0; i < nrec; ++i) {
    acc += w[i] * w[i];
    f64 s = acc / total_sq;
    f64 expected = (f64)(i + 1) / (f64)nrec;
    f64 dev = fabs(s - expected);
    if (i == 0 || dev > best)
      best = dev;
  }
  *stat_out = best / sqrt(2.0);
  free(w);
  return DIAG_OK;
}

int sdsge_recursive_residuals(const f64 *SDSGE_RESTRICT y,
                              const f64 *SDSGE_RESTRICT X, i64 T, i64 p,
                              f64 *SDSGE_RESTRICT w_out) {
  if (T == 0 || T <= p)
    return DIAG_INSUFFICIENT_SAMPLES;

  /* arena: G(p*p) + L(p*p) + P(p*p) + Xty(p) + beta(p) + Px(p) */
  const i64 total = 3 * p * p + 3 * p;
  f64 *arena = (f64 *)malloc((size_t)total * sizeof(f64));
  if (arena == NULL)
    return DIAG_LINALG;
  f64 *ptr = arena;
  f64 *G = ptr;
  ptr += p * p;
  f64 *L = ptr;
  ptr += p * p;
  f64 *P = ptr;
  ptr += p * p;
  f64 *Xty = ptr;
  ptr += p;
  f64 *beta = ptr;
  ptr += p;
  f64 *Px = ptr;
  ptr += p;

  /* Seed from the first p rows: P = (X_p' X_p)^-1, beta = P X_p' y_p. */
  sdsge_gram(X, G, p, p);
  sdsge_gram_rhs(X, y, Xty, p, p);
  if (sdsge_chol_inv(G, P, L, p) != SDSGE_OK) {
    free(arena);
    return DIAG_FALLBACK;
  }
  for (i64 a = 0; a < p; ++a) {
    f64 s = 0.0;
    for (i64 b = 0; b < p; ++b)
      s += P[a * p + b] * Xty[b];
    beta[a] = s;
  }

  /* Recursive residuals via the symmetric rank-1 downdate of P. */
  for (i64 i = 0; i < T - p; ++i) {
    const f64 *xt = X + (p + i) * p;

    f64 e = y[p + i];
    for (i64 a = 0; a < p; ++a)
      e -= xt[a] * beta[a];

    f64 quad = 0.0;
    for (i64 a = 0; a < p; ++a) {
      f64 s = 0.0;
      for (i64 b = 0; b < p; ++b)
        s += P[a * p + b] * xt[b];
      Px[a] = s;
      quad += xt[a] * s;
    }

    f64 ft = 1.0 + quad;
    w_out[i] = e / sqrt(ft);

    f64 inv_ft = 1.0 / ft;
    f64 coef = e * inv_ft;
    for (i64 a = 0; a < p; ++a) {
      f64 pa = Px[a];
      beta[a] += pa * coef;
      for (i64 b = 0; b < p; ++b)
        P[a * p + b] -= pa * Px[b] * inv_ft;
    }
  }

  free(arena);
  return DIAG_OK;
}

int sdsge_acorr(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 L,
                f64 *SDSGE_RESTRICT z_scratch, f64 *SDSGE_RESTRICT out) {
  f64 mu = 0.0;
  for (i64 i = 0; i < n; ++i) {
    mu += x[i];
  }
  mu /= (f64)n;

  f64 denom = 0.0;
  for (i64 i = 0; i < n; ++i) {
    z_scratch[i] = x[i] - mu;
    denom += z_scratch[i] * z_scratch[i];
  }

  if (denom <= 0.0 || !isfinite(denom)) {
    for (i64 l = 0; l <= L; ++l)
      out[l] = NAN;
    return DIAG_UDEF_VARIANCE;
  }

  out[0] = 1.0;
  for (i64 ell = 1; ell <= L; ++ell) {
    f64 num = 0.0;
    for (i64 t = ell; t < n; ++t) {
      num += z_scratch[t] * z_scratch[t - ell];
    }
    out[ell] = num / denom;
  }
  return DIAG_OK;
}

int sdsge_lb_stat(const f64 *SDSGE_RESTRICT x, const i64 n, i64 L,
                  f64 *SDSGE_RESTRICT z_scratch,
                  f64 *SDSGE_RESTRICT acorr_scratch, f64 *SDSGE_RESTRICT out) {
  if (n <= 1) {
    *out = NAN;
    return DIAG_INSUFFICIENT_SAMPLES;
  }
  if (L >= n) {
    L = n - 1;
  } else if (L <= 0) {
    *out = NAN;
    return DIAG_BAD_LAG;
  }

  int acorr_err = sdsge_acorr(x, n, L, z_scratch, acorr_scratch);
  if (acorr_err != DIAG_OK) {
    *out = NAN;
    return acorr_err;
  }

  f64 stat = 0.0;
  for (i64 ell = 1; ell <= L; ++ell) {
    stat += (acorr_scratch[ell] * acorr_scratch[ell]) / (f64)(n - ell);
  }
  stat *= (f64)n * (f64)(n + 2);
  *out = stat;
  return DIAG_OK;
}
