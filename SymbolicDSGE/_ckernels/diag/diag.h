#ifndef SDSGE_DIAG_H
#define SDSGE_DIAG_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"

/* Native diagnostic-test statistic kernels (Breusch-Godfrey, Breusch-Pagan,
 * Chow, Brown-Durbin-Evans recursive residuals). Each mirrors the numba kernel
 * in SymbolicDSGE/_diag_tests/. All matrices are C-contiguous, row-major, f64.
 *
 * Error/status convention: the full-rank fast path (Cholesky on the normal
 * equations) runs entirely here. The negative codes below mirror the Python
 * TestStatus IntEnum exactly, so the Cython shim returns them verbatim.
 * DIAG_FALLBACK is the one extra value: it means "the design is rank-deficient,
 * the Cholesky path can't proceed -- re-run the whole statistic via the numba
 * kernel (which has the SVD-based lstsq fallback)". There is deliberately no
 * lstsq/SVD in C; this keeps the native side LAPACK-free. */

#define DIAG_OK 0
#define DIAG_BAD_SHAPE -1
#define DIAG_LINALG -2
#define DIAG_UDEF_VARIANCE -3
#define DIAG_BAD_LAG -4
#define DIAG_INSUFFICIENT_SAMPLES -5
#define DIAG_ITERATIVE_NONCONVERGENCE -6
#define DIAG_BAD_PARAMETER -7
#define DIAG_FALLBACK 1

/* Breusch-Godfrey LM statistic. eps(n), X(n,K). Builds the auxiliary design
 * [1 | X | lagged eps] internally. Writes the statistic to *stat_out. */
int sdsge_bg_stat(const f64 *SDSGE_RESTRICT eps, const f64 *SDSGE_RESTRICT X,
                  i64 n, i64 K, i64 lags, f64 *SDSGE_RESTRICT stat_out);

/* Breusch-Pagan auxiliary regression. eps(n), X_aug(n,p) already augmented with
 * the intercept column (the augment + constant-column validation stay in the
 * Python wrapper). Writes RSS and centered TSS of the auxiliary fit. */
int sdsge_bp_aux(const f64 *SDSGE_RESTRICT eps, const f64 *SDSGE_RESTRICT X_aug,
                 i64 n, i64 p, f64 *SDSGE_RESTRICT rss_out,
                 f64 *SDSGE_RESTRICT tss_out);

/* Chow break-point F statistic. y(T), X(T,p), split at t_break. Fits the pooled
 * and the two sub-sample regressions; DIAG_FALLBACK if any is rank-deficient.
 */
int sdsge_chow_stat(const f64 *SDSGE_RESTRICT y, const f64 *SDSGE_RESTRICT X,
                    i64 T, i64 p, i64 t_break, f64 *SDSGE_RESTRICT stat_out);

/* Brown-Durbin-Evans recursive residuals (the w series, length T-p). y(T),
 * X(T,p). Fully native: seeds beta/P from the first p rows via an SPD inverse,
 * then the rank-1 downdate recursion. DIAG_FALLBACK if the seed Gram is
 * singular. */
int sdsge_recursive_residuals(const f64 *SDSGE_RESTRICT y,
                              const f64 *SDSGE_RESTRICT X, i64 T, i64 p,
                              f64 *SDSGE_RESTRICT w_out);

/* Standardized CUSUM series (length T-p): cumsum(recursive residuals) / sigma,
 * where sigma is the full-sample OLS residual std. DIAG_FALLBACK if the seed
 * Gram or the full-sample normal equations are rank-deficient. */
int sdsge_cusum_series(const f64 *SDSGE_RESTRICT y, const f64 *SDSGE_RESTRICT X,
                       i64 T, i64 p, f64 *SDSGE_RESTRICT series_out);

/* CUSUM statistic: max over t of |series_t| / (sqrt(T-p) + 2 (t-p) /
 * sqrt(T-p)). */
int sdsge_cusum_stat(const f64 *SDSGE_RESTRICT y, const f64 *SDSGE_RESTRICT X,
                     i64 T, i64 p, f64 *SDSGE_RESTRICT stat_out);

/* CUSUM-of-squares statistic. Writes the residual count n = T-p and the
 * Kolmogorov-type max deviation of the normalized squared-residual partial sums
 * from the t/n line, divided by sqrt(2). */
int sdsge_cusumsq_stat(const f64 *SDSGE_RESTRICT y, const f64 *SDSGE_RESTRICT X,
                       i64 T, i64 p, i64 *SDSGE_RESTRICT n_out,
                       f64 *SDSGE_RESTRICT stat_out);

int sdsge_acorr(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 L,
                f64 *SDSGE_RESTRICT z_scratch, f64 *SDSGE_RESTRICT out);

int sdsge_lb_stat(const f64 *SDSGE_RESTRICT x, const i64 n, i64 L,
                  f64 *SDSGE_RESTRICT z_scratch,
                  f64 *SDSGE_RESTRICT acorr_scratch, f64 *SDSGE_RESTRICT out);

/* Jarque-Bera normality statistic. x(n). Writes the statistic to *out; returns
 * INSUFFICIENT_SAMPLES (still writing the stat) when n < 10, UDEF_VARIANCE when
 * the variance is zero, OK otherwise. */
int sdsge_jb_stat(const f64 *SDSGE_RESTRICT x, i64 n, f64 *SDSGE_RESTRICT out);

#endif /* SDSGE_DIAG_H */
