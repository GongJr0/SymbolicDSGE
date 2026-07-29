#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"

#ifndef SDSGE_DIAG_WALD_H
#define SDSGE_DIAG_WALD_H

// Kernel IDs
typedef enum { BARTLETT = 0, PARZEN = 1, QS = 2, KERNEL_COUNT = 3 } KernelID;
typedef enum {
  WALD_BW_MANUAL = 0,
  WALD_BW_WOOLDRIDGE = 1,
  WALD_BW_ANDREWS = 2,
  WALD_BW_AUTO = 3,
} WaldBandwidthMode;

// Kernel Constants

#define C_BARTLETT 1.1447
#define C_PARZEN 2.6614
#define C_QS 1.3221

// ID to kernel struct
typedef struct {
  f64 c;
  f64 q;
} kernel_inp_t;

static const kernel_inp_t KERNEL_SPECS[KERNEL_COUNT] = {
    [BARTLETT] = {.c = C_BARTLETT, .q = 1.0},
    [PARZEN] = {.c = C_PARZEN, .q = 2.0},
    [QS] = {.c = C_QS, .q = 2.0}};

/* Column means of x(n,p) over axis 0; writes mean(p). */
void sdsge_fill_mean_ax0(const f64 *SDSGE_RESTRICT x, const i64 n, const i64 p,
                         f64 *SDSGE_RESTRICT mean);

/* x(n,p) with its column means subtracted; writes centered(n,p). */
void sdsge_fill_centered_ax0(const f64 *SDSGE_RESTRICT x,
                             const f64 *SDSGE_RESTRICT mean, const i64 n,
                             const i64 p, f64 *SDSGE_RESTRICT centered);

/* Full HAC long-run covariance: out(p,p) := (Gamma_0 + sum_{j=1..L} w_j
 * (Gamma_j
 * + Gamma_j^T)) / n, mirroring the numba jit_hac_estimator_matmul. `r` is the
 * (n, p) centered moment array; `gamma_scratch` and `out` are caller-owned (p,
 * p) buffers (out must not alias gamma_scratch or r). */
void sdsge_hac_estimator_matmul(f64 *SDSGE_RESTRICT r, KernelID kernel_id,
                                i64 L, i64 n, i64 p,
                                f64 *SDSGE_RESTRICT gamma_scratch,
                                f64 *SDSGE_RESTRICT out);

int sdsge_wald_stat_from_mean_and_cov(const f64 *SDSGE_RESTRICT mean,
                                      const f64 *SDSGE_RESTRICT target,
                                      const f64 *SDSGE_RESTRICT omega,
                                      const i64 n, const i64 p,
                                      f64 *SDSGE_RESTRICT dev_scratch,
                                      f64 *SDSGE_RESTRICT factor_scratch,
                                      i64 *SDSGE_RESTRICT pivot_scratch,
                                      f64 *SDSGE_RESTRICT solved_scratch,
                                      f64 *SDSGE_RESTRICT stat_out);

/* Caller-owned arena layouts:
 * mean: n*q + 3*q*q + 4*q f64, pivot(q) i64.
 * covariance: n*q + n*v + 3*v*v + 5*v f64, pivot(v) i64.
 * second moment: n*v + 3*v*v + 5*v f64, pivot(v) i64,
 * where v = q * (q + 1) / 2.
 *
 * `manual_bandwidth` is used only for WALD_BW_MANUAL. All other modes resolve
 * bandwidth in C, then clamp it to n - 1. Each function returns a diagnostic
 * status and writes the Wald statistic to `stat_out` on success. */
arena_size sdsge_wald_mean_hac_arena_size(i64 n, i64 q);
arena_size sdsge_wald_covariance_hac_arena_size(i64 n, i64 q);
arena_size sdsge_wald_second_moment_hac_arena_size(i64 n, i64 q);

int sdsge_wald_mean_hac(const f64 *SDSGE_RESTRICT g,
                        const f64 *SDSGE_RESTRICT target, i64 n, i64 q,
                        KernelID kernel_id, WaldBandwidthMode bandwidth_mode,
                        i64 manual_bandwidth, f64 *SDSGE_RESTRICT arena,
                        i64 *SDSGE_RESTRICT pivot_scratch,
                        f64 *SDSGE_RESTRICT stat_out);

int sdsge_wald_covariance_hac(const f64 *SDSGE_RESTRICT g,
                              const f64 *SDSGE_RESTRICT target, i64 n, i64 q,
                              KernelID kernel_id,
                              WaldBandwidthMode bandwidth_mode,
                              i64 manual_bandwidth,
                              f64 *SDSGE_RESTRICT arena,
                              i64 *SDSGE_RESTRICT pivot_scratch,
                              f64 *SDSGE_RESTRICT stat_out);

int sdsge_wald_second_moment_hac(
    const f64 *SDSGE_RESTRICT g, const f64 *SDSGE_RESTRICT target, i64 n,
    i64 q, KernelID kernel_id, WaldBandwidthMode bandwidth_mode,
    i64 manual_bandwidth, f64 *SDSGE_RESTRICT arena,
    i64 *SDSGE_RESTRICT pivot_scratch, f64 *SDSGE_RESTRICT stat_out);

int sdsge_symmetric_outer_prod_2dim(const f64 *SDSGE_RESTRICT x, const i64 n,
                                    const i64 p, const i64 q,
                                    f64 *SDSGE_RESTRICT out);

int sdsge_fill_symmetric_target_vec(const f64 *SDSGE_RESTRICT target,
                                    const f64 atol, const f64 rtol, const i64 p,
                                    f64 *SDSGE_RESTRICT out);

#endif // SDSGE_DIAG_WALD_H
