#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"

#ifndef SDSGE_DIAG_WALD_H
#define SDSGE_DIAG_WALD_H

// Kernel IDs
typedef enum { BARTLETT = 0, PARZEN = 1, QS = 2, KERNEL_COUNT = 3 } KernelID;

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
                                      f64 *SDSGE_RESTRICT L_scratch,
                                      f64 *SDSGE_RESTRICT stat_out);

int sdsge_symmetric_outer_prod_2dim(const f64 *SDSGE_RESTRICT x, const i64 n,
                                    const i64 p, const i64 q,
                                    f64 *SDSGE_RESTRICT out);

int sdsge_fill_symmetric_target_vec(const f64 *SDSGE_RESTRICT target,
                                    const f64 atol, const f64 rtol, const i64 p,
                                    f64 *SDSGE_RESTRICT out);

#endif // SDSGE_DIAG_WALD_H
