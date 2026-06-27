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

void hac_estimator_matmul(f64 *SDSGE_RESTRICT r, KernelID kernel_id, i64 L,
                          i64 n, i64 p, f64 *SDSGE_RESTRICT gamma_scratch,
                          f64 *SDSGE_RESTRICT out);

#endif // SDSGE_DIAG_WALD_H
