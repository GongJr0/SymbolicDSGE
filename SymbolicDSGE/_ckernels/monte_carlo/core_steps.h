#ifndef SDSGE_MC_CORE_STEPS
#define SDSGE_MC_CORE_STEPS

#include "../_common/sdsge_common.h"
#include "../core/core.h"
#include "../kalman/kalman.h"

void sdsge_simulate_order1_step(
    const f64 *SDSGE_RESTRICT A, const f64 *SDSGE_RESTRICT B,
    const f64 *SDSGE_RESTRICT C, const f64 *SDSGE_RESTRICT d,
    const f64 *SDSGE_RESTRICT x0, const f64 *SDSGE_RESTRICT shock, const i64 T,
    const i64 n, const i64 k, const i64 m, f64 *SDSGE_RESTRICT simout);

#endif /* SDSGE_MC_CORE_STEPS */
