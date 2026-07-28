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

void sdsge_simulate_order2_step(
    const f64 *SDSGE_RESTRICT hx, const f64 *SDSGE_RESTRICT gx,
    const f64 *SDSGE_RESTRICT bx, const f64 *SDSGE_RESTRICT hxx,
    const f64 *SDSGE_RESTRICT gxx, const f64 *SDSGE_RESTRICT hss,
    const f64 *SDSGE_RESTRICT gss, const f64 *SDSGE_RESTRICT steady_state,
    const f64 *SDSGE_RESTRICT x0, const f64 *SDSGE_RESTRICT shock,
    sdsge_measurement_fn measurement, f64 *SDSGE_RESTRICT params, i64 T, i64 nx,
    i64 ny, i64 n_exog, i64 m, f64 *SDSGE_RESTRICT simout,
    f64 *SDSGE_RESTRICT scratch);

#endif /* SDSGE_MC_CORE_STEPS */
