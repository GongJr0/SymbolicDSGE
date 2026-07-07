#ifndef SDSGE_CORE_H
#define SDSGE_CORE_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"

/* Linear state-space simulation kernels (pure C; no Python).
 * All arrays are C-contiguous, row-major, f64. Mirrors the numba kernels in
 * SymbolicDSGE/core/simulation.py. */

/* out[(T+1, n)] : out[0] = x0; out[t+1] = A @ out[t] + B @ shock[t]. */
void sdsge_simulate_linear_states(const f64 *SDSGE_RESTRICT A,     /* (n, n) */
                                  const f64 *SDSGE_RESTRICT B,     /* (n, k) */
                                  const f64 *SDSGE_RESTRICT x0,    /* (n,)   */
                                  const f64 *SDSGE_RESTRICT shock, /* (T, k) */
                                  f64 *SDSGE_RESTRICT out, /* (T+1, n) */
                                  i64 T, i64 n, i64 k);

/* out[(T, m)] : out[t] = d + C @ states[state_start + t]. */
void sdsge_affine_observations(
    const f64 *SDSGE_RESTRICT states, /* (>= state_start + T, n) row-major */
    const f64 *SDSGE_RESTRICT C,      /* (m, n) */
    const f64 *SDSGE_RESTRICT d,      /* (m,)   */
    i64 state_start, f64 *SDSGE_RESTRICT out, /* (T, m) */
    i64 T, i64 m, i64 n);

/* Pruned second order simulation.
 * x_out[(T+1, nx)] and y_out[(T+1, ny)] are dense split outputs. */
i64 sdsge_simulate_second_order_pruned(
    const f64 *SDSGE_RESTRICT hx,    /* (nx, nx) */
    const f64 *SDSGE_RESTRICT gx,    /* (ny, nx), nullable when ny == 0 */
    const f64 *SDSGE_RESTRICT bx,    /* (nx, n_exog), nullable when n_exog == 0 */
    const f64 *SDSGE_RESTRICT hxx,   /* (nx, nx, nx) */
    const f64 *SDSGE_RESTRICT gxx,   /* (ny, nx, nx), nullable when ny == 0 */
    const f64 *SDSGE_RESTRICT hss,   /* (nx,) */
    const f64 *SDSGE_RESTRICT gss,   /* (ny,), nullable when ny == 0 */
    const f64 *SDSGE_RESTRICT x0,    /* (nx,) */
    const f64 *SDSGE_RESTRICT shock, /* (T, n_exog), nullable when empty */
    i64 T, i64 nx, i64 ny, i64 n_exog,
    f64 *SDSGE_RESTRICT x_out,       /* (T+1, nx) */
    f64 *SDSGE_RESTRICT y_out);      /* (T+1, ny), nullable when ny == 0 */

/* ERROR CODES */
#define SDSGE_CORE_SUCCESS 0
#define SDSGE_CORE_ALLOC_FAIL -1

#endif /* SDSGE_CORE_H */
