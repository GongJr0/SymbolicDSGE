#ifndef SDSGE_CORE_H
#define SDSGE_CORE_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include "../_common/sdsge_linalg.h"

/* Measurement / observable-jacobian @cfunc ABI: ``void(vars*, par*, out*)``. */
typedef void (*sdsge_measurement_fn)(f64 *vars, f64 *par, f64 *out);

/* Assemble the first-order transition from the solved rule.
 *
 * A = [[p,   0],
 *      [f@p, 0]]
 *
 * The shock loading is not assembled here. It is one solve over every variable,
 * not a state block with the controls derived from it: an innovation reaches a
 * control contemporaneously through whatever equation carries it, which no
 * product of `f` with a state loading can express. The pencil stage emits it
 * whole beside `p` and `f`. */
void sdsge_assemble_transition(const f64 *SDSGE_RESTRICT p,
                               const f64 *SDSGE_RESTRICT f, const i64 n_state,
                               const i64 n_control, f64 *SDSGE_RESTRICT A);

/* Linear state-space simulation kernels */

/* out[(T+1, n)] : out[0] = x0; out[t+1] = A @ out[t] + B @ shock[t]. */
void sdsge_simulate_linear_states(const f64 *SDSGE_RESTRICT A,     /* (n, n) */
                                  const f64 *SDSGE_RESTRICT B,     /* (n, k) */
                                  const f64 *SDSGE_RESTRICT x0,    /* (n,)   */
                                  const f64 *SDSGE_RESTRICT shock, /* (T, k) */
                                  f64 *SDSGE_RESTRICT out, /* (T+1, n) */
                                  i64 T, i64 n, i64 k);

/* out[(T, m)] : out[t] = d + C @ states[t]. */
void sdsge_affine_observations(const f64 *SDSGE_RESTRICT
                                   states, /* (T, n) row-major */
                               const f64 *SDSGE_RESTRICT C, /* (m, n) */
                               const f64 *SDSGE_RESTRICT d, /* (m,)   */
                               f64 *SDSGE_RESTRICT out,     /* (T, m) */
                               i64 T, i64 m, i64 n);

/* Pruned second order simulation.
 * x_out[(T+1, nx)] and y_out[(T+1, ny)] are dense split outputs. */
i64 sdsge_simulate_second_order_pruned(
    const f64 *SDSGE_RESTRICT hx,  /* (nx, nx) */
    const f64 *SDSGE_RESTRICT gx,  /* (ny, nx), nullable when ny == 0 */
    const f64 *SDSGE_RESTRICT bx,  /* (nx, n_exog), nullable when n_exog == 0 */
    const f64 *SDSGE_RESTRICT hxx, /* (nx, nx, nx) */
    const f64 *SDSGE_RESTRICT gxx, /* (ny, nx, nx), nullable when ny == 0 */
    const f64 *SDSGE_RESTRICT hss, /* (nx,) */
    const f64 *SDSGE_RESTRICT gss, /* (ny,), nullable when ny == 0 */
    const f64 *SDSGE_RESTRICT x0,  /* (nx,) */
    const f64 *SDSGE_RESTRICT shock, /* (T, n_exog), nullable when empty */
    i64 T, i64 nx, i64 ny, i64 n_exog,
    f64 *SDSGE_RESTRICT out /* (T+1, nx + ny) */
);
/* ERROR CODES */
#define SDSGE_CORE_SUCCESS 0
#define SDSGE_CORE_ALLOC_FAIL -201

#endif /* SDSGE_CORE_H */
