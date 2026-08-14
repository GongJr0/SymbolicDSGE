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

/* out[(T, n)] : out[t] = ss + A @ (t ? dev[t-1] : x0) + B @ shock[t].
 *
 * The initial condition is not a row: a caller that wants one writes it and
 * passes out + n.
 *
 * `ss` selects what the rows are denominated in, and is the only such switch:
 * NULL leaves them deviations, a pointer makes them levels. The recursion
 * itself always runs in deviations, off `arena` rather than off `out`, because
 * a level row fed back through A is only correct when the steady state happens
 * to be a fixed point of it. */
arena_size sdsge_simulate_linear_states_arena_size(i64 n);

void sdsge_simulate_linear_states(const f64 *SDSGE_RESTRICT A,     /* (n, n) */
                                  const f64 *SDSGE_RESTRICT B,     /* (n, k) */
                                  const f64 *SDSGE_RESTRICT x0,    /* (n,)   */
                                  const f64 *SDSGE_RESTRICT shock, /* (T, k) */
                                  const f64 *SDSGE_RESTRICT ss, /* (n,), nullable */
                                  f64 *SDSGE_RESTRICT out,      /* (T, n) */
                                  f64 *SDSGE_RESTRICT arena, i64 T, i64 n,
                                  i64 k);

/* out[(T, m)] : out[t] = d + C @ states[t]. */
void sdsge_affine_observations(const f64 *SDSGE_RESTRICT
                                   states, /* (T, n) row-major */
                               const f64 *SDSGE_RESTRICT C, /* (m, n) */
                               const f64 *SDSGE_RESTRICT d, /* (m,)   */
                               f64 *SDSGE_RESTRICT out,     /* (T, m) */
                               i64 T, i64 m, i64 n);

/* Pruned second order simulation (Kim-Kim-Schaumburg, as Dynare's simult_.m
 * writes it at order 2).
 *
 * Every row of a period is one expression in the previous state and this
 * period's innovation: `out[t]` is the whole variable vector, states first.
 * The controls are not derived from the updated state -- an innovation reaches
 * a control contemporaneously, which is why `bu` spans every variable and not
 * just the state block, exactly as at first order.
 *
 * The quadratic terms are evaluated at the first-order state alone; that, and
 * nothing else, is the pruning. `ghxu` carries no 1/2: the cross pair is
 * counted once.
 *
 * `ss` denominates the rows exactly as at first order: NULL leaves them
 * deviations, a pointer makes them levels. The pruned state the recursion
 * carries is always a deviation. */
arena_size sdsge_simulate_second_order_pruned_arena_size(i64 nx, i64 n_exog);

/* Total: the recursion has no failure mode once its scratch comes from the
 * caller. */
void sdsge_simulate_second_order_pruned(
    const f64 *SDSGE_RESTRICT hx,  /* (nx, nx) */
    const f64 *SDSGE_RESTRICT gx,  /* (ny, nx), nullable when ny == 0 */
    const f64 *SDSGE_RESTRICT bu,  /* (nx+ny, n_exog), nullable when n_exog==0 */
    const f64 *SDSGE_RESTRICT hxx, /* (nx, nx, nx) */
    const f64 *SDSGE_RESTRICT gxx, /* (ny, nx, nx), nullable when ny == 0 */
    const f64 *SDSGE_RESTRICT hxu, /* (nx, nx, n_exog), nullable when n_exog==0 */
    const f64 *SDSGE_RESTRICT gxu, /* (ny, nx, n_exog), nullable as gx or bu */
    const f64 *SDSGE_RESTRICT huu, /* (nx, n_exog, n_exog), nullable as bu */
    const f64 *SDSGE_RESTRICT guu, /* (ny, n_exog, n_exog), nullable as gx or bu */
    const f64 *SDSGE_RESTRICT hss, /* (nx,) */
    const f64 *SDSGE_RESTRICT gss, /* (ny,), nullable when ny == 0 */
    const f64 *SDSGE_RESTRICT x0,    /* (nx,) */
    const f64 *SDSGE_RESTRICT shock, /* (T, n_exog), nullable when empty */
    const f64 *SDSGE_RESTRICT ss,    /* (nx+ny,), nullable */
    f64 *SDSGE_RESTRICT out,         /* (T, nx + ny) */
    f64 *SDSGE_RESTRICT arena, i64 T, i64 nx, i64 ny, i64 n_exog);

#endif /* SDSGE_CORE_H */
