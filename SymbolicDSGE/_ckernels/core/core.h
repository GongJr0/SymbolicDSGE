#ifndef SDSGE_CORE_H
#define SDSGE_CORE_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include "../_common/sdsge_linalg.h"

/* Shock jacobian @cfunc ABI: ``void(fwd*, cur*, par*, out*)``. Call it with a
 * scratch buffer for cur, never ss twice: fwd and cur are both restrict. */
typedef void (*shock_jacobian_fn)(const f64 *SDSGE_RESTRICT fwd,
                                  const f64 *SDSGE_RESTRICT cur,
                                  const f64 *SDSGE_RESTRICT par,
                                  f64 *SDSGE_RESTRICT out);

/* How to build the exogenous impact block, and where to evaluate it.
 *
 * ``fn`` writes the shock-carrying rows of d(residual)/d(shock) as a square
 * ``(n_exog, n_exog)`` row-major block, row k being equation ``rows[k]``. The
 * same rows index ``a``, so both sides of the solve are ordered by ``rows``.
 *
 * ``log_linear`` mirrors klein_preproc: the pencil is the jacobian of the
 * transformed residual, so the shock jacobian is evaluated at exp(ss) to match.
 * The log(1 + .) wrap contributes 1/(1 + resid), which is 1 at the steady
 * state, so only the evaluation point differs.
 */
typedef struct {
  shock_jacobian_fn fn;
  const f64 *SDSGE_RESTRICT a; /* (n_eq, n_var) row-major, from klein_preproc */
  const i64 *SDSGE_RESTRICT rows; /* (n_exog,), the row order fn emits */
  const f64 *SDSGE_RESTRICT ss;   /* (n_var,) */
  const f64 *SDSGE_RESTRICT par;  /* (n_par,) */
  i64 log_linear;
} sdsge_shock_ctx;

/* Scratch for the impact solve: the two evaluation buffers, the shock block,
 * the impact matrix (factored in place) and its solution, plus the pivots.
 * Zero on both counts when n_exog == 0. */
arena_size sdsge_assemble_arena_size(const i64 n_state, const i64 n_control,
                                     const i64 n_exog);

/* Assemble state-space into a caller-owned arena, so a per-draw loop allocates
 * once. ``shock``, ``arena`` and ``pivot`` are read only when n_exog > 0. */
i64 sdsge_assemble_state_space_into(
    const c128 *SDSGE_RESTRICT p, const c128 *SDSGE_RESTRICT f,
    const sdsge_shock_ctx *shock, const i64 n_state, const i64 n_control,
    const i64 n_exog, f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT pivot,
    f64 *SDSGE_RESTRICT A, f64 *SDSGE_RESTRICT B);

/* One-shot wrapper: allocates the arena, calls the kernel, frees. */
i64 sdsge_assemble_state_space(const c128 *SDSGE_RESTRICT p,
                               const c128 *SDSGE_RESTRICT f,
                               const sdsge_shock_ctx *shock, const i64 n_state,
                               const i64 n_control, const i64 n_exog,
                               f64 *SDSGE_RESTRICT A, f64 *SDSGE_RESTRICT B);

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
    f64 *SDSGE_RESTRICT x_out,  /* (T+1, nx) */
    f64 *SDSGE_RESTRICT y_out); /* (T+1, ny), nullable when ny == 0 */

/* ERROR CODES */
#define SDSGE_CORE_SUCCESS 0
#define SDSGE_CORE_ALLOC_FAIL -1
/* Exogenous impact block is singular: no unique within-period response. */
#define SDSGE_CORE_SINGULAR -2

#endif /* SDSGE_CORE_H */
