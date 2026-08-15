#ifndef SDSGE_SECOND_ORDER_H
#define SDSGE_SECOND_ORDER_H

#include "../_common/sdsge_common.h"

/* Second-order policy tensors from the first-order solution and the residual
 * Hessian, by the chain-rule contraction of Juillard and Kamenik (2004).
 * Parity oracle: core.second_order.solve_second_order.
 *
 * Inputs (all C-contiguous, row-major, f64), with n = n_var = n_eq (square),
 * ny = n - nx and nz = 3n + ne:
 *   a    (n, n)      dF/dy_{t+1}
 *   b    (n, n)      -(dF/dy_t)
 *   f_xx (n, nz, nz) residual Hessian over z = (lag, cur, lead, eps)
 *   gx   (ny, nx)    controls from states
 *   hx   (nx, nx)    state transition
 *   bu   (n, ne)     shock impact, states over controls
 *   q    (ne, ne)    shock covariance
 *
 * dF/dy_{t-1} is not an input: y_{t-1} is the differentiation variable, so its
 * second derivative vanishes and the lag block reaches the result only through
 * the identity block of dz/dx.
 *
 * Outputs, each symmetric in its two trailing indices where the pair repeats:
 *   gxx (ny, nx, nx)  hxx (nx, nx, nx)
 *   gxu (ny, nx, ne)  hxu (nx, nx, ne)
 *   guu (ny, ne, ne)  huu (nx, ne, ne)
 *   gss (ny,)         hss (nx,)
 *
 * The cross and shock blocks are separate outputs rather than columns of one
 * tensor because the shocks are no longer states: an expansion reads x (x) x,
 * x (x) u and u (x) u with their own coefficients. gss/hss carry the same
 * convention as the quadratic blocks, with the 1/2 applied at use.
 *
 * Returns one of the SDSGE_SECOND_ORDER_* codes. */
arena_size sdsge_second_order_arena_size(i64 n, i64 nx, i64 ne);

i64 sdsge_second_order(const f64 *SDSGE_RESTRICT a, const f64 *SDSGE_RESTRICT b,
                       const f64 *SDSGE_RESTRICT f_xx,
                       const f64 *SDSGE_RESTRICT gx, const f64 *SDSGE_RESTRICT hx,
                       const f64 *SDSGE_RESTRICT bu, const f64 *SDSGE_RESTRICT q,
                       const i64 n, const i64 nx, const i64 ne,
                       f64 *SDSGE_RESTRICT gxx, f64 *SDSGE_RESTRICT hxx,
                       f64 *SDSGE_RESTRICT gxu, f64 *SDSGE_RESTRICT hxu,
                       f64 *SDSGE_RESTRICT guu, f64 *SDSGE_RESTRICT huu,
                       f64 *SDSGE_RESTRICT gss, f64 *SDSGE_RESTRICT hss,
                       f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena);

/* ERROR CODES */
#define SDSGE_SECOND_ORDER_OK 0
#define SDSGE_SECOND_ORDER_SINGULAR -801 /* Sylvester or cross-block system */
#define SDSGE_SECOND_ORDER_RISK -802     /* risk-correction system singular */

#endif /* SDSGE_SECOND_ORDER_H */
