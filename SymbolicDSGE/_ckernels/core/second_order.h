#ifndef SDSGE_SECOND_ORDER_H
#define SDSGE_SECOND_ORDER_H

#include "../_common/sdsge_common.h"

/* Second-order (SGU) policy tensors from the first-order solution and the
 * residual Hessian. Native transcription of core.second_order.solve_second_order
 * (row-major, allocation-free inner loop): builds the symmetry-reduced linear
 * system big_q @ sym, solves it with the f64 LU, and expands via sym.
 *
 * Inputs (all C-contiguous, row-major, f64), with n = n_var = n_eq (square) and
 * ny = n - nx:
 *   a    (n, n)      first-order pencil dF/dfwd   (fxp = a[:, :nx], fyp = a[:, nx:])
 *   b    (n, n)      -(dF/dcur)                    (fx = -b[:, :nx], fy = -b[:, nx:])
 *   f_xx (n, 2n, 2n) residual Hessian, stacked z = [x'; y'; x; y]
 *   gx   (ny, nx)    controls-from-states (g_x)
 *   hx   (nx, nx)    state transition (h_x)
 * Outputs:
 *   gxx  (ny, nx, nx)  controls, symmetric in the last two indices
 *   hxx  (nx, nx, nx)  states, symmetric in the last two indices
 *
 * Returns one of the SDSGE_SECOND_ORDER_* codes. */
i64 sdsge_second_order(const f64 *SDSGE_RESTRICT a, const f64 *SDSGE_RESTRICT b,
                       const f64 *SDSGE_RESTRICT f_xx,
                       const f64 *SDSGE_RESTRICT gx,
                       const f64 *SDSGE_RESTRICT hx, const i64 n, const i64 nx,
                       f64 *SDSGE_RESTRICT gxx, f64 *SDSGE_RESTRICT hxx);

/* Sigma^2 risk correction (g_ss, h_ss) -- native transcription of
 * core.second_order.solve_second_order_risk. Only the forward-forward Hessian
 * blocks enter. Inputs as above plus:
 *   gxx  (ny, nx, nx)  second-order controls (from sdsge_second_order)
 *   eta  (nx, ne)      shock loading (eta @ eta^T = state innovation covariance)
 * Outputs:
 *   gss  (ny,)   controls risk correction
 *   hss  (nx,)   states risk correction
 * Solves the (n, n) system [Qg Qh] [gss; hss] = -q. Same return codes. */
i64 sdsge_second_order_risk(const f64 *SDSGE_RESTRICT a,
                            const f64 *SDSGE_RESTRICT b,
                            const f64 *SDSGE_RESTRICT f_xx,
                            const f64 *SDSGE_RESTRICT gx,
                            const f64 *SDSGE_RESTRICT gxx,
                            const f64 *SDSGE_RESTRICT eta, const i64 n,
                            const i64 nx, const i64 ne, f64 *SDSGE_RESTRICT gss,
                            f64 *SDSGE_RESTRICT hss);

/* ERROR CODES */
#define SDSGE_SECOND_ORDER_OK 0
#define SDSGE_SECOND_ORDER_ALLOC_FAIL -1
#define SDSGE_SECOND_ORDER_SINGULAR -2 /* symmetry-reduced system singular */

#endif /* SDSGE_SECOND_ORDER_H */
