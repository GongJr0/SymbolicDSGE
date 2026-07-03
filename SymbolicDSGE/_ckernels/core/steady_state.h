#ifndef SDSGE_STEADY_STATE_H
#define SDSGE_STEADY_STATE_H

#include "../_common/sdsge_common.h"
#include "klein_preproc.h" /* sdsge_residual_fn */

/* Newton solve of the deterministic steady state: F(ss, ss) = 0 with shocks off
 * (the residual cfunc already has them substituted to zero). Seeded at `seed`
 * (the configured steady state, essentially exact -> converges in 1-2 steps).
 *
 * Each iteration takes the exact Jacobian dF(x, x)/dx = a - b straight from
 * klein_preproc (a = dF/dfwd, b = -dF/dcur, both complex-step), forms the Newton
 * step by LU-solving (a - b) dx = F(x, x), and updates x -= dx. Levels residual
 * only (order-2 is levels-only); `log_linear` is not threaded here.
 *
 * Writes the converged point into `ss` (n_var); `*iters` gets the iteration
 * count. Returns one of the SDSGE_NEWTON_* codes. */
i64 sdsge_steady_state_newton(sdsge_residual_fn residual,
                              const f64 *SDSGE_RESTRICT seed,
                              const f64 *SDSGE_RESTRICT par, const i64 n_var,
                              const i64 n_par, const i64 max_iter, const f64 tol,
                              f64 *SDSGE_RESTRICT ss, i64 *SDSGE_RESTRICT iters);

/* ERROR CODES */
#define SDSGE_NEWTON_OK 0
#define SDSGE_NEWTON_ALLOC_FAIL -1
#define SDSGE_NEWTON_SINGULAR -2   /* Jacobian a - b singular */
#define SDSGE_NEWTON_NO_CONVERGE -3 /* tol not met within max_iter (or non-finite) */

#endif /* SDSGE_STEADY_STATE_H */
