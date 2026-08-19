#ifndef SDSGE_OPTIM_NELDER_MEAD_H
#define SDSGE_OPTIM_NELDER_MEAD_H

#include "../_common/sdsge_common.h" /* i64, f64, SDSGE_RESTRICT */
#include "optim.h" /* sdsge_objective_fn, SDSGE_OPTIM_EALLOC, _EINVAL */

/* Native Nelder-Mead simplex driver (issue #335): a faithful transpilation of
 * scipy's BSD-3 `_minimize_neldermead` (scipy/optimize/_optimize.py). Gradient
 * free: no FD gradient, no Hessian, no standard errors.
 *
 * Standard (non-adaptive) coefficients only: rho=1, chi=2, psi=0.5, sigma=0.5.
 * Bounds clip rather than transform: the initial simplex is reflected-then-
 * clipped into the box and every trial point is clipped. Non-finite objective
 * returns (+INFINITY) rank worst in the simplex ordering. */

/* Nelder-Mead on a box-bounded objective. `x` is start -> optimum (length n).
 * Bounds follow the L-BFGS-B ABI: lo[i]/hi[i] gated by nbd[i] in
 * {0 none, 1 lower, 2 both, 3 upper}; nbd == NULL means fully unbounded (no
 * clipping at all, matching scipy's bounds=None path). The driver makes one
 * workspace allocation up front (never in the eval loop) and frees it on
 * return; it never longjmps. Returns the exit status (also in out->status).
 * SDSGE_OPTIM_EALLOC on a failed allocation, SDSGE_OPTIM_EINVAL on n < 1. */
i64 sdsge_neldermead(sdsge_objective_fn obj, void *obj_ctx, i64 n,
                     f64 *SDSGE_RESTRICT x, const f64 *lo, const f64 *hi,
                     const i64 *nbd, const sdsge_optim_options *opt,
                     sdsge_optim_result *out);

#endif /* SDSGE_OPTIM_NELDER_MEAD_H */
