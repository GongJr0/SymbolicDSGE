#ifndef SDSGE_OPTIM_NELDER_MEAD_H
#define SDSGE_OPTIM_NELDER_MEAD_H

#include "../_common/sdsge_common.h" /* i64, f64, SDSGE_RESTRICT */
#include "optim.h"                   /* sdsge_objective_fn, SDSGE_OPTIM_EALLOC */

/* Native Nelder-Mead simplex driver (issue #335): a faithful transpilation of
 * scipy's BSD-3 `_minimize_neldermead` (scipy/optimize/_optimize.py). Gradient
 * free by construction: no FD gradient, no Hessian, no standard errors (the
 * contrast with #329's L-BFGS-B). It shares the objective-cfunc ABI
 * (`sdsge_objective_fn`) and the lean result-struct shape with that driver.
 *
 * Standard (non-adaptive) coefficients only: rho=1, chi=2, psi=0.5, sigma=0.5.
 * Bounds use scipy's clipping scheme, not a transform: the initial simplex is
 * reflected-then-clipped into the box and every trial point is clipped, so a
 * BK-violation region never yields an out-of-box vertex. Non-finite objective
 * returns (+INFINITY) rank worst in the simplex ordering and do not derail the
 * search. Parity against scipy is asserted within tolerance on x/fun, never on
 * the per-iteration trajectory: scipy re-sorts with a non-stable argsort, so
 * tie ordering is not reproducible across implementations. */

typedef struct {
  i64 maxiter; /* iteration cap; <= 0 -> N*200 (scipy default) */
  i64 maxfun;  /* objective-evaluation cap; <= 0 -> N*200 */
  f64 xatol;   /* converge when max|vertex_i - best| <= xatol AND ... */
  f64 fatol;   /* ... max|f_i - f_best| <= fatol */
} sdsge_neldermead_options;

typedef struct {
  i64 status;          /* warnflag: 0 success, 1 maxfun, 2 maxiter */
  i64 nfev;            /* objective evaluations */
  i64 nit;             /* iterations */
  f64 fun;             /* objective at the returned x */
  int success;         /* warnflag == 0 */
  const char *message; /* static string, keyed off status */
} sdsge_neldermead_result;

/* Nelder-Mead on a box-bounded objective. `x` is start -> optimum (length n).
 * Bounds follow the L-BFGS-B ABI: lo[i]/hi[i] gated by nbd[i] in
 * {0 none, 1 lower, 2 both, 3 upper}; nbd == NULL means fully unbounded (no
 * clipping at all, matching scipy's bounds=None path). The driver makes one
 * workspace allocation up front (never in the eval loop) and frees it on
 * return; it never longjmps. Returns the exit status (also in out->status).
 * SDSGE_OPTIM_EALLOC on a failed allocation, SDSGE_OPTIM_EINVAL on n < 1. */
i64 sdsge_neldermead(sdsge_objective_fn obj, void *obj_ctx, i64 n,
                     f64 *SDSGE_RESTRICT x, const f64 *lo, const f64 *hi,
                     const i64 *nbd, const sdsge_neldermead_options *opt,
                     sdsge_neldermead_result *out);

#define SDSGE_OPTIM_EINVAL (-2)

#endif /* SDSGE_OPTIM_NELDER_MEAD_H */
