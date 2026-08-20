#ifndef SDSGE_OPTIM_H
#define SDSGE_OPTIM_H

#include "../_common/sdsge_common.h"

/* Native optimizer drivers (issue #329). Shared subsystem: any consumer links
 * optim via _EXTRA_DEPS. Linear algebra comes from self-contained shims
 * (shim.c), so the driver takes no backend argument. */

/* Objective ABI: minimize f(x). Returns the scalar objective at x (length n).
 * A non-finite return (+INFINITY) marks an infeasible point; the driver's FD
 * gradient and line search treat it as "no decrease" and backtrack. `ctx` is
 * the caller's closure. */
typedef f64 (*sdsge_objective_fn)(const f64 *SDSGE_RESTRICT x, void *ctx);

/* Shared optimizer inputs. Each driver reads only its relevant fields:
 * L-BFGS-B uses m, maxiter, maxfun, maxls, factr, pgtol, and fd_step;
 * Nelder-Mead uses maxiter, maxfun, xatol, and fatol. */
typedef struct {
  i64 m;        /* limited-memory history length (L-BFGS-B) */
  i64 maxiter;  /* iteration cap */
  i64 maxfun;   /* objective-evaluation cap */
  i64 maxls;    /* max line-search steps (L-BFGS-B) */
  f64 factr;    /* L-BFGS-B f-progress tolerance multiplier */
  f64 pgtol;    /* L-BFGS-B projected-gradient tolerance */
  f64 fd_step;  /* L-BFGS-B forward-difference step; <= 0 uses its default */
  f64 xatol;    /* Nelder-Mead simplex-position tolerance */
  f64 fatol;    /* Nelder-Mead objective tolerance */
} sdsge_optim_options;

/* Shared optimizer outputs. `status` remains driver-specific. */
typedef struct {
  i64 status;
  i64 nfev;            /* objective evaluations */
  i64 nit;             /* iterations */
  f64 fun;             /* objective at the returned x */
  int success;
  const char *message; /* static string, keyed off driver status */
} sdsge_optim_result;

/* L-BFGS-B on a box-bounded objective. `x` is start -> optimum (length n).
 * Bounds: lo[i]/hi[i] gated by nbd[i] in {0 none, 1 lower, 2 both, 3 upper};
 * nbd == NULL means fully unbounded. The driver makes one workspace allocation
 * up front (not in the eval loop) and frees it on return; it never longjmps.
 * Returns the exit status (also in out->status). SDSGE_OPTIM_EALLOC on a failed
 * workspace allocation. */
i64 sdsge_lbfgsb(sdsge_objective_fn obj, void *obj_ctx, i64 n,
                 f64 *SDSGE_RESTRICT x, const f64 *lo, const f64 *hi,
                 const i64 *nbd, const sdsge_optim_options *opt,
                 sdsge_optim_result *out);

/* Optimizer status, shared with nelder_mead.h: one failure vocabulary for
 * every driver in this directory. */
#define SDSGE_OPTIM_EALLOC (-1701)
#define SDSGE_OPTIM_EINVAL (-1702)

#endif /* SDSGE_OPTIM_H */
