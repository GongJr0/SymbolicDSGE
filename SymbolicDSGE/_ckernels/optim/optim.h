#ifndef SDSGE_OPTIM_H
#define SDSGE_OPTIM_H

#include "../_common/sdsge_common.h"

/* Native optimizer drivers (issue #329). Shared subsystem: any consumer links
 * optim via _EXTRA_DEPS. The estimation-specific objective trampoline lives in
 * the estimation module and wires in at #330; nothing here depends on it. The
 * linear-algebra primitives the L-BFGS-B kernel needs are served by
 * self-contained shims (shim.c), so the driver takes no backend argument. */

/* Objective ABI: minimize f(x). Returns the scalar objective at x (length n).
 * A non-finite return (+INFINITY) marks an infeasible point; the driver's FD
 * gradient and line search treat it as "no decrease" and backtrack. `ctx` is the
 * caller's closure: benchmark params in tests, or the estimation trampoline over
 * the native sdsge_obj_* objective at wiring time. */
typedef f64 (*sdsge_objective_fn)(const f64 *SDSGE_RESTRICT x, void *ctx);

typedef struct {
  i64 m;        /* limited-memory history length (L-BFGS-B) */
  i64 maxiter;  /* iteration cap (NEW_X boundaries) */
  i64 maxfun;   /* objective-evaluation cap */
  i64 maxls;    /* max line-search steps per iteration */
  f64 factr;    /* stop when f-progress <= factr*eps_machine */
  f64 pgtol;    /* stop when the projected gradient inf-norm <= pgtol */
  f64 fd_step;  /* forward-difference step; <= 0 -> sqrt(eps)*max(1,|x|) default */
} sdsge_lbfgsb_options;

typedef struct {
  i64 status;          /* raw L-BFGS-B task code at exit */
  i64 nfev;            /* objective evaluations */
  i64 nit;             /* iterations */
  f64 fun;             /* objective at the returned x */
  int success;         /* converged (grad/f test) vs stopped/error */
  const char *message; /* static string, keyed off status */
} sdsge_lbfgsb_result;

/* L-BFGS-B on a box-bounded objective. `x` is start -> optimum (length n).
 * Bounds: lo[i]/hi[i] gated by nbd[i] in {0 none, 1 lower, 2 both, 3 upper};
 * nbd == NULL means fully unbounded. The driver makes one workspace allocation
 * up front (not in the eval loop) and frees it on return; it never longjmps.
 * Returns the exit status (also in out->status). SDSGE_OPTIM_EALLOC on a failed
 * workspace allocation. */
i64 sdsge_lbfgsb(sdsge_objective_fn obj, void *obj_ctx, i64 n,
                 f64 *SDSGE_RESTRICT x, const f64 *lo, const f64 *hi,
                 const i64 *nbd, const sdsge_lbfgsb_options *opt,
                 sdsge_lbfgsb_result *out);

#define SDSGE_OPTIM_EALLOC (-1)

#endif /* SDSGE_OPTIM_H */
