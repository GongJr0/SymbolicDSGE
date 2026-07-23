#include "optim.h"
#include "lbfgsb.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* L-BFGS-B task codes (mirror enum Status in lbfgsb.c). */
#define TASK_START 0
#define TASK_NEW_X 1
#define TASK_FG 3
#define TASK_CONVERGENCE 4
#define TASK_STOP 5

/* Stop-reason message codes we inject at NEW_X boundaries (mirror StatusMsg). */
#define MSG_STOP_ITER 502  /* nfev over maxfun */
#define MSG_STOP_ITERC 504 /* nit over maxiter */

static const char *sdsge_lbfgsb_message(i64 task) {
  switch (task) {
  case TASK_CONVERGENCE:
    return "CONVERGENCE";
  case TASK_STOP:
    return "STOP: iteration or evaluation limit reached";
  case 6:
    return "WARNING: rounding errors prevent progress";
  case 7:
    return "ERROR: invalid input";
  case 8:
    return "ABNORMAL: line search failed";
  default:
    return "UNKNOWN";
  }
}

/* Forward 2-point finite-difference gradient at x, matching scipy's default
 * (no jac): step = fd_step if > 0 else sqrt(eps)*max(1,|x_i|). f0 is f(x). Each
 * probe is one objective eval; *nfev is bumped per probe. A non-finite f0 marks
 * an infeasible point: the gradient is left at zero so the line search reduces
 * the step on the (non-finite) f value alone. Returns the probe count. */
static void sdsge_fd_grad(sdsge_objective_fn obj, void *ctx, i64 n, f64 *x,
                          f64 f0, f64 fd_step, f64 *g, i64 *nfev) {
  if (!isfinite(f0)) {
    for (i64 i = 0; i < n; ++i) {
      g[i] = 0.0;
    }
    return;
  }
  const f64 rel = sqrt(DBL_EPSILON);
  for (i64 i = 0; i < n; ++i) {
    const f64 xi = x[i];
    f64 h = fd_step > 0.0 ? fd_step : rel * fmax(1.0, fabs(xi));
    x[i] = xi + h;
    const f64 fp = obj(x, ctx);
    (*nfev)++;
    x[i] = xi;
    g[i] = isfinite(fp) ? (fp - f0) / h : 0.0;
  }
}

i64 sdsge_lbfgsb(sdsge_objective_fn obj, void *obj_ctx, i64 n,
                 f64 *SDSGE_RESTRICT x, const f64 *lo, const f64 *hi,
                 const i64 *nbd_in, const sdsge_lbfgsb_options *opt,
                 sdsge_lbfgsb_result *out) {
  const i64 N = n;
  const i64 m = opt->m;
  const i64 maxls = opt->maxls;
  const f64 factr = opt->factr;
  const f64 pgtol = opt->pgtol;

  /* Workspace (sizes per scipy's _lbfgsb_py wrapper). One allocation, freed on
   * return; the reverse-communication loop itself allocates nothing. */
  const size_t wa_len = (size_t)(2 * m * N + 5 * N + 11 * m * m + 8 * m);
  f64 *wa = (f64 *)malloc(wa_len * sizeof(f64));
  f64 *g = (f64 *)malloc((size_t)N * sizeof(f64));
  f64 *l = (f64 *)malloc((size_t)N * sizeof(f64));
  f64 *u = (f64 *)malloc((size_t)N * sizeof(f64));
  i64 *iwa = (i64 *)malloc((size_t)(3 * N) * sizeof(i64));
  i64 *nbd = (i64 *)malloc((size_t)N * sizeof(i64));
  if (!wa || !g || !l || !u || !iwa || !nbd) {
    free(wa);
    free(g);
    free(l);
    free(u);
    free(iwa);
    free(nbd);
    if (out) {
      out->status = SDSGE_OPTIM_EALLOC;
      out->success = 0;
      out->message = "ERROR: workspace allocation failed";
      out->nfev = 0;
      out->nit = 0;
      out->fun = NAN;
    }
    return SDSGE_OPTIM_EALLOC;
  }

  for (i64 i = 0; i < N; ++i) {
    l[i] = lo ? lo[i] : 0.0;
    u[i] = hi ? hi[i] : 0.0;
    nbd[i] = nbd_in ? nbd_in[i] : 0;
  }

  i64 task[2] = {TASK_START, 0};
  i64 ln_task[2] = {0, 0};
  i64 lsave[4] = {0, 0, 0, 0};
  i64 isave[44];
  f64 dsave[29];
  memset(isave, 0, sizeof(isave));
  memset(dsave, 0, sizeof(dsave));

  f64 f = 0.0;
  i64 nfev = 0, nit = 0;

  for (;;) {
    setulb(N, m, x, l, u, nbd, &f, g, factr, pgtol, wa, iwa, task, lsave, isave,
           dsave, maxls, ln_task);

    if (task[0] == TASK_FG) {
      f = obj(x, obj_ctx);
      nfev++;
      sdsge_fd_grad(obj, obj_ctx, N, x, f, opt->fd_step, g, &nfev);
    } else if (task[0] == TASK_NEW_X) {
      nit++;
      if (opt->maxiter > 0 && nit >= opt->maxiter) {
        task[0] = TASK_STOP;
        task[1] = MSG_STOP_ITERC;
      } else if (opt->maxfun > 0 && nfev > opt->maxfun) {
        task[0] = TASK_STOP;
        task[1] = MSG_STOP_ITER;
      }
    } else {
      break;
    }
  }

  const i64 status = (i64)task[0];
  if (out) {
    out->status = status;
    out->nfev = nfev;
    out->nit = nit;
    out->fun = f;
    out->success = (status == TASK_CONVERGENCE);
    out->message = sdsge_lbfgsb_message(status);
  }

  free(wa);
  free(g);
  free(l);
  free(u);
  free(iwa);
  free(nbd);
  return status;
}
