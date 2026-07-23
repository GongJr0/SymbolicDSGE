#include "nelder_mead.h"

#include <math.h>   /* fabs, isfinite, INFINITY, NAN */
#include <stdlib.h> /* malloc, free */
#include <string.h> /* memcpy */

/* scipy `_status_message` equivalents (scipy/optimize/_optimize.py). */
static const char *const MSG_SUCCESS = "Optimization terminated successfully.";
static const char *const MSG_MAXFEV =
    "Maximum number of function evaluations has been exceeded.";
static const char *const MSG_MAXITER =
    "Maximum number of iterations has been exceeded.";

/* np.clip(v, lb, ub) elementwise, matching scipy's per-trial-point clip. Only
 * called when bounds are active; lb/ub carry -/+INFINITY on absent sides, so an
 * unbounded coordinate is a no-op. */
static inline void nm_clip(f64 *SDSGE_RESTRICT v, const f64 *lb, const f64 *ub,
                           i64 n) {
  for (i64 i = 0; i < n; ++i) {
    if (v[i] < lb[i]) {
      v[i] = lb[i];
    } else if (v[i] > ub[i]) {
      v[i] = ub[i];
    }
  }
}

/* Stable insertion sort of the simplex rows + fsim by ascending fsim. scipy
 * re-sorts with np.argsort (introsort, not stable) every iteration, so tie
 * order is not reproducible across implementations; this uses a stable sort and
 * parity is asserted within tolerance, never on the trajectory. +inf entries
 * (infeasible / not-yet-evaluated vertices) compare greatest and rank last,
 * which is exactly scipy's ordering for them. Only inversions are moved, so a
 * near-sorted simplex (one replaced worst vertex) costs O(N). */
static void nm_sort(f64 *sim, f64 *fsim, i64 np1, i64 n, f64 *tmprow) {
  for (i64 i = 1; i < np1; ++i) {
    const f64 key = fsim[i];
    memcpy(tmprow, sim + i * n, (size_t)n * sizeof(f64));
    i64 j = i - 1;
    while (j >= 0 && fsim[j] > key) {
      fsim[j + 1] = fsim[j];
      memcpy(sim + (j + 1) * n, sim + j * n, (size_t)n * sizeof(f64));
      --j;
    }
    fsim[j + 1] = key;
    memcpy(sim + (j + 1) * n, tmprow, (size_t)n * sizeof(f64));
  }
}

static void nm_fill_error(sdsge_neldermead_result *out, i64 status,
                          const char *message) {
  if (out) {
    out->status = status;
    out->nfev = 0;
    out->nit = 0;
    out->fun = NAN;
    out->success = 0;
    out->message = message;
  }
}

i64 sdsge_neldermead(sdsge_objective_fn obj, void *obj_ctx, i64 n,
                     f64 *SDSGE_RESTRICT x, const f64 *lo, const f64 *hi,
                     const i64 *nbd, const sdsge_neldermead_options *opt,
                     sdsge_neldermead_result *out) {
  if (n < 1) {
    nm_fill_error(out, SDSGE_OPTIM_EINVAL, "ERROR: n must be >= 1");
    return SDSGE_OPTIM_EINVAL;
  }

  const i64 N = n;
  const i64 np1 = N + 1;
  const int has_bounds = (nbd != NULL);

  /* scipy defaults maxiter = maxfun = N*200 when unset. */
  const i64 maxiter = opt->maxiter > 0 ? opt->maxiter : N * 200;
  const i64 maxfun = opt->maxfun > 0 ? opt->maxfun : N * 200;
  const f64 xatol = opt->xatol;
  const f64 fatol = opt->fatol;

  /* Standard Nelder-Mead coefficients (non-adaptive branch of scipy). */
  const f64 rho = 1.0, chi = 2.0, psi = 0.5, sigma = 0.5;
  const f64 nonzdelt = 0.05, zdelt = 0.00025;

  /* One workspace allocation; the eval loop allocates nothing. Layout:
   * sim[(N+1)*N] | fsim[N+1] | xbar[N] | xr[N] | xt[N] | lb[N] | ub[N] |
   * tmprow[N]. */
  const size_t nd = (size_t)N;
  const size_t total = (size_t)np1 * nd + (size_t)np1 + nd * 6;
  f64 *buf = (f64 *)malloc(total * sizeof(f64));
  if (!buf) {
    nm_fill_error(out, SDSGE_OPTIM_EALLOC,
                  "ERROR: workspace allocation failed");
    return SDSGE_OPTIM_EALLOC;
  }

  f64 *sim = buf;
  f64 *fsim = sim + (size_t)np1 * nd;
  f64 *xbar = fsim + (size_t)np1;
  f64 *xr = xbar + nd;
  f64 *xt = xr + nd;
  f64 *lb = xt + nd;
  f64 *ub = lb + nd;
  f64 *tmprow = ub + nd;

  if (has_bounds) {
    for (i64 i = 0; i < N; ++i) {
      const i64 b = nbd[i];
      lb[i] = (b == 1 || b == 2) ? lo[i] : -INFINITY;
      ub[i] = (b == 2 || b == 3) ? hi[i] : INFINITY;
    }
  }

  /* Initial simplex: sim[0] = clip(x0); each sim[k+1] perturbs coordinate k of
   * the (clipped) x0 by nonzdelt (relative) or zdelt (when the coordinate is
   * exactly zero). */
  memcpy(sim, x, nd * sizeof(f64));
  if (has_bounds) {
    nm_clip(sim, lb, ub, N);
  }
  for (i64 k = 0; k < N; ++k) {
    f64 *row = sim + (size_t)(k + 1) * nd;
    memcpy(row, sim, nd * sizeof(f64));
    const f64 yk = row[k];
    row[k] = (yk != 0.0) ? (1.0 + nonzdelt) * yk : zdelt;
  }
  /* Reflect-then-clip the whole simplex into the box: if x0 sits near an upper
   * bound the step can push every vertex past it, so reflect those back before
   * clipping to avoid a degenerate (all-clamped) simplex. Unbounded sides carry
   * +/-INFINITY, so the reflect/clip is a no-op there. */
  if (has_bounds) {
    for (i64 r = 0; r < np1; ++r) {
      f64 *row = sim + (size_t)r * nd;
      for (i64 j = 0; j < N; ++j) {
        f64 s = row[j];
        if (s > ub[j]) {
          s = 2.0 * ub[j] - s;
        }
        if (s < lb[j]) {
          s = lb[j];
        } else if (s > ub[j]) {
          s = ub[j];
        }
        row[j] = s;
      }
    }
  }

  i64 nfev = 0;
  for (i64 k = 0; k < np1; ++k) {
    fsim[k] = INFINITY;
  }
  for (i64 k = 0; k < np1; ++k) {
    if (nfev >= maxfun) {
      break;
    }
    fsim[k] = obj(sim + (size_t)k * nd, obj_ctx);
    ++nfev;
  }
  nm_sort(sim, fsim, np1, N, tmprow);

  i64 nit = 1;

  while (nfev < maxfun && nit < maxiter) {
    /* Convergence: simplex spread and objective spread both within tol. A +inf
     * fsim entry keeps max_df at +inf, so an infeasible vertex never lets the
     * search terminate early. The `> || isnan` form is a NaN-propagating max
     * matching numpy's np.max: when the best vertex is itself infeasible,
     * fsim[0]-fsim[r] is inf-inf == NaN, and a NaN spread must fail the test
     * (scipy: `nan <= fatol` is False) rather than read as zero. */
    f64 max_dx = 0.0, max_df = 0.0;
    for (i64 r = 1; r < np1; ++r) {
      const f64 *row = sim + (size_t)r * nd;
      for (i64 j = 0; j < N; ++j) {
        const f64 d = fabs(row[j] - sim[j]);
        if (d > max_dx || isnan(d)) {
          max_dx = d;
        }
      }
      const f64 df = fabs(fsim[0] - fsim[r]);
      if (df > max_df || isnan(df)) {
        max_df = df;
      }
    }
    if (max_dx <= xatol && max_df <= fatol) {
      break;
    }

    /* Centroid of the best N vertices (excludes the worst, sim[-1]). Summed in
     * row order to mirror numpy's add.reduce accumulation. */
    f64 *worst = sim + (size_t)N * nd;
    for (i64 j = 0; j < N; ++j) {
      f64 acc = 0.0;
      for (i64 r = 0; r < N; ++r) {
        acc += sim[(size_t)r * nd + j];
      }
      xbar[j] = acc / (f64)N;
    }

    /* Reflection. */
    for (i64 j = 0; j < N; ++j) {
      xr[j] = (1.0 + rho) * xbar[j] - rho * worst[j];
    }
    if (has_bounds) {
      nm_clip(xr, lb, ub, N);
    }

    f64 fxr = 0.0;
    int doshrink = 0;
    if (nfev >= maxfun) {
      goto resort;
    }
    fxr = obj(xr, obj_ctx);
    ++nfev;

    if (fxr < fsim[0]) {
      /* Expansion. */
      for (i64 j = 0; j < N; ++j) {
        xt[j] = (1.0 + rho * chi) * xbar[j] - rho * chi * worst[j];
      }
      if (has_bounds) {
        nm_clip(xt, lb, ub, N);
      }
      if (nfev >= maxfun) {
        goto resort;
      }
      const f64 fxe = obj(xt, obj_ctx);
      ++nfev;
      if (fxe < fxr) {
        memcpy(worst, xt, nd * sizeof(f64));
        fsim[N] = fxe;
      } else {
        memcpy(worst, xr, nd * sizeof(f64));
        fsim[N] = fxr;
      }
    } else if (fxr < fsim[N - 1]) {
      /* Reflection accepted (better than the second-worst). */
      memcpy(worst, xr, nd * sizeof(f64));
      fsim[N] = fxr;
    } else if (fxr < fsim[N]) {
      /* Outside contraction. */
      for (i64 j = 0; j < N; ++j) {
        xt[j] = (1.0 + psi * rho) * xbar[j] - psi * rho * worst[j];
      }
      if (has_bounds) {
        nm_clip(xt, lb, ub, N);
      }
      if (nfev >= maxfun) {
        goto resort;
      }
      const f64 fxc = obj(xt, obj_ctx);
      ++nfev;
      if (fxc <= fxr) {
        memcpy(worst, xt, nd * sizeof(f64));
        fsim[N] = fxc;
      } else {
        doshrink = 1;
      }
    } else {
      /* Inside contraction. */
      for (i64 j = 0; j < N; ++j) {
        xt[j] = (1.0 - psi) * xbar[j] + psi * worst[j];
      }
      if (has_bounds) {
        nm_clip(xt, lb, ub, N);
      }
      if (nfev >= maxfun) {
        goto resort;
      }
      const f64 fxcc = obj(xt, obj_ctx);
      ++nfev;
      if (fxcc < fsim[N]) {
        memcpy(worst, xt, nd * sizeof(f64));
        fsim[N] = fxcc;
      } else {
        doshrink = 1;
      }
    }

    if (doshrink) {
      /* Shrink every vertex toward the best (sim[0], never moved). */
      for (i64 r = 1; r < np1; ++r) {
        f64 *row = sim + (size_t)r * nd;
        for (i64 j = 0; j < N; ++j) {
          row[j] = sim[j] + sigma * (row[j] - sim[j]);
        }
        if (has_bounds) {
          nm_clip(row, lb, ub, N);
        }
        if (nfev >= maxfun) {
          goto resort;
        }
        fsim[r] = obj(row, obj_ctx);
        ++nfev;
      }
    }

    ++nit;
  resort:
    /* Re-sort even on a mid-iteration cap (scipy sorts at the loop bottom after
     * catching its _MaxFuncCallError), so the returned sim[0] is the true best
     * so far. The forward goto skips the nit increment above, matching scipy's
     * "capped iteration does not count" behavior. */
    nm_sort(sim, fsim, np1, N, tmprow);
  }

  i64 warnflag = 0;
  const char *msg = MSG_SUCCESS;
  if (nfev >= maxfun) {
    warnflag = 1;
    msg = MSG_MAXFEV;
  } else if (nit >= maxiter) {
    warnflag = 2;
    msg = MSG_MAXITER;
  }

  /* The simplex is sorted at every exit (init, loop-bottom, or convergence
   * break after a prior sort), so sim[0]/fsim[0] are the optimum. */
  memcpy(x, sim, nd * sizeof(f64));
  const f64 fval = fsim[0];

  if (out) {
    out->status = warnflag;
    out->nfev = nfev;
    out->nit = nit;
    out->fun = fval;
    out->success = (warnflag == 0);
    out->message = msg;
  }

  free(buf);
  return warnflag;
}
