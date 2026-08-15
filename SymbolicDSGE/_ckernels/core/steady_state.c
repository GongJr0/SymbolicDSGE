#include "steady_state.h"
#include "../_common/sdsge_linalg.h"
#include "klein_preproc.h"
#include <math.h>

arena_size sdsge_newton_arena_size(const i64 n_var, const i64 n_par,
                                   const i64 n_exog) {
  const i64 own =
      2 * (2 * n_var + n_par + n_exog) /* x_c, par_c, out_c, eps_c */
      + 2 * n_var                      /* r, dx */
      + 4 * n_var * n_var              /* a, b, c, J */
      + n_var * n_exog;                /* d */
  const arena_size pre = klein_preproc_arena_size(n_var, n_par, n_exog, n_var);
  return make_sizer(own + pre.n_float, n_var /* LU pivot */);
}

i64 sdsge_steady_state_newton(sdsge_residual_fn residual,
                              const f64 *SDSGE_RESTRICT seed,
                              const f64 *SDSGE_RESTRICT par, const i64 n_var,
                              const i64 n_par, const i64 n_exog,
                              const i64 max_iter, const f64 tol,
                              f64 *SDSGE_RESTRICT ss, i64 *SDSGE_RESTRICT iters,
                              f64 *SDSGE_RESTRICT arena,
                              i64 *SDSGE_RESTRICT iarena) {
  const i64 n = n_var;
  const i64 nn = n * n;

  /* n_eq == n_var: the steady-state system is square. */
  c128 *cp = (c128 *)arena;
  c128 *x_c = cp;
  cp += n;
  c128 *par_c = cp;
  cp += n_par;
  c128 *out_c = cp;
  cp += n;
  c128 *eps_c = cp;
  cp += n_exog;

  f64 *p = (f64 *)cp;
  f64 *r = p;
  p += n;
  f64 *dx = p;
  p += n;
  f64 *a = p;
  p += nn;
  f64 *b = p;
  p += nn;
  f64 *c = p;
  p += nn;
  f64 *d = p;
  p += n * n_exog;
  f64 *J = p;
  p += nn;
  f64 *sweep = p;

  for (i64 i = 0; i < n; ++i) {
    ss[i] = seed[i];
  }
  for (i64 j = 0; j < n_par; ++j) {
    par_c[j] = c128_from_real(par[j]);
  }
  /* The deterministic steady state holds at a zero innovation. */
  for (i64 j = 0; j < n_exog; ++j) {
    eps_c[j] = c128_from_real(0.0);
  }

  *iters = 0;
  for (i64 it = 0; it < max_iter; ++it) {
    /* r = Re F(ss, ss, ss) (every date == ss, zero imaginary part). */
    for (i64 i = 0; i < n; ++i) {
      x_c[i] = c128_from_real(ss[i]);
    }
    residual(x_c, x_c, x_c, eps_c, par_c, out_c);

    f64 nrm = 0.0;
    for (i64 k = 0; k < n; ++k) {
      r[k] = c128_real(out_c[k]);
      f64 ak = fabs(r[k]);
      if (ak > nrm) {
        nrm = ak;
      }
    }
    if (!isfinite(nrm)) {
      return SDSGE_NEWTON_NO_CONVERGE;
    }
    if (nrm < tol) {
      *iters = it;
      return SDSGE_NEWTON_OK;
    }

    /* Jacobian of F(x, x, x): dF/dfwd + dF/dcur + dF/dprev = a - b - c. */
    klein_preproc(residual, ss, par, n_var, n_par, n_exog, n_var, a, b, c, d,
                  sweep);
    sdsge_vsub(a, b, J, nn);
    sdsge_vsub(J, c, J, nn);

    /* dx = J^-1 r, then x -= dx. J is rebuilt next iteration, so the
     * factorization overwrites it. */
    if (sdsge_lu_factor_inplace(J, iarena, n) != SDSGE_LU_SUCCESS) {
      return SDSGE_NEWTON_SINGULAR;
    }
    sdsge_lu_solve(J, iarena, r, dx, n, 1);
    sdsge_vsub(ss, dx, ss, n);
    *iters = it + 1;
  }

  return SDSGE_NEWTON_NO_CONVERGE;
}
