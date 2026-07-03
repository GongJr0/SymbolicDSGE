#include "steady_state.h"
#include "../_common/sdsge_linalg.h"
#include <math.h>
#include <stdlib.h>

i64 sdsge_steady_state_newton(sdsge_residual_fn residual,
                              const f64 *SDSGE_RESTRICT seed,
                              const f64 *SDSGE_RESTRICT par, const i64 n_var,
                              const i64 n_par, const i64 max_iter, const f64 tol,
                              f64 *SDSGE_RESTRICT ss, i64 *SDSGE_RESTRICT iters) {
  const i64 n = n_var;
  const i64 nn = n * n;

  /* n_eq == n_var: the steady-state system is square. */
  c128 *SDSGE_RESTRICT x_c = (c128 *)malloc((size_t)(n > 0 ? n : 1) * sizeof(c128));
  c128 *SDSGE_RESTRICT par_c =
      (c128 *)malloc((size_t)(n_par > 0 ? n_par : 1) * sizeof(c128));
  c128 *SDSGE_RESTRICT out_c = (c128 *)malloc((size_t)(n > 0 ? n : 1) * sizeof(c128));
  f64 *SDSGE_RESTRICT r = (f64 *)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
  f64 *SDSGE_RESTRICT dx = (f64 *)malloc((size_t)(n > 0 ? n : 1) * sizeof(f64));
  f64 *SDSGE_RESTRICT a = (f64 *)malloc((size_t)(nn > 0 ? nn : 1) * sizeof(f64));
  f64 *SDSGE_RESTRICT b = (f64 *)malloc((size_t)(nn > 0 ? nn : 1) * sizeof(f64));
  f64 *SDSGE_RESTRICT J = (f64 *)malloc((size_t)(nn > 0 ? nn : 1) * sizeof(f64));

  i64 status = SDSGE_NEWTON_NO_CONVERGE;

  if (!x_c || !par_c || !out_c || !r || !dx || !a || !b || !J) {
    status = SDSGE_NEWTON_ALLOC_FAIL;
    goto done;
  }

  for (i64 i = 0; i < n; ++i) {
    ss[i] = seed[i];
  }
  for (i64 j = 0; j < n_par; ++j) {
    par_c[j] = c128_from_real(par[j]);
  }

  *iters = 0;
  for (i64 it = 0; it < max_iter; ++it) {
    /* r = Re F(ss, ss) (fwd == cur == ss, zero imaginary part). */
    for (i64 i = 0; i < n; ++i) {
      x_c[i] = c128_from_real(ss[i]);
    }
    residual(x_c, x_c, par_c, out_c);

    f64 nrm = 0.0;
    for (i64 k = 0; k < n; ++k) {
      r[k] = c128_real(out_c[k]);
      f64 ak = fabs(r[k]);
      if (ak > nrm) {
        nrm = ak;
      }
    }
    if (!isfinite(nrm)) {
      status = SDSGE_NEWTON_NO_CONVERGE;
      goto done;
    }
    if (nrm < tol) {
      *iters = it;
      status = SDSGE_NEWTON_OK;
      goto done;
    }

    /* Jacobian of F(x, x): dF/dfwd + dF/dcur = a + (-b) = a - b. */
    i64 perr = klein_preproc(residual, ss, par, n_var, n_par, n_var, 0, a, b);
    if (perr != SDSGE_PREKLEIN_OK) {
      status = SDSGE_NEWTON_ALLOC_FAIL;
      goto done;
    }
    sdsge_vsub(a, b, J, nn);

    /* dx = J^-1 r, then x -= dx. */
    i64 serr = sdsge_solve(J, r, n, 1, dx);
    if (serr == SDSGE_LU_SINGULAR) {
      status = SDSGE_NEWTON_SINGULAR;
      goto done;
    }
    if (serr != SDSGE_LU_SUCCESS) {
      status = SDSGE_NEWTON_ALLOC_FAIL;
      goto done;
    }
    sdsge_vsub(ss, dx, ss, n);
    *iters = it + 1;
  }

done:
  free(x_c);
  free(par_c);
  free(out_c);
  free(r);
  free(dx);
  free(a);
  free(b);
  free(J);
  return status;
}
