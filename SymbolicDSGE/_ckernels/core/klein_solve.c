#include "klein_solve.h"
#include "core.h"           /* sdsge_assemble_state_space */
#include "klein_postproc.h" /* klein_postproc */
#include "klein_preproc.h"  /* klein_preproc */
#include "klein_qz.h"       /* klein_qz */
#include "steady_state.h"   /* sdsge_steady_state_newton */

/* Newton steady-state config, matching the Python solver defaults. */
#define SDSGE_SS_MAX_ITER 50
#define SDSGE_SS_TOL 1e-12

/* Real pencil (row-major) -> complex Schur input (column-major), widened. */
static inline void sdsge_to_complex_colmajor(const f64 *SDSGE_RESTRICT a,
                                             c128 *SDSGE_RESTRICT s,
                                             const i64 n) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; ++j) {
      s[j * n + i] = c128_from_real(a[i * n + j]);
    }
  }
}

/* In-place square transpose (column-major <-> row-major). */
static inline void sdsge_transpose_sq(c128 *SDSGE_RESTRICT m, const i64 n) {
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = i + 1; j < n; ++j) {
      const c128 tmp = m[i * n + j];
      m[i * n + j] = m[j * n + i];
      m[j * n + i] = tmp;
    }
  }
}

i64 sdsge_klein_linearize(const sdsge_klein_spec *spec, sdsge_solve1 *out) {
  const i64 n = spec->n_var;

  /* Resolve the steady state at the current params by Newton from ss_seed, then
   * linearize there. A gap model (ss = 0) seeds at 0 and converges in one step;
   * a params draw with no steady state fails and is rejected as infeasible. */
  i64 iters = 0;
  const i64 rc = sdsge_steady_state_newton(spec->residual, spec->ss_seed,
                                           spec->params, n, spec->n_par,
                                           SDSGE_SS_MAX_ITER, SDSGE_SS_TOL,
                                           out->ss, &iters);
  if (rc != SDSGE_NEWTON_OK) {
    return rc;
  }

  if (klein_preproc(spec->residual, out->ss, spec->params, n, spec->n_par, n,
                    spec->log_linear, out->a_real,
                    out->b_real) != SDSGE_PREKLEIN_OK) {
    return SDSGE_KLEIN_SOLVE_ALLOC;
  }
  return SDSGE_KLEIN_SOLVE_OK;
}

i64 sdsge_klein_from_pencil(const sdsge_klein_spec *spec, sdsge_solve1 *out) {
  const i64 n = spec->n_var;

  sdsge_to_complex_colmajor(out->a_real, out->s, n);
  sdsge_to_complex_colmajor(out->b_real, out->t, n);
  const i64 qz = klein_qz(spec->zgges, n, out->s, out->t, out->z);
  if (qz == KLEIN_QZ_ALLOC_FAIL) {
    return SDSGE_KLEIN_SOLVE_ALLOC;
  }
  if (qz != KLEIN_QZ_OK) {
    return SDSGE_KLEIN_SOLVE_QZ;
  }

  /* klein_qz emits column-major, klein_postproc reads row-major. */
  sdsge_transpose_sq(out->s, n);
  sdsge_transpose_sq(out->t, n);
  sdsge_transpose_sq(out->z, n);

  switch (klein_postproc(out->s, out->t, out->z, spec->n_state, spec->n_ctrl,
                         out->f, out->p, &out->stab, out->eig)) {
  case SDSGE_KLEIN_POSTPROC_SUCCESS:
    break;
  case SDSGE_KLEIN_POSTPROC_ALLOC_FAIL:
    return SDSGE_KLEIN_SOLVE_ALLOC;
  case SDSGE_KLEIN_POSTPROC_INVALID:
    return SDSGE_KLEIN_SOLVE_NO_STATES;
  default:
    return SDSGE_KLEIN_SOLVE_SINGULAR;
  }

  sdsge_assemble_state_space(out->p, out->f, spec->n_state, spec->n_ctrl,
                             spec->n_exog, out->A, out->B);
  return SDSGE_KLEIN_SOLVE_OK;
}

i64 sdsge_klein_solve1(const sdsge_klein_spec *spec, sdsge_solve1 *out) {
  const i64 rc = sdsge_klein_linearize(spec, out);
  if (rc != SDSGE_KLEIN_SOLVE_OK) {
    return rc;
  }
  return sdsge_klein_from_pencil(spec, out);
}
