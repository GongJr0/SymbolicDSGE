#include "klein_solve.h"
#include "bicomplex_hessian.h" /* sdsge_bicomplex_hessian */
#include "core.h"              /* sdsge_assemble_state_space */
#include "klein_postproc.h"    /* klein_postproc */
#include "klein_preproc.h"     /* klein_preproc */
#include "klein_qz.h"          /* klein_qz */
#include "second_order.h"      /* sdsge_second_order, _risk */
#include "steady_state.h"      /* sdsge_steady_state_newton */

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

/* Real part of a contiguous complex buffer. */
static inline void sdsge_real_part(const c128 *SDSGE_RESTRICT src,
                                   f64 *SDSGE_RESTRICT dst, const i64 len) {
  for (i64 k = 0; k < len; ++k) {
    dst[k] = c128_real(src[k]);
  }
}

static inline void sdsge_bx_from_B(const f64 *SDSGE_RESTRICT B,
                                   const i64 n_state, const i64 n_exog,
                                   f64 *SDSGE_RESTRICT out) {
  /* out = B[:n_state, :] */
  for (i64 k = 0; k < n_state * n_exog; ++k) {
    out[k] = B[k];
  }
}

/* Componentwise max: the stages run one after another off the same arena. */
static inline arena_size sdsge_max_arena(const arena_size a,
                                         const arena_size b) {
  return make_sizer(max_i64(a.n_float, b.n_float), max_i64(a.n_int, b.n_int));
}

arena_size sdsge_klein_solve1_arena_size(const i64 n_var, const i64 n_state,
                                         const i64 n_ctrl, const i64 n_par) {
  arena_size size = sdsge_newton_arena_size(n_var, n_par);
  size = sdsge_max_arena(size, klein_preproc_arena_size(n_var, n_par, n_var));
  size = sdsge_max_arena(size, klein_qz_arena_size(n_var));
  return sdsge_max_arena(size, klein_postproc_arena_size(n_state, n_ctrl));
}

arena_size sdsge_sgu_klein_solve2_arena_size(const i64 n_var, const i64 n_state,
                                             const i64 n_ctrl, const i64 n_par,
                                             const i64 n_exog) {
  arena_size size =
      sdsge_klein_solve1_arena_size(n_var, n_state, n_ctrl, n_par);
  size = sdsge_max_arena(
      size, sdsge_bicomplex_hessian_arena_size(n_var, n_par, n_var));
  size = sdsge_max_arena(size, sdsge_second_order_arena_size(n_var, n_state));
  return sdsge_max_arena(
      size, sdsge_second_order_risk_arena_size(n_var, n_state, n_exog));
}

i64 sdsge_klein_linearize(const klein_spec *spec, sdsge_solve1 *out,
                          f64 *arena, i64 *iarena) {
  const i64 n = spec->n_var;

  /* Resolve the steady state at the current params by Newton from ss_seed, then
   * linearize there. A gap model (ss = 0) seeds at 0 and converges in one step;
   * a params draw with no steady state fails and is rejected as infeasible. */
  i64 iters = 0;
  const i64 rc = sdsge_steady_state_newton(
      spec->residual, spec->ss_seed, spec->params, n, spec->n_par,
      SDSGE_SS_MAX_ITER, SDSGE_SS_TOL, out->ss, &iters, arena, iarena);
  if (rc != SDSGE_NEWTON_OK) {
    return rc;
  }

  klein_preproc(spec->residual, out->ss, spec->params, n, spec->n_par, n,
                out->a_real, out->b_real, arena);
  return SDSGE_KLEIN_SOLVE_OK;
}

i64 sdsge_klein_from_pencil(const klein_spec *spec, sdsge_solve1 *out,
                            f64 *arena, i64 *iarena) {
  const i64 n = spec->n_var;

  sdsge_to_complex_colmajor(out->a_real, out->s, n);
  sdsge_to_complex_colmajor(out->b_real, out->t, n);
  if (klein_qz(spec->zgges, n, out->s, out->t, out->z, arena, iarena) !=
      KLEIN_QZ_OK) {
    return SDSGE_KLEIN_SOLVE_QZ;
  }

  /* klein_qz emits column-major, klein_postproc reads row-major. */
  sdsge_transpose_sq(out->s, n);
  sdsge_transpose_sq(out->t, n);
  sdsge_transpose_sq(out->z, n);

  switch (klein_postproc(out->s, out->t, out->z, spec->n_state, spec->n_ctrl,
                         out->f, out->p, &out->stab, out->eig, arena, iarena)) {
  case SDSGE_KLEIN_POSTPROC_SUCCESS:
    break;
  case SDSGE_KLEIN_POSTPROC_INVALID:
    return SDSGE_KLEIN_SOLVE_NO_STATES;
  default:
    return SDSGE_KLEIN_SOLVE_SINGULAR;
  }

  sdsge_assemble_state_space(out->p, out->f, spec->n_state, spec->n_ctrl,
                             spec->n_exog, out->A, out->B);
  return SDSGE_KLEIN_SOLVE_OK;
}

i64 sdsge_klein_solve1(const klein_spec *spec, sdsge_solve1 *out, f64 *arena,
                       i64 *iarena) {
  const i64 rc = sdsge_klein_linearize(spec, out, arena, iarena);
  if (rc != SDSGE_KLEIN_SOLVE_OK) {
    return rc;
  }
  return sdsge_klein_from_pencil(spec, out, arena, iarena);
}

i64 sdsge_sgu_klein_solve2(const sgu_klein_spec *spec, sdsge_solve1 *out1,
                           sdsge_solve2 *out2, f64 *arena, i64 *iarena) {
  const klein_spec *s1 = &spec->first;

  const i64 rc = sdsge_klein_solve1(s1, out1, arena, iarena);
  if (rc != SDSGE_KLEIN_SOLVE_OK) {
    return rc;
  }

  /* The tensors are real; p/f carry ~1e-16 imaginary roundoff from the complex
   * Schur form. */
  sdsge_real_part(out1->p, out2->hx_real, s1->n_state * s1->n_state);
  sdsge_real_part(out1->f, out2->gx_real, s1->n_ctrl * s1->n_state);
  sdsge_bx_from_B(out1->B, s1->n_state, s1->n_exog, out2->bx);

  sdsge_bicomplex_hessian(spec->bc_residual, out1->ss, s1->params, s1->n_var,
                          s1->n_par, s1->n_var, out2->f_xx, arena);

  if (sdsge_second_order(out1->a_real, out1->b_real, out2->f_xx, out2->gx_real,
                         out2->hx_real, s1->n_var, s1->n_state, out2->gxx,
                         out2->hxx, arena, iarena) != SDSGE_SECOND_ORDER_OK) {
    return SDSGE_KLEIN_SOLVE_SECOND_ORDER;
  }

  if (sdsge_second_order_risk(out1->a_real, out1->b_real, out2->f_xx,
                              out2->gx_real, out2->gxx, out2->eta, s1->n_var,
                              s1->n_state, s1->n_exog, out2->gss, out2->hss,
                              arena, iarena) != SDSGE_SECOND_ORDER_OK) {
    return SDSGE_KLEIN_SOLVE_RISK;
  }
  return SDSGE_KLEIN_SOLVE_OK;
}
