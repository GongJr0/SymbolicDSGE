#ifndef SDSGE_KLEIN_SOLVE_H
#define SDSGE_KLEIN_SOLVE_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include "klein_preproc.h" /* sdsge_residual_fn */
#include "klein_qz.h"      /* klein_zgges_fn */

/* First-order Klein solve: the dimensions and the two cfuncs it drives. */
typedef struct {
  sdsge_residual_fn residual;
  klein_zgges_fn zgges;
  const f64 *ss_seed; /* n_var: Newton seed for the steady state */
  const f64 *params;  /* n_par: calib_params order */
  i64 n_var;          /* n_state + n_ctrl (square pencil) */
  i64 n_state;
  i64 n_ctrl;
  i64 n_exog;
  i64 n_par;
} sdsge_klein_spec;

/* First-order Klein solve outputs; every buffer is caller-owned. */
typedef struct {
  f64 *ss;     /* n_var: Newton-resolved steady state (from ss_seed) */
  f64 *a_real; /* n_var*n_var */
  f64 *b_real; /* n_var*n_var */
  c128 *s;     /* n_var*n_var */
  c128 *t;     /* n_var*n_var */
  c128 *z;     /* n_var*n_var */
  c128 *f;     /* n_ctrl*n_state, or NULL when n_ctrl == 0 */
  c128 *p;     /* n_state*n_state */
  c128 *eig;   /* n_var */
  i64 stab;
  f64 *A; /* n_var*n_var */
  f64 *B; /* n_var*n_exog */
} sdsge_solve1;

/* Newton-resolve the steady state from spec->ss_seed, then linearize there.
 * Writes out->ss, out->a_real and out->b_real. */
i64 sdsge_klein_linearize(const sdsge_klein_spec *spec, sdsge_solve1 *out);

/* QZ and post-proc on the assembled real pencil (out->a_real, out->b_real),
 * then the state space. Split from the linearization so a caller can patch the
 * pencil rows in between and re-solve without another Newton or preproc.
 *
 * out->stab is reported, never acted on: a nonzero stab still leaves f/p/A/B
 * usable and the caller decides whether that is fatal. */
i64 sdsge_klein_from_pencil(const sdsge_klein_spec *spec, sdsge_solve1 *out);

/* sdsge_klein_linearize, then sdsge_klein_from_pencil. */
i64 sdsge_klein_solve1(const sdsge_klein_spec *spec, sdsge_solve1 *out);

/* ERROR CODES. -1..-3 come straight off sdsge_steady_state_newton, so the
 * linearization half passes its status through unmapped. */
#define SDSGE_KLEIN_SOLVE_OK 0
#define SDSGE_KLEIN_SOLVE_ALLOC -1
#define SDSGE_KLEIN_SOLVE_SS_SINGULAR -2
#define SDSGE_KLEIN_SOLVE_SS_NO_CONVERGE -3
#define SDSGE_KLEIN_SOLVE_QZ -4
#define SDSGE_KLEIN_SOLVE_SINGULAR -5  /* singular z11/s11 (Blanchard-Kahn) */
#define SDSGE_KLEIN_SOLVE_NO_STATES -6 /* stateless model */

#endif /* SDSGE_KLEIN_SOLVE_H */
