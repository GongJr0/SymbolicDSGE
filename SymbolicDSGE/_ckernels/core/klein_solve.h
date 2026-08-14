#ifndef SDSGE_KLEIN_SOLVE_H
#define SDSGE_KLEIN_SOLVE_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include "bicomplex_hessian.h" /* bc_residual_fn */
#include "klein_preproc.h"     /* sdsge_residual_fn */
#include "klein_qz.h"          /* klein_zgges_fn */
#include "pencil.h"            /* SDSGE_INC_*, sdsge_pencil_partition */

/* First-order Klein solve: the dimensions and the two cfuncs it drives. */
typedef struct {
  sdsge_residual_fn residual;
  klein_zgges_fn zgges;
  sdsge_dgeqrf_fn dgeqrf;
  sdsge_dormqr_fn dormqr;
  const f64 *ss_seed;  /* n_var: Newton seed for the steady state */
  const f64 *params;   /* n_par: calib_params order */
  const i8 *incidence; /* n_var: SDSGE_INC_* bits, unioned over the regimes */
  i64 n_var;           /* n_state + n_ctrl (square pencil) */
  i64 n_state;
  i64 n_ctrl;
  i64 n_exog;
  i64 n_par;
} klein_spec;

/* First-order Klein solve outputs; every buffer is caller-owned.
 *
 * `f` and `p` are real. The Schur algebra that produces them is complex, but
 * its imaginary parts are roundoff on a real pencil and no consumer has ever
 * read them, so the projection happens once here rather than at each use. The
 * complex originals live in the arena and do not outlive the solve. */
typedef struct {
  f64 *ss;     /* n_var: Newton-resolved steady state (from ss_seed) */
  f64 *a_real; /* n_var*n_var */
  f64 *b_real; /* n_var*n_var */
  f64 *c_real; /* n_var*n_var */
  f64 *d_real; /* n_var*n_exog */
  c128 *s;     /* nd*nd, nd from sdsge_pencil_dim */
  c128 *t;     /* nd*nd */
  c128 *z;     /* nd*nd */
  f64 *f;      /* n_ctrl*n_state, or NULL when n_ctrl == 0 */
  f64 *p;      /* n_state*n_state */
  c128 *eig;   /* nd */
  i64 stab;
  f64 *A; /* n_var*n_var */
  f64 *B; /* n_var*n_exog */

  /* The incidence's partition, derived by the solve and read again after the
   * QZ: the post-proc splits the reduced pencil on these counts and scatters
   * its rules back to full width through `order`. */
  i64 *order; /* n_var: original indices, [static | pred | both | fwd] */
  i64 n_static;
  i64 n_pred;
  i64 n_both;
  i64 n_fwd;
} sdsge_solve1;

/* Scratch for a whole first-order solve: a reserved head holding the complex
 * `f`/`p` the Schur post-proc emits, then the componentwise max over the
 * stages, which run one after another off the buffer past that head. `arena`
 * holds n_float f64, `iarena` n_int i64; both are caller-owned and may be
 * reused across solves. */
arena_size sdsge_klein_solve1_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                         i64 n_par, i64 n_exog, i64 nd);

/* Newton-resolve the steady state from spec->ss_seed, then linearize there.
 * Writes out->ss, the four Jacobian blocks a_real, b_real, c_real, d_real, and
 * the incidence's partition. The partition comes from the linearization rather
 * than the pencil half because it is a fact about the model, fixed across the
 * regime patches a caller may apply before solving. */
i64 sdsge_klein_linearize(const klein_spec *spec, sdsge_solve1 *out, f64 *arena,
                          i64 *iarena);

/* QZ and post-proc on the assembled real pencil (out->a_real, out->b_real),
 * then the state space. Split from the linearization so a caller can patch the
 * pencil rows in between and re-solve without another Newton or preproc.
 *
 * out->stab is reported, never acted on: a nonzero stab still leaves f/p/A/B
 * usable and the caller decides whether that is fatal. */
i64 sdsge_klein_from_pencil(const klein_spec *spec, sdsge_solve1 *out,
                            f64 *arena, i64 *iarena);

/* sdsge_klein_linearize, then sdsge_klein_from_pencil. */
i64 sdsge_klein_solve1(const klein_spec *spec, sdsge_solve1 *out, f64 *arena,
                       i64 *iarena);

/* Second-order (SGU) solve: the first-order spec plus the bicomplex residual
 * the Hessian sweep drives. Klein supplies the first order and nothing else,
 * hence the name split. */
typedef struct {
  klein_spec first;
  bc_residual_fn bc_residual;
  /* n_exog*n_exog, caller-filled: the Cholesky of the shock covariance. It is
   * refactored only when that covariance moves, and only the caller knows
   * whether it did, so a constant covariance is factored once and held. */
  f64 *chol;
} sgu_klein_spec;

/* Second-order solve buffers, all caller-owned outputs.
 *
 * `bx` is the state rows of `B`, which the risk correction pairs with the
 * spec's `chol` to load the innovations. It is an output rather than a caller's
 * object because `B` is the solve's own.
 *
 * The first-order rules the SGU tensors are built from are `sdsge_solve1.p` and
 * `.f`, which are already real; this struct does not restate them. */
typedef struct {
  f64 *f_xx; // n_var*(2*n_var)*(2*n_var)
  f64 *bx;   // n_state*n_exog
  f64 *gxx;  // n_ctrl*n_state*n_state
  f64 *hxx;  // n_state*n_state*n_state
  f64 *gss;  // n_ctrl
  f64 *hss;  // n_state
} sdsge_solve2;

/* sdsge_klein_solve1, then the second-order tail: the state rows of B, the
 * bicomplex Hessian at the resolved steady state, the SGU tensors and the
 * sigma^2 risk correction. Every first-order output stays in `out1`.
 *
 * out1->stab is reported, never acted on, exactly as at first order. */
arena_size sdsge_sgu_klein_solve2_arena_size(i64 n_var, i64 n_state, i64 n_ctrl,
                                             i64 n_par, i64 n_exog, i64 nd);

i64 sdsge_sgu_klein_solve2(const sgu_klein_spec *spec, sdsge_solve1 *out1,
                           sdsge_solve2 *out2, f64 *arena, i64 *iarena);

/* ERROR CODES. -2 and -3 come straight off sdsge_steady_state_newton, so the
 * linearization half passes its status through unmapped. */
#define SDSGE_KLEIN_SOLVE_OK 0
#define SDSGE_KLEIN_SOLVE_SS_SINGULAR -501
#define SDSGE_KLEIN_SOLVE_SS_NO_CONVERGE -502
#define SDSGE_KLEIN_SOLVE_QZ -503
#define SDSGE_KLEIN_SOLVE_SINGULAR -504     // singular z11/s11 (Blanchard-Kahn)
#define SDSGE_KLEIN_SOLVE_NO_STATES -505    // stateless model
#define SDSGE_KLEIN_SOLVE_SECOND_ORDER -506 // SGU system singular
#define SDSGE_KLEIN_SOLVE_RISK -507         // risk-correction system singular
#define SDSGE_KLEIN_SOLVE_ABSENT_VAR -508   // a variable occurs at no date
#define SDSGE_KLEIN_SOLVE_QR -509           // static rotation failed
#define SDSGE_KLEIN_SOLVE_STATIC -510       // static block singular

#endif /* SDSGE_KLEIN_SOLVE_H */
