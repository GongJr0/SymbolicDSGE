#include "klein_solve.h"
#include "../_common/sdsge_complex.h" /* c128_* */
#include "../_common/sdsge_linalg.h"  /* sdsge_matmul */
#include "bicomplex_hessian.h"        /* sdsge_bicomplex_hessian */
#include "core.h"                     /* sdsge_assemble_transition */
#include "klein_classify.h"           /* klein_z11_pair, klein_reorder_argmax */
#include "klein_preproc.h"            /* klein_preproc */
#include "klein_qz.h"                 /* klein_qz */
#include "second_order.h"             /* sdsge_second_order */
#include "steady_state.h"             /* sdsge_steady_state_newton */
#include <string.h>                   /* memcpy */

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

/* f64 head reserved for the complex f/p the post-proc emits, which must outlive
 * the post-proc's own scratch: the state-space assembly reads them after it. */
static inline i64 sdsge_solve1_fp_reserve(const i64 n_state, const i64 n_ctrl) {
  return 2 * (n_ctrl * n_state + n_state * n_state);
}

/* Scratch for klein_rule_from_schur (z21, s11, t11, dyn) plus the pivots it
 * shares with the classify half. Declared rather than left to fit under the
 * classify entry, which is the larger of the two only by coincidence. */
static inline arena_size sdsge_rule_arena_size(const i64 n_s, const i64 n_cs) {
  return make_sizer(2 * (n_cs * n_s + 3 * n_s * n_s), n_s /* LU pivots */);
}

/* Stage max only. The reserve is added once by the public sizers, so it is
 * never folded into a max and then compared against a later stage. */
static inline arena_size sdsge_pencil_stage_arena(const i64 n_var,
                                                  const i64 n_state,
                                                  const i64 n_exog,
                                                  const i64 nd) {
  /* n_state is nspred, which the frontend checks against the incidence before a
   * solve runs. nsfwrd has no such twin among these counts, so the terms it
   * scales stay bounded by nd. Held flat rather than maxed: the rotated blocks
   * and the recovered rules coexist across the stage. */
  const i64 own = 3 * n_var * n_var /* a_rot, b_rot, c_rot */
                  + n_var * n_exog  /* d_rot */
                  + 2 * nd * nd     /* E, D */
                  + 2 * (nd * n_state + n_state * n_state) /* complex gx, hx */
                  + nd * n_state + n_state * n_state       /* real gx, hx */
                  + 4 * n_state * n_state /* complex z11, z11i */
                  + 4 * n_state * n_state /* complex tmp, eye */
                  + n_var * n_state       /* ghx */
                  + n_var * n_var         /* amat */
                  + n_var * n_state       /* C@gx, then the static rhs */
                  + n_var * n_exog;       /* ghu */
  const arena_size rot = sdsge_pencil_rotate_arena_size(
      n_var, n_var, n_var > n_exog ? n_var : n_exog);
  arena_size tail = sdsge_max_arena(rot, klein_qz_arena_size(nd));
  tail = sdsge_max_arena(tail, klein_classify_arena_size(nd, n_state));
  tail = sdsge_max_arena(tail, sdsge_rule_arena_size(n_state, nd));
  return make_sizer(own + tail.n_float, tail.n_int + n_var + nd);
}

static inline arena_size
sdsge_solve1_stage_arena(const i64 n_var, const i64 n_par, const i64 n_exog) {
  arena_size size = sdsge_newton_arena_size(n_var, n_par, n_exog);
  return sdsge_max_arena(size,
                         klein_preproc_arena_size(n_var, n_par, n_exog, n_var));
}

arena_size sdsge_klein_solve1_arena_size(const i64 n_var, const i64 n_state,
                                         const i64 n_ctrl, const i64 n_par,
                                         const i64 n_exog, const i64 nd) {
  arena_size size = sdsge_solve1_stage_arena(n_var, n_par, n_exog);
  size = sdsge_max_arena(size,
                         sdsge_pencil_stage_arena(n_var, n_state, n_exog, nd));
  size.n_float += sdsge_solve1_fp_reserve(n_state, n_ctrl);
  return size;
}

arena_size sdsge_sgu_klein_solve2_arena_size(const i64 n_var, const i64 n_state,
                                             const i64 n_ctrl, const i64 n_par,
                                             const i64 n_exog, const i64 nd) {
  arena_size size = sdsge_solve1_stage_arena(n_var, n_par, n_exog);
  size = sdsge_max_arena(size,
                         sdsge_pencil_stage_arena(n_var, n_state, n_exog, nd));
  size = sdsge_max_arena(
      size, sdsge_bicomplex_hessian_arena_size(n_var, n_par, n_exog, n_var));
  size = sdsge_max_arena(size,
                         sdsge_second_order_arena_size(n_var, n_state, n_exog));
  /* Second-order stages run past the same head: solve1 is nested inside. */
  size.n_float += sdsge_solve1_fp_reserve(n_state, n_ctrl);
  return size;
}

i64 sdsge_klein_linearize(const klein_spec *spec, sdsge_solve1 *out, f64 *arena,
                          i64 *iarena) {
  const i64 n = spec->n_var;

  /* Resolve the steady state at the current params by Newton from ss_seed, then
   * linearize there. A gap model (ss = 0) seeds at 0 and converges in one step;
   * a params draw with no steady state fails and is rejected as infeasible. */
  i64 iters = 0;
  f64 *stage = arena + sdsge_solve1_fp_reserve(spec->n_state, spec->n_ctrl);

  i64 rc = sdsge_steady_state_newton(
      spec->residual, spec->ss_seed, spec->params, n, spec->n_par, spec->n_exog,
      SDSGE_SS_MAX_ITER, SDSGE_SS_TOL, out->ss, &iters, stage, iarena);
  if (rc != SDSGE_NEWTON_OK) {
    return rc;
  }

  klein_preproc(spec->residual, out->ss, spec->params, n, spec->n_par,
                spec->n_exog, n, out->a_real, out->b_real, out->c_real,
                out->d_real, stage);

  if (sdsge_pencil_partition(spec->incidence, n, out->order, &out->n_static,
                             &out->n_pred, &out->n_both,
                             &out->n_fwd) != SDSGE_PENCIL_OK) {
    return SDSGE_KLEIN_SOLVE_ABSENT_VAR;
  }
  return SDSGE_KLEIN_SOLVE_OK;
}

/* f and p from the chosen Schur form and its z11 pair, row-major throughout.
 * No return code: every way this can fail is already decided upstream. s11 is
 * the leading block of an upper triangular S, so partial pivoting picks its own
 * diagonal, and klein_root_finite has already rejected a near-zero one; the LU
 * cannot report singular. Takes n_cs*n_s + 3*n_s*n_s c128 off `arena`. */
static void klein_rule_from_schur(
    const c128 *SDSGE_RESTRICT s, const c128 *SDSGE_RESTRICT t,
    const c128 *SDSGE_RESTRICT z, const i64 n_s, const i64 n_cs,
    const c128 *SDSGE_RESTRICT z11, const c128 *SDSGE_RESTRICT z11i,
    c128 *SDSGE_RESTRICT f, c128 *SDSGE_RESTRICT p, i64 *SDSGE_RESTRICT piv,
    f64 *SDSGE_RESTRICT arena) {
  const i64 N = n_s + n_cs;
  const i64 ssq = n_s * n_s;
  c128 *z21 = (c128 *)arena;
  c128 *s11 = z21 + n_cs * n_s;
  c128 *t11 = s11 + ssq;
  c128 *dyn = t11 + ssq;

  for (i64 i = 0; i < n_s; ++i) {
    for (i64 j = 0; j < n_s; ++j) {
      s11[i * n_s + j] = s[i * N + j];
      t11[i * n_s + j] = t[i * N + j];
    }
  }
  for (i64 i = 0; i < n_cs; ++i) {
    for (i64 j = 0; j < n_s; ++j) {
      z21[i * n_s + j] = z[(n_s + i) * N + j];
    }
  }

  c128_lu_factor_inplace(s11, piv, n_s);
  c128_lu_solve(s11, piv, t11, dyn, n_s, n_s);

  c128_matmul(z21, z11i, n_cs, n_s, n_s, f);
  /* s11 is dead past the solve, so it carries z11 @ dyn into the second half.
   */
  c128_matmul(z11, dyn, n_s, n_s, n_s, s11);
  c128_matmul(s11, z11i, n_s, n_s, n_s, p);
}

/* Scratch for the pencil half, past the f/p reserve. Held flat rather than
 * maxed: the rotated blocks and the recovered rules coexist across the whole
 * stage, and at these sizes the slack is a few kilobytes. */
i64 sdsge_klein_from_pencil(const klein_spec *spec, sdsge_solve1 *out,
                            f64 *arena, i64 *iarena) {
  const i64 n = spec->n_var;
  const i64 ne = spec->n_exog;
  const i64 nstatic = out->n_static;
  const i64 npred = out->n_pred;
  const i64 nboth = out->n_both;
  const i64 nfwd = out->n_fwd;
  const i64 nspred = npred + nboth;
  const i64 nsfwrd = nboth + nfwd;
  const i64 nd = npred + nboth + nfwd + nboth;
  const i64 *ord = out->order;

  /* Stamped up front so every early return leaves a non-zero (undetermined)
   * stab. Estimation reads it in the same expression as the return code, so an
   * unwritten one is an uninitialized read even where the value goes unused. */
  out->stab = SDSGE_KLEIN_STAB_UNSET;

  if (nspred <= 0) {
    return SDSGE_KLEIN_SOLVE_NO_STATES;
  }

  /* The two complex blocks step in c128 units */
  f64 *a_rot = arena + sdsge_solve1_fp_reserve(spec->n_state, spec->n_ctrl);
  f64 *b_rot = a_rot + n * n;
  f64 *c_rot = b_rot + n * n;
  f64 *d_rot = c_rot + n * n;
  f64 *emat = d_rot + n * ne;
  f64 *dmat = emat + nd * nd;
  c128 *gx_c = (c128 *)(dmat + nd * nd);
  c128 *hx_c = gx_c + nsfwrd * nspred;
  f64 *gx = (f64 *)(hx_c + nspred * nspred);
  f64 *hx = gx + nsfwrd * nspred;
  c128 *z11 = (c128 *)(hx + nspred * nspred);
  c128 *z11i = z11 + nspred * nspred;
  c128 *tmp = z11i + nspred * nspred; /* the z11 LU runs on a copy */
  c128 *eye = tmp + nspred * nspred;
  f64 *ghx = (f64 *)(eye + nspred * nspred);
  f64 *amat = ghx + n * nspred;
  f64 *work = amat + n * n; /* n by nspred: C@gx, then the static RHS */
  f64 *ghu = work + n * nspred;
  f64 *stage = ghu + n * ne;
  i64 *piv = iarena; /* shared by the z11 pair and the rule's s11 */

  memcpy(a_rot, out->a_real, n * n * sizeof(f64));
  memcpy(b_rot, out->b_real, n * n * sizeof(f64));
  memcpy(c_rot, out->c_real, n * n * sizeof(f64));
  memcpy(d_rot, out->d_real, n * ne * sizeof(f64));

  /* Rotate the static equations to the top. Every block turns with the same Q
   * so they stay one system; `b` supplies the static columns being cleared. */
  f64 *blocks[4] = {a_rot, b_rot, c_rot, d_rot};
  const i64 widths[4] = {n, n, n, ne};
  if (sdsge_pencil_rotate_static(spec->dgeqrf, spec->dormqr, out->b_real, ord,
                                 n, nstatic, blocks, widths, 4,
                                 stage) != SDSGE_PENCIL_OK) {
    return SDSGE_KLEIN_SOLVE_QR;
  }

  sdsge_pencil_assemble(a_rot, b_rot, c_rot, ord, n, nstatic, npred, nboth,
                        nfwd, emat, dmat);

  sdsge_to_complex_colmajor(dmat, out->s, nd);
  sdsge_to_complex_colmajor(emat, out->t, nd);
  i64 sdim = 0;
  if (klein_qz(spec->zgges, nd, out->s, out->t, out->z, &sdim, stage, iarena) !=
      KLEIN_QZ_OK) {
    return SDSGE_KLEIN_SOLVE_QZ;
  }

  /* Too few stable roots: the leading block carries unstable ones, so its z11
   * pair means nothing and no rule is reachable. Returned before the
   * factorization rather than after it. */
  if (sdim < nspred) {
    klein_fill_eig(out->s, out->t, out->eig, nd);
    sdsge_transpose_sq(out->s, nd);
    sdsge_transpose_sq(out->t, nd);
    sdsge_transpose_sq(out->z, nd);
    return SDSGE_KLEIN_NO_STABLE_SOLUTION;
  }

  /* More stable roots than states leaves the QZ's leading block one arbitrary
   * choice among several, so take the best-conditioned one it can reach by a
   * single swap. Estimation leaves ztgexc NULL and keeps the QZ's own: an
   * indeterminate draw is rejected either way. */
  klein_fill_eye(eye, nspred);
  f64 rcond = 0.0;
  i64 rc;
  if (sdim > nspred && spec->ztgexc != NULL) {
    rc = klein_reorder_argmax(spec->ztgexc, out->s, out->t, out->z, nd, nspred,
                              sdim, z11, z11i, tmp, eye, piv, stage);
  } else {
    rc = klein_z11_pair(out->s, out->z, nd, nspred, z11, z11i, tmp, eye, piv,
                        &rcond);
  }

  /* After any reorder, so eig[i] still names the root in slot i. */
  klein_fill_eig(out->s, out->t, out->eig, nd);

  /* klein_qz emits column-major, the rule step reads row-major. */
  sdsge_transpose_sq(out->s, nd);
  sdsge_transpose_sq(out->t, nd);
  sdsge_transpose_sq(out->z, nd);

  if (rc != SDSGE_KLEIN_OK) {
    return rc;
  }
  out->stab = (sdim > nspred);

  klein_rule_from_schur(out->s, out->t, out->z, nspred, nsfwrd, z11, z11i, gx_c,
                        hx_c, piv, stage);
  sdsge_real_part(gx_c, gx, nsfwrd * nspred);
  sdsge_real_part(hx_c, hx, nspred * nspred);

  /* The dynamic rule in decision-rule order. `gx`'s leading nboth rows are the
   * mixed variables, which `hx` already carries, so only the forward tail is
   * appended below the predetermined block. */
  for (i64 i = 0; i < nspred; ++i) {
    for (i64 j = 0; j < nspred; ++j) {
      ghx[(nstatic + i) * nspred + j] = hx[i * nspred + j];
    }
  }
  for (i64 i = 0; i < nfwd; ++i) {
    for (i64 j = 0; j < nspred; ++j) {
      ghx[(nstatic + nspred + i) * nspred + j] = gx[(nboth + i) * nspred + j];
    }
  }

  /* Dynare's blocks in its own signs: B = -b_rot, A = -c_rot, C = a_rot. */
  if (nstatic > 0) {
    /* work = -C_static @ gx @ hx - A_static: the static rows' own dynamics. */
    for (i64 i = 0; i < nstatic; ++i) {
      for (i64 j = 0; j < nspred; ++j) {
        f64 acc = 0.0;
        for (i64 q = 0; q < nsfwrd; ++q) {
          f64 gh = 0.0;
          for (i64 r = 0; r < nspred; ++r) {
            gh += gx[q * nspred + r] * hx[r * nspred + j];
          }
          acc += a_rot[i * n + ord[nstatic + npred + q]] * gh;
        }
        work[i * nspred + j] = -acc + c_rot[i * n + ord[nstatic + j]];
      }
    }
    /* work -= B over the dynamic columns @ the dynamic rule. */
    for (i64 i = 0; i < nstatic; ++i) {
      for (i64 j = 0; j < nspred; ++j) {
        f64 acc = 0.0;
        for (i64 q = nstatic; q < n; ++q) {
          acc += -b_rot[i * n + ord[q]] * ghx[q * nspred + j];
        }
        work[i * nspred + j] -= acc;
      }
    }
    for (i64 i = 0; i < nstatic; ++i) {
      for (i64 j = 0; j < nstatic; ++j) {
        amat[i * nstatic + j] = -b_rot[i * n + ord[j]];
      }
    }
    if (sdsge_solve(amat, work, nstatic, nspred, ghx) != SDSGE_LU_SUCCESS) {
      return SDSGE_KLEIN_SOLVE_STATIC_SINGULAR;
    }
  }

  /* ghu = A_ \ d_rot, with A_ = [B_static | C@gx + B_pred | B_fyd]. Dynare's
   * -A_ \ fu, with the sign appearing twice: B is -b_rot and fu is -d_rot. */
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < nspred; ++j) {
      f64 acc = 0.0;
      for (i64 q = 0; q < nsfwrd; ++q) {
        acc += a_rot[i * n + ord[nstatic + npred + q]] * gx[q * nspred + j];
      }
      work[i * nspred + j] = acc;
    }
  }
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < nstatic; ++j) {
      amat[i * n + j] = -b_rot[i * n + ord[j]];
    }
    for (i64 j = 0; j < nspred; ++j) {
      amat[i * n + nstatic + j] =
          work[i * nspred + j] - b_rot[i * n + ord[nstatic + j]];
    }
    for (i64 j = 0; j < nfwd; ++j) {
      amat[i * n + nstatic + nspred + j] =
          -b_rot[i * n + ord[nstatic + nspred + j]];
    }
  }
  if (sdsge_solve(amat, d_rot, n, ne, ghu) != SDSGE_LU_SUCCESS) {
    out->stab = SDSGE_KLEIN_STAB_UNSET;
    return SDSGE_KLEIN_SOLVE_SHOCK_SINGULAR;
  }

  /* Scatter decision-rule order back to the canonical layout. Row i is variable
   * ord[i]; column j is state ord[nstatic + j], and the states lead the
   * canonical order, so a state's canonical index is its own column index. */
  for (i64 i = 0; i < n; ++i) {
    const i64 v = ord[i];
    for (i64 j = 0; j < nspred; ++j) {
      const i64 col = ord[nstatic + j];
      if (v < spec->n_state) {
        out->p[v * spec->n_state + col] = ghx[i * nspred + j];
      } else if (spec->n_ctrl > 0) {
        out->f[(v - spec->n_state) * spec->n_state + col] = ghx[i * nspred + j];
      }
    }
    for (i64 j = 0; j < ne; ++j) {
      out->B[v * ne + j] = ghu[i * ne + j];
    }
  }

  sdsge_assemble_transition(out->p, out->f, spec->n_state, spec->n_ctrl,
                            out->A);
  return SDSGE_KLEIN_SOLVE_OK;
}

i64 sdsge_klein_solve1(const klein_spec *spec, sdsge_solve1 *out, f64 *arena,
                       i64 *iarena) {
  i64 rc = sdsge_klein_linearize(spec, out, arena, iarena);
  if (rc != SDSGE_KLEIN_SOLVE_OK) {
    return rc;
  }
  return sdsge_klein_from_pencil(spec, out, arena, iarena);
}

i64 sdsge_sgu_klein_solve2(const sgu_klein_spec *spec, sdsge_solve1 *out1,
                           sdsge_solve2 *out2, f64 *arena, i64 *iarena) {
  const klein_spec *s1 = &spec->first;
  i64 rc = sdsge_klein_solve1(s1, out1, arena, iarena);
  if (rc != SDSGE_KLEIN_SOLVE_OK)
    return rc;

  f64 *stage = arena + sdsge_solve1_fp_reserve(s1->n_state, s1->n_ctrl);
  sdsge_bx_from_B(out1->B, s1->n_state, s1->n_exog, out2->bx);
  sdsge_bicomplex_hessian(spec->bc_residual, out1->ss, s1->params, s1->n_var,
                          s1->n_par, s1->n_exog, s1->n_var, out2->f_xx, stage);
  rc = sdsge_second_order(out1->a_real, out1->b_real, out2->f_xx, out1->f,
                          out1->p, out1->B, spec->Q, s1->n_var, s1->n_state,
                          s1->n_exog, out2->gxx, out2->hxx, out2->gxu,
                          out2->hxu, out2->guu, out2->huu, out2->gss, out2->hss,
                          stage, iarena);

  if (rc != SDSGE_SECOND_ORDER_OK) {
    return rc;
  }
  return SDSGE_KLEIN_SOLVE_OK;
}
