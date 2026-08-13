#include "klein_solve.h"
#include "bicomplex_hessian.h" /* sdsge_bicomplex_hessian */
#include "core.h"              /* sdsge_assemble_transition */
#include "klein_postproc.h"    /* klein_postproc */
#include "klein_preproc.h"     /* klein_preproc */
#include "../_common/sdsge_linalg.h" /* sdsge_matmul */
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

/* f64 head reserved for the complex f/p the post-proc emits, which must outlive
 * the post-proc's own scratch: the state-space assembly reads them after it. */
static inline i64 sdsge_solve1_fp_reserve(const i64 n_state, const i64 n_ctrl) {
  return 2 * (n_ctrl * n_state + n_state * n_state);
}

/* Stage max only. The reserve is added once by the public sizers, so it is
 * never folded into a max and then compared against a later stage. */
static inline arena_size sdsge_pencil_stage_arena(const i64 n_var,
                                                  const i64 n_exog,
                                                  const i64 nd) {
  /* nspred and nsfwrd are each at most nd, so the partition's own counts buy
   * nothing here. Held flat rather than maxed: the rotated blocks and the
   * recovered rules coexist across the stage, and these are tens of doubles. */
  const i64 own = 3 * n_var * n_var /* a_rot, b_rot, c_rot */
                  + n_var * n_exog  /* d_rot */
                  + 2 * nd * nd     /* E, D */
                  + 4 * nd * nd     /* complex gx, hx */
                  + 2 * nd * nd     /* real gx, hx */
                  + n_var * nd      /* ghx */
                  + n_var * n_var   /* amat */
                  + n_var * nd      /* C@gx, then the static rhs */
                  + n_var * n_exog; /* ghu */
  const arena_size rot = sdsge_pencil_rotate_arena_size(
      n_var, n_var, n_var > n_exog ? n_var : n_exog);
  arena_size tail = sdsge_max_arena(rot, klein_qz_arena_size(nd));
  tail = sdsge_max_arena(tail, klein_postproc_arena_size(nd, nd));
  return make_sizer(own + tail.n_float, tail.n_int + n_var + nd);
}

static inline arena_size sdsge_solve1_stage_arena(const i64 n_var,
                                                  const i64 n_par,
                                                  const i64 n_exog) {
  arena_size size = sdsge_newton_arena_size(n_var, n_par, n_exog);
  return sdsge_max_arena(size,
                         klein_preproc_arena_size(n_var, n_par, n_exog, n_var));
}

arena_size sdsge_klein_solve1_arena_size(const i64 n_var, const i64 n_state,
                                         const i64 n_ctrl, const i64 n_par,
                                         const i64 n_exog, const i64 nd) {
  arena_size size = sdsge_solve1_stage_arena(n_var, n_par, n_exog);
  size = sdsge_max_arena(size, sdsge_pencil_stage_arena(n_var, n_exog, nd));
  size.n_float += sdsge_solve1_fp_reserve(n_state, n_ctrl);
  return size;
}

arena_size sdsge_sgu_klein_solve2_arena_size(const i64 n_var, const i64 n_state,
                                             const i64 n_ctrl, const i64 n_par,
                                             const i64 n_exog, const i64 nd) {
  arena_size size = sdsge_solve1_stage_arena(n_var, n_par, n_exog);
  size = sdsge_max_arena(size, sdsge_pencil_stage_arena(n_var, n_exog, nd));
  size = sdsge_max_arena(
      size, sdsge_bicomplex_hessian_arena_size(n_var, n_par, n_exog, n_var));
  size = sdsge_max_arena(size, sdsge_second_order_arena_size(n_var, n_state));
  size = sdsge_max_arena(
      size, sdsge_second_order_risk_arena_size(n_var, n_state, n_exog));
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
  const i64 rc = sdsge_steady_state_newton(
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

  if (nspred <= 0) {
    out->stab = SDSGE_KLEIN_STAB_UNSET;
    return SDSGE_KLEIN_SOLVE_NO_STATES;
  }

  f64 *cur = arena + sdsge_solve1_fp_reserve(spec->n_state, spec->n_ctrl);
  f64 *a_rot = cur;
  cur += n * n;
  f64 *b_rot = cur;
  cur += n * n;
  f64 *c_rot = cur;
  cur += n * n;
  f64 *d_rot = cur;
  cur += n * ne;
  f64 *emat = cur;
  cur += nd * nd;
  f64 *dmat = cur;
  cur += nd * nd;
  c128 *gx_c = (c128 *)cur;
  cur += 2 * nsfwrd * nspred;
  c128 *hx_c = (c128 *)cur;
  cur += 2 * nspred * nspred;
  f64 *gx = cur;
  cur += nsfwrd * nspred;
  f64 *hx = cur;
  cur += nspred * nspred;
  f64 *ghx = cur;
  cur += n * nspred;
  f64 *amat = cur;
  cur += n * n;
  f64 *work = cur; /* n by nspred: C@gx, then the static right-hand side */
  cur += n * nspred;
  f64 *ghu = cur;
  cur += n * ne;
  f64 *stage = cur;

  for (i64 k = 0; k < n * n; ++k) {
    a_rot[k] = out->a_real[k];
    b_rot[k] = out->b_real[k];
    c_rot[k] = out->c_real[k];
  }
  for (i64 k = 0; k < n * ne; ++k) {
    d_rot[k] = out->d_real[k];
  }

  /* Rotate the static equations to the top. Every block turns with the same Q
   * so they stay one system; `b` supplies the static columns being cleared. */
  f64 *blocks[4] = {a_rot, b_rot, c_rot, d_rot};
  const i64 widths[4] = {n, n, n, ne};
  if (sdsge_pencil_rotate_static(spec->dgeqrf, spec->dormqr, out->b_real, ord, n,
                                 nstatic, blocks, widths, 4,
                                 stage) != SDSGE_PENCIL_OK) {
    return SDSGE_KLEIN_SOLVE_QR;
  }

  sdsge_pencil_assemble(a_rot, b_rot, c_rot, ord, n, nstatic, npred, nboth, nfwd,
                        emat, dmat);

  sdsge_to_complex_colmajor(dmat, out->s, nd);
  sdsge_to_complex_colmajor(emat, out->t, nd);
  if (klein_qz(spec->zgges, nd, out->s, out->t, out->z, stage, iarena) !=
      KLEIN_QZ_OK) {
    return SDSGE_KLEIN_SOLVE_QZ;
  }

  /* klein_qz emits column-major, klein_postproc reads row-major. */
  sdsge_transpose_sq(out->s, nd);
  sdsge_transpose_sq(out->t, nd);
  sdsge_transpose_sq(out->z, nd);

  switch (klein_postproc(out->s, out->t, out->z, nspred, nsfwrd, gx_c, hx_c,
                         &out->stab, out->eig, stage, iarena)) {
  case SDSGE_KLEIN_POSTPROC_SUCCESS:
    break;
  case SDSGE_KLEIN_POSTPROC_INVALID:
    return SDSGE_KLEIN_SOLVE_NO_STATES;
  default:
    return SDSGE_KLEIN_SOLVE_SINGULAR;
  }
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
      return SDSGE_KLEIN_SOLVE_STATIC;
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
    return SDSGE_KLEIN_SOLVE_SINGULAR;
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
  const i64 rc = sdsge_klein_linearize(spec, out, arena, iarena);
  if (rc != SDSGE_KLEIN_SOLVE_OK) {
    return rc;
  }
  return sdsge_klein_from_pencil(spec, out, arena, iarena);
}

i64 sdsge_sgu_klein_solve2(const sgu_klein_spec *spec, sdsge_solve1 *out1,
                           sdsge_solve2 *out2, f64 *arena, i64 *iarena) {
  /* The SGU system is built on the two-date pencil `a y' = b y`, and the
   * bicomplex sweep spans `(fwd, cur)` only. Neither has seen the lag block or
   * the innovations since the residual gained them, so the tensors below would
   * be taken against a system the model no longer is. Refused here rather than
   * in the Python solve: the native estimation objective reaches this directly
   * and would otherwise walk straight past a caller-side gate. */
  (void)spec;
  (void)out1;
  (void)out2;
  (void)arena;
  (void)iarena;
  return SDSGE_KLEIN_SOLVE_SECOND_ORDER;
}
