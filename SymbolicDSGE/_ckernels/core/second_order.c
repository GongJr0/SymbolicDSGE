#include "second_order.h"
#include "../_common/sdsge_linalg.h" /* sdsge_matmul, LU */

/* Second-order perturbation by the chain rule of Juillard and Kamenik (2004),
 * following Dynare's dyn_second_order_solver.m block for block.
 *
 * The Hessian `f_xx` spans z = (lag, cur, lead, eps), so the contraction to
 * state space is one product against zx = dz/dx and zu = dz/du. Nothing is
 * folded into a two-date form, which is what lets a variable entering at all
 * three dates carry its own second-order terms.
 *
 * The lag block of the Jacobian never appears in a coefficient: y_{t-1} is the
 * differentiation variable, so its second derivative is zero. It reaches the
 * result only through zx's identity block. */

/* out(n, cl*cr)[i, p*cr + q] = sum_{u,v} f_xx[i,u,v] Zl[u,p] Zr[v,q].
 * `stage` is nz*cr scratch, reused per equation. */
static void sdsge_contract2(const f64 *SDSGE_RESTRICT f_xx,
                            const f64 *SDSGE_RESTRICT zl, const i64 cl,
                            const f64 *SDSGE_RESTRICT zr, const i64 cr,
                            const i64 n, const i64 nz,
                            f64 *SDSGE_RESTRICT stage,
                            f64 *SDSGE_RESTRICT out) {
  for (i64 i = 0; i < n; ++i) {
    const f64 *SDSGE_RESTRICT fi = f_xx + i * nz * nz;
    for (i64 u = 0; u < nz; ++u) {
      for (i64 q = 0; q < cr; ++q) {
        f64 s = 0.0;
        for (i64 v = 0; v < nz; ++v) {
          s += fi[u * nz + v] * zr[v * cr + q];
        }
        stage[u * cr + q] = s;
      }
    }
    for (i64 p = 0; p < cl; ++p) {
      for (i64 q = 0; q < cr; ++q) {
        f64 s = 0.0;
        for (i64 u = 0; u < nz; ++u) {
          s += zl[u * cl + p] * stage[u * cr + q];
        }
        out[i * cl * cr + p * cr + q] = s;
      }
    }
  }
}

/* out(n, cl*cr)[j, p*cr + q] = sum_{k,l} X[j,k,l] L[k,p] R[l,q], the
 * A_times_B_kronecker_C of the .m. */
static void sdsge_kron_right(const f64 *SDSGE_RESTRICT x,
                             const f64 *SDSGE_RESTRICT l_mat, const i64 nl,
                             const i64 cl, const f64 *SDSGE_RESTRICT r_mat,
                             const i64 nr, const i64 cr, const i64 n,
                             f64 *SDSGE_RESTRICT out) {
  for (i64 j = 0; j < n; ++j) {
    const f64 *SDSGE_RESTRICT xj = x + j * nl * nr;
    for (i64 p = 0; p < cl; ++p) {
      for (i64 q = 0; q < cr; ++q) {
        f64 s = 0.0;
        for (i64 k = 0; k < nl; ++k) {
          const f64 lk = l_mat[k * cl + p];
          if (lk == 0.0) {
            continue;
          }
          for (i64 m = 0; m < nr; ++m) {
            s += xj[k * nr + m] * lk * r_mat[m * cr + q];
          }
        }
        out[j * cl * cr + p * cr + q] = s;
      }
    }
  }
}

arena_size sdsge_second_order_arena_size(const i64 n, const i64 nx,
                                         const i64 ne) {
  const i64 nz = 3 * n + ne;
  const i64 nxx = nx * nx, nxu = nx * ne, nuu = ne * ne;
  const i64 big = n * nxx;
  const i64 wide = nxx > nxu ? (nxx > nuu ? nxx : nuu) : (nxu > nuu ? nxu : nuu);
  return make_sizer(n * nx            /* ghx */
                        + 2 * n * n   /* A, Bm */
                        + n * n       /* LU copy */
                        + nz * nx     /* zx */
                        + 2 * nz * ne /* zu, zlead */
                        + n * nx      /* n-by-nx staging */
                        + nz * wide   /* contraction stage */
                        + 3 * n * wide /* rhs, kron staging, solution */
                        + n * nxx     /* ghxx */
                        + n * nuu     /* ghuu */
                        + big * big   /* sylvester system */
                        + 2 * big     /* its rhs and solution */
                        + n,          /* ghs2 */
                    big > n ? big : n);
}

i64 sdsge_second_order(const f64 *SDSGE_RESTRICT a, const f64 *SDSGE_RESTRICT b,
                       const f64 *SDSGE_RESTRICT f_xx,
                       const f64 *SDSGE_RESTRICT gx, const f64 *SDSGE_RESTRICT hx,
                       const f64 *SDSGE_RESTRICT bu, const f64 *SDSGE_RESTRICT q,
                       const i64 n, const i64 nx, const i64 ne,
                       f64 *SDSGE_RESTRICT gxx, f64 *SDSGE_RESTRICT hxx,
                       f64 *SDSGE_RESTRICT gxu, f64 *SDSGE_RESTRICT hxu,
                       f64 *SDSGE_RESTRICT guu, f64 *SDSGE_RESTRICT huu,
                       f64 *SDSGE_RESTRICT gss, f64 *SDSGE_RESTRICT hss,
                       f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena) {
  const i64 ny = n - nx;
  const i64 nz = 3 * n + ne;
  const i64 nxx = nx * nx, nxu = nx * ne, nuu = ne * ne;
  const i64 big = n * nxx;
  const i64 wide = nxx > nxu ? (nxx > nuu ? nxx : nuu) : (nxu > nuu ? nxu : nuu);
  const i64 lag = 0, cur = n, lead = 2 * n, eps = 3 * n;
  /* The state rows of the impact, which is where a shock has to land to move
   * anything forward. Row-major puts them at the head of bu. */
  const f64 *bx = bu;

  f64 *SDSGE_RESTRICT p = arena;
  f64 *SDSGE_RESTRICT ghx = p;
  p += n * nx;
  f64 *SDSGE_RESTRICT amat = p;
  p += n * n;
  f64 *SDSGE_RESTRICT bmat = p;
  p += n * n;
  f64 *SDSGE_RESTRICT lu = p;
  p += n * n;
  f64 *SDSGE_RESTRICT zx = p;
  p += nz * nx;
  f64 *SDSGE_RESTRICT zu = p;
  p += nz * ne;
  f64 *SDSGE_RESTRICT zlead = p;
  p += nz * ne;
  f64 *SDSGE_RESTRICT nnx = p;
  p += n * nx;
  f64 *SDSGE_RESTRICT stage = p;
  p += nz * wide;
  f64 *SDSGE_RESTRICT rhs = p;
  p += n * wide;
  f64 *SDSGE_RESTRICT kron = p;
  p += n * wide;
  f64 *SDSGE_RESTRICT sol = p;
  p += n * wide;
  f64 *SDSGE_RESTRICT ghxx = p;
  p += n * nxx;
  f64 *SDSGE_RESTRICT ghuu = p;
  p += n * nuu;
  f64 *SDSGE_RESTRICT sys = p;
  p += big * big;
  f64 *SDSGE_RESTRICT sysrhs = p;
  p += big;
  f64 *SDSGE_RESTRICT syssol = p;
  p += big;
  f64 *SDSGE_RESTRICT ghs2 = p;

  /* ghx stacked over the canonical order: the states lead, so hx sits on top of
   * gx and the two are contiguous views of one policy. */
  for (i64 i = 0; i < nx * nx; ++i) {
    ghx[i] = hx[i];
  }
  for (i64 i = 0; i < ny * nx; ++i) {
    ghx[nx * nx + i] = gx[i];
  }

  /* A = dF/dy_t, with the lead's own state dependence folded into the state
   * columns; B = dF/dy_{t+1}. klein_preproc negates the cur sweep. */
  for (i64 i = 0; i < n * n; ++i) {
    amat[i] = -b[i];
    bmat[i] = a[i];
  }
  sdsge_matmul(a, ghx, nnx, n, n, nx);
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < nx; ++j) {
      amat[i * n + j] += nnx[i * nx + j];
    }
  }

  /* zx = dz/dx: the lag block is the state selection, the current block the
   * rule, the lead block the rule applied twice, and the innovations do not
   * move with the state. */
  for (i64 i = 0; i < nz * nx; ++i) {
    zx[i] = 0.0;
  }
  for (i64 i = 0; i < nx; ++i) {
    zx[(lag + i) * nx + i] = 1.0;
  }
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < nx; ++j) {
      zx[(cur + i) * nx + j] = ghx[i * nx + j];
    }
  }
  sdsge_matmul(ghx, hx, nnx, n, nx, nx);
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < nx; ++j) {
      zx[(lead + i) * nx + j] = nnx[i * nx + j];
    }
  }

  /* zu = dz/du: nothing lagged moves, the current block is the impact, the lead
   * block carries it forward, and the innovation block is the identity. */
  for (i64 i = 0; i < nz * ne; ++i) {
    zu[i] = 0.0;
    zlead[i] = 0.0;
  }
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < ne; ++j) {
      zu[(cur + i) * ne + j] = bu[i * ne + j];
      zlead[(lead + i) * ne + j] = bu[i * ne + j];
    }
  }
  sdsge_matmul(ghx, bx, rhs, n, nx, ne); /* ghx @ bx, staged in rhs */
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < ne; ++j) {
      zu[(lead + i) * ne + j] = rhs[i * ne + j];
    }
  }
  for (i64 i = 0; i < ne; ++i) {
    zu[(eps + i) * ne + i] = 1.0;
  }

  /* --- ghxx: A X + B X (hx (x) hx) = -f_xx (zx (x) zx) ---------------------
   * Solved flat rather than by gensylv: at these dimensions the system is a few
   * dozen rows, and the Kronecker structure buys nothing back. */
  sdsge_contract2(f_xx, zx, nx, zx, nx, n, nz, stage, rhs);
  for (i64 i = 0; i < big; ++i) {
    sysrhs[i] = -rhs[i];
  }
  for (i64 i = 0; i < n; ++i) {
    for (i64 k = 0; k < nx; ++k) {
      for (i64 l = 0; l < nx; ++l) {
        const i64 row = (i * nx + k) * nx + l;
        for (i64 j = 0; j < n; ++j) {
          for (i64 k2 = 0; k2 < nx; ++k2) {
            for (i64 l2 = 0; l2 < nx; ++l2) {
              const i64 col = (j * nx + k2) * nx + l2;
              f64 v = bmat[i * n + j] * hx[k2 * nx + k] * hx[l2 * nx + l];
              if (k2 == k && l2 == l) {
                v += amat[i * n + j];
              }
              sys[row * big + col] = v;
            }
          }
        }
      }
    }
  }
  if (sdsge_lu_factor_inplace(sys, iarena, big) != SDSGE_LU_SUCCESS) {
    return SDSGE_SECOND_ORDER_SINGULAR;
  }
  sdsge_lu_solve(sys, iarena, sysrhs, syssol, big, 1);
  for (i64 i = 0; i < big; ++i) {
    ghxx[i] = syssol[i];
  }

  /* A is the coefficient of every remaining block, so it factors once. */
  for (i64 i = 0; i < n * n; ++i) {
    lu[i] = amat[i];
  }
  if (sdsge_lu_factor_inplace(lu, iarena, n) != SDSGE_LU_SUCCESS) {
    return SDSGE_SECOND_ORDER_SINGULAR;
  }

  /* --- ghxu: A X = -f_xx (zx (x) zu) - B ghxx (hx (x) bu) ------------------ */
  sdsge_contract2(f_xx, zx, nx, zu, ne, n, nz, stage, rhs);
  sdsge_kron_right(ghxx, hx, nx, nx, bx, nx, ne, n, kron);
  for (i64 i = 0; i < n; ++i) {
    for (i64 col = 0; col < nxu; ++col) {
      f64 s = 0.0;
      for (i64 j = 0; j < n; ++j) {
        s += bmat[i * n + j] * kron[j * nxu + col];
      }
      rhs[i * nxu + col] = -rhs[i * nxu + col] - s;
    }
  }
  sdsge_lu_solve(lu, iarena, rhs, sol, n, nxu);
  for (i64 i = 0; i < nx * nxu; ++i) {
    hxu[i] = sol[i];
  }
  for (i64 i = 0; i < ny * nxu; ++i) {
    gxu[i] = sol[nx * nxu + i];
  }

  /* --- ghuu: A X = -f_xx (zu (x) zu) - B ghxx (bu (x) bu) ------------------ */
  sdsge_contract2(f_xx, zu, ne, zu, ne, n, nz, stage, rhs);
  sdsge_kron_right(ghxx, bx, nx, ne, bx, nx, ne, n, kron);
  for (i64 i = 0; i < n; ++i) {
    for (i64 col = 0; col < nuu; ++col) {
      f64 s = 0.0;
      for (i64 j = 0; j < n; ++j) {
        s += bmat[i * n + j] * kron[j * nuu + col];
      }
      rhs[i * nuu + col] = -rhs[i * nuu + col] - s;
    }
  }
  sdsge_lu_solve(lu, iarena, rhs, sol, n, nuu);
  for (i64 i = 0; i < n * nuu; ++i) {
    ghuu[i] = sol[i];
  }
  for (i64 i = 0; i < nx * nuu; ++i) {
    huu[i] = sol[i];
  }
  for (i64 i = 0; i < ny * nuu; ++i) {
    guu[i] = sol[nx * nuu + i];
  }

  /* --- ghs2: (A + B) X = -(B ghuu + f_xx (zlead (x) zlead)) vec(Q) ---------
   * The risk correction is the only block that reads the covariance rather than
   * its factor, and only the lead block of the Hessian enters: the term is the
   * expectation of next period's innovation. */
  sdsge_contract2(f_xx, zlead, ne, zlead, ne, n, nz, stage, kron);
  for (i64 i = 0; i < n; ++i) {
    f64 acc = 0.0;
    for (i64 col = 0; col < nuu; ++col) {
      f64 s = kron[i * nuu + col];
      for (i64 j = 0; j < n; ++j) {
        s += bmat[i * n + j] * ghuu[j * nuu + col];
      }
      acc += s * q[col];
    }
    rhs[i] = -acc;
  }
  for (i64 i = 0; i < n * n; ++i) {
    lu[i] = amat[i] + bmat[i];
  }
  if (sdsge_lu_factor_inplace(lu, iarena, n) != SDSGE_LU_SUCCESS) {
    return SDSGE_SECOND_ORDER_RISK;
  }
  sdsge_lu_solve(lu, iarena, rhs, ghs2, n, 1);
  for (i64 i = 0; i < nx; ++i) {
    hss[i] = ghs2[i];
  }
  for (i64 i = 0; i < ny; ++i) {
    gss[i] = ghs2[nx + i];
  }

  /* The split is a row cut at n_state: the states lead the canonical order. */
  for (i64 i = 0; i < nx * nxx; ++i) {
    hxx[i] = ghxx[i];
  }
  for (i64 i = 0; i < ny * nxx; ++i) {
    gxx[i] = ghxx[nx * nxx + i];
  }
  return SDSGE_SECOND_ORDER_OK;
}
