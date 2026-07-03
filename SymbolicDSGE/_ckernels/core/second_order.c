#include "second_order.h"
#include "../_common/sdsge_linalg.h"
#include <stdlib.h>
#include <string.h>

/* malloc size guard: never request 0 bytes (implementation-defined). */
static void *so_alloc(size_t count) {
  return malloc((count > 0 ? count : 1) * sizeof(f64));
}

i64 sdsge_second_order(const f64 *SDSGE_RESTRICT a, const f64 *SDSGE_RESTRICT b,
                       const f64 *SDSGE_RESTRICT f_xx,
                       const f64 *SDSGE_RESTRICT gx,
                       const f64 *SDSGE_RESTRICT hx, const i64 n, const i64 nx,
                       f64 *SDSGE_RESTRICT gxx, f64 *SDSGE_RESTRICT hxx) {
  const i64 ny = n - nx;
  const i64 n2 = 2 * n;
  /* stacked-arg block offsets in f_xx: z = [x'; y'; x; y]. */
  const i64 XP = 0, YP = nx, X = n, Y = n + nx;

  const i64 ngxx = ny * nx * nx;
  const i64 nhxx = nx * nx * nx;
  const i64 ncols = n * nx * nx; /* full unknowns: gxx block then hxx block */
  const i64 n_red_g = ny * nx * (nx + 1) / 2;
  const i64 n_red_h = nx * nx * (nx + 1) / 2;
  const i64 R = n_red_g + n_red_h; /* reduced unknowns == number of rows */

  f64 *SDSGE_RESTRICT gxhx = so_alloc((size_t)(ny * nx));   /* gx @ hx  (ny, nx) */
  f64 *SDSGE_RESTRICT hcoef = so_alloc((size_t)(n * nx));   /* fyp@gx + fxp (n, nx) */
  f64 *SDSGE_RESTRICT big_q = so_alloc((size_t)(R * ncols)); /* (R, ncols) */
  f64 *SDSGE_RESTRICT sym = so_alloc((size_t)(ncols * R));   /* (ncols, R) */
  f64 *SDSGE_RESTRICT qt = so_alloc((size_t)(R * R));        /* big_q @ sym */
  f64 *SDSGE_RESTRICT q = so_alloc((size_t)R);
  f64 *SDSGE_RESTRICT xt = so_alloc((size_t)R);
  f64 *SDSGE_RESTRICT full = so_alloc((size_t)ncols);

  i64 status = SDSGE_SECOND_ORDER_OK;

  if (!gxhx || !hcoef || !big_q || !sym || !qt || !q || !xt || !full) {
    status = SDSGE_SECOND_ORDER_ALLOC_FAIL;
    goto done;
  }

  /* big_q starts zeroed: the loop writes only the structural nonzeros (the gxx
   * block is dense, the hxx block is one line per row). */
  memset(big_q, 0, (size_t)(R * ncols) * sizeof(f64));

  /* --- loop-invariant precomputes ---------------------------------------- */
  /* gxhx = gx @ hx  (contiguous inputs). */
  if (ny > 0) {
    sdsge_matmul(gx, hx, gxhx, ny, nx, nx);
  }
  /* hcoef[i, c] = fxp[i, c] + sum_p fyp[i, p] * gx[p, c]  (hxx-block coeff). */
  for (i64 i = 0; i < n; ++i) {
    for (i64 c = 0; c < nx; ++c) {
      f64 s = a[i * n + c]; /* fxp[i, c] */
      for (i64 p = 0; p < ny; ++p) {
        s += a[i * n + nx + p] * gx[p * nx + c]; /* fyp[i,p] * gx[p,c] */
      }
      hcoef[i * nx + c] = s;
    }
  }

  /* --- assemble big_q (R, ncols) and q (R) -------------------------------- */
  {
    i64 m = 0;
    for (i64 i = 0; i < n; ++i) {
      const f64 *SDSGE_RESTRICT fxx_i = f_xx + i * n2 * n2;
      for (i64 j = 0; j < nx; ++j) {
        for (i64 k = 0; k <= j; ++k) {
          f64 *SDSGE_RESTRICT row = big_q + m * ncols;

          /* gxx block: outer product fyp[i,a]*hx[b,j]*hx[c,k] (2nd) + fy (5th). */
          for (i64 aa = 0; aa < ny; ++aa) {
            i64 base = aa * nx * nx;
            f64 fa = a[i * n + nx + aa]; /* fyp[i,aa] */
            for (i64 bb = 0; bb < nx; ++bb) {
              f64 fab = fa * hx[bb * nx + j];
              for (i64 cc = 0; cc < nx; ++cc) {
                row[base + bb * nx + cc] = fab * hx[cc * nx + k];
              }
            }
            row[base + j * nx + k] += -b[i * n + nx + aa]; /* fy[i,aa] */
          }

          /* hxx block: line (., j, k) = hcoef[i, a]  (3rd + 7th). */
          for (i64 aa = 0; aa < nx; ++aa) {
            row[ngxx + aa * nx * nx + j * nx + k] = hcoef[i * nx + aa];
          }

          /* q[m]: scalar accumulation, no buffers. */
          f64 t = fxx_i[(X + j) * n2 + (X + k)]; /* fxx[i,j,k] (8th const) */
          for (i64 cc = 0; cc < ny; ++cc) {
            t += fxx_i[(X + j) * n2 + (YP + cc)] * gxhx[cc * nx + k] /* fxyp */
                 + fxx_i[(X + j) * n2 + (Y + cc)] * gx[cc * nx + k]; /* fxy */
          }
          for (i64 cc = 0; cc < nx; ++cc) {
            t += fxx_i[(X + j) * n2 + (XP + cc)] * hx[cc * nx + k]; /* fxxp */
          }
          for (i64 p = 0; p < ny; ++p) {                     /* 1st + 4th chains */
            f64 c1 = fxx_i[(YP + p) * n2 + (X + k)];          /* fypx[i,p,k] */
            f64 c4 = fxx_i[(Y + p) * n2 + (X + k)];           /* fyx[i,p,k] */
            for (i64 cc = 0; cc < ny; ++cc) {
              c1 += fxx_i[(YP + p) * n2 + (YP + cc)] * gxhx[cc * nx + k]
                    + fxx_i[(YP + p) * n2 + (Y + cc)] * gx[cc * nx + k];
              c4 += fxx_i[(Y + p) * n2 + (YP + cc)] * gxhx[cc * nx + k]
                    + fxx_i[(Y + p) * n2 + (Y + cc)] * gx[cc * nx + k];
            }
            for (i64 cc = 0; cc < nx; ++cc) {
              c1 += fxx_i[(YP + p) * n2 + (XP + cc)] * hx[cc * nx + k];
              c4 += fxx_i[(Y + p) * n2 + (XP + cc)] * hx[cc * nx + k];
            }
            t += c1 * gxhx[p * nx + j] + c4 * gx[p * nx + j];
          }
          for (i64 p = 0; p < nx; ++p) {                     /* 6th chain */
            f64 c6 = fxx_i[(XP + p) * n2 + (X + k)];          /* fxpx[i,p,k] */
            for (i64 cc = 0; cc < ny; ++cc) {
              c6 += fxx_i[(XP + p) * n2 + (YP + cc)] * gxhx[cc * nx + k]
                    + fxx_i[(XP + p) * n2 + (Y + cc)] * gx[cc * nx + k];
            }
            for (i64 cc = 0; cc < nx; ++cc) {
              c6 += fxx_i[(XP + p) * n2 + (XP + cc)] * hx[cc * nx + k];
            }
            t += c6 * hx[p * nx + j];
          }
          q[m] = t;
          ++m;
        }
      }
    }
  }

  /* --- symmetry selector sym (ncols, R): maps reduced -> full ------------- */
  memset(sym, 0, (size_t)(ncols * R) * sizeof(f64));
  {
    i64 my = 0, mx = 0;
    for (i64 kk = 0; kk < nx; ++kk) {
      for (i64 jj = kk; jj < nx; ++jj) {
        for (i64 ii = 0; ii < ny; ++ii) {
          sym[(ii * nx * nx + jj * nx + kk) * R + my] = 1.0;
          sym[(ii * nx * nx + kk * nx + jj) * R + my] = 1.0;
          ++my;
        }
        for (i64 ii = 0; ii < nx; ++ii) {
          sym[(ngxx + ii * nx * nx + jj * nx + kk) * R + (n_red_g + mx)] = 1.0;
          sym[(ngxx + ii * nx * nx + kk * nx + jj) * R + (n_red_g + mx)] = 1.0;
          ++mx;
        }
      }
    }
  }

  /* qt = big_q @ sym  (R, R);  xt = -solve(qt, q);  full = sym @ xt. */
  sdsge_matmul(big_q, sym, qt, R, ncols, R);
  {
    i64 serr = sdsge_solve(qt, q, R, 1, xt);
    if (serr == SDSGE_LU_SINGULAR) {
      status = SDSGE_SECOND_ORDER_SINGULAR;
      goto done;
    }
    if (serr != SDSGE_LU_SUCCESS) {
      status = SDSGE_SECOND_ORDER_ALLOC_FAIL;
      goto done;
    }
  }
  for (i64 r = 0; r < R; ++r) {
    xt[r] = -xt[r];
  }
  sdsge_matvec(sym, xt, full, ncols, R);

  memcpy(gxx, full, (size_t)ngxx * sizeof(f64));
  memcpy(hxx, full + ngxx, (size_t)nhxx * sizeof(f64));

done:
  free(gxhx);
  free(hcoef);
  free(big_q);
  free(sym);
  free(qt);
  free(q);
  free(xt);
  free(full);
  return status;
}

i64 sdsge_second_order_risk(const f64 *SDSGE_RESTRICT a,
                            const f64 *SDSGE_RESTRICT b,
                            const f64 *SDSGE_RESTRICT f_xx,
                            const f64 *SDSGE_RESTRICT gx,
                            const f64 *SDSGE_RESTRICT gxx,
                            const f64 *SDSGE_RESTRICT eta, const i64 n,
                            const i64 nx, const i64 ne, f64 *SDSGE_RESTRICT gss,
                            f64 *SDSGE_RESTRICT hss) {
  const i64 ny = n - nx;
  const i64 n2 = 2 * n;
  const i64 XP = 0, YP = nx; /* only the forward blocks enter */

  f64 *SDSGE_RESTRICT gxe = so_alloc((size_t)(ny * ne)); /* gx @ eta (ny, ne) */
  f64 *SDSGE_RESTRICT g4 = so_alloc((size_t)(nx * nx));  /* fyp[i] . gxx (nx, nx) */
  f64 *SDSGE_RESTRICT coeff = so_alloc((size_t)(n * n)); /* [Qg | Qh] (n, n) */
  f64 *SDSGE_RESTRICT q = so_alloc((size_t)n);
  f64 *SDSGE_RESTRICT x = so_alloc((size_t)n);

  i64 status = SDSGE_SECOND_ORDER_OK;

  if (!gxe || !g4 || !coeff || !q || !x) {
    status = SDSGE_SECOND_ORDER_ALLOC_FAIL;
    goto done;
  }

  /* gxe = gx @ eta (contiguous inputs). */
  if (ny > 0 && ne > 0) {
    sdsge_matmul(gx, eta, gxe, ny, nx, ne);
  }

  for (i64 i = 0; i < n; ++i) {
    const f64 *SDSGE_RESTRICT fxx_i = f_xx + i * n2 * n2;

    /* --- coeff row i: Qg = fyp[i] + fy[i]  |  Qh = fyp[i]@gx + fxp[i] ------ */
    for (i64 aa = 0; aa < ny; ++aa) {
      /* fyp[i,aa] + fy[i,aa], fy = -b. */
      coeff[i * n + aa] = a[i * n + nx + aa] - b[i * n + nx + aa];
    }
    for (i64 c = 0; c < nx; ++c) {
      f64 s = a[i * n + c]; /* fxp[i, c] */
      for (i64 p = 0; p < ny; ++p) {
        s += a[i * n + nx + p] * gx[p * nx + c]; /* fyp[i,p] * gx[p,c] */
      }
      coeff[i * n + ny + c] = s;
    }

    /* --- g4[b,c] = sum_a fyp[i,a] * gxx[a,b,c]  (reused scratch) ---------- */
    for (i64 bb = 0; bb < nx; ++bb) {
      for (i64 cc = 0; cc < nx; ++cc) {
        f64 s = 0.0;
        for (i64 aa = 0; aa < ny; ++aa) {
          s += a[i * n + nx + aa] * gxx[aa * nx * nx + bb * nx + cc];
        }
        g4[bb * nx + cc] = s;
      }
    }

    /* --- q[i]: five trace terms as scalar sum_{p,e} of Hadamard products -- */
    f64 t = 0.0;
    for (i64 p = 0; p < ny; ++p) {
      for (i64 e = 0; e < ne; ++e) {
        f64 in2 = 0.0; /* 2nd: fypyp[i] @ gxe */
        for (i64 c = 0; c < ny; ++c) {
          in2 += fxx_i[(YP + p) * n2 + (YP + c)] * gxe[c * ne + e];
        }
        f64 in3 = 0.0; /* 3rd: fypxp[i] @ eta */
        for (i64 c = 0; c < nx; ++c) {
          in3 += fxx_i[(YP + p) * n2 + (XP + c)] * eta[c * ne + e];
        }
        t += (in2 + in3) * gxe[p * ne + e];
      }
    }
    for (i64 p = 0; p < nx; ++p) {
      for (i64 e = 0; e < ne; ++e) {
        f64 in4 = 0.0; /* 4th: g4 @ eta */
        for (i64 c = 0; c < nx; ++c) {
          in4 += g4[p * nx + c] * eta[c * ne + e];
        }
        f64 in8 = 0.0; /* 8th: fxpyp[i] @ gxe */
        for (i64 c = 0; c < ny; ++c) {
          in8 += fxx_i[(XP + p) * n2 + (YP + c)] * gxe[c * ne + e];
        }
        f64 in9 = 0.0; /* 9th: fxpxp[i] @ eta */
        for (i64 c = 0; c < nx; ++c) {
          in9 += fxx_i[(XP + p) * n2 + (XP + c)] * eta[c * ne + e];
        }
        t += (in4 + in8 + in9) * eta[p * ne + e];
      }
    }
    q[i] = t;
  }

  /* x = -solve(coeff, q); gss = x[:ny], hss = x[ny:]. */
  {
    i64 serr = sdsge_solve(coeff, q, n, 1, x);
    if (serr == SDSGE_LU_SINGULAR) {
      status = SDSGE_SECOND_ORDER_SINGULAR;
      goto done;
    }
    if (serr != SDSGE_LU_SUCCESS) {
      status = SDSGE_SECOND_ORDER_ALLOC_FAIL;
      goto done;
    }
  }
  for (i64 aa = 0; aa < ny; ++aa) {
    gss[aa] = -x[aa];
  }
  for (i64 aa = 0; aa < nx; ++aa) {
    hss[aa] = -x[ny + aa];
  }

done:
  free(gxe);
  free(g4);
  free(coeff);
  free(q);
  free(x);
  return status;
}
