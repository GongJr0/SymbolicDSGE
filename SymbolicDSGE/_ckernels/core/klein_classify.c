#include "klein_classify.h"
#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include <math.h>   /* INFINITY */
#include <string.h> /* memcpy */
/* ||A||_1 of a row-major n by n matrix: the largest absolute column sum. The
 * column walk strides a row at a time, which costs nothing at these sizes: the
 * matrix is resident after the first pass, and the two calls sit beside three
 * n^3 products. Accumulating row-major would buy a scratch parameter and the
 * liveness of the caller with it. */
static inline f64 c128_norm1(const c128 *SDSGE_RESTRICT A, const i64 n) {
  f64 best = 0.0;
  for (i64 j = 0; j < n; ++j) {
    f64 sum = 0.0;
    for (i64 i = 0; i < n; ++i) {
      sum += c128_abs(A[i * n + j]);
    }
    if (sum > best) {
      best = sum;
    }
  }
  return best;
}

i64 klein_z11_pair(const c128 *SDSGE_RESTRICT s, const c128 *SDSGE_RESTRICT z,
                   const i64 nd, const i64 n_s, c128 *SDSGE_RESTRICT z11,
                   c128 *SDSGE_RESTRICT z11i, c128 *SDSGE_RESTRICT tmp,
                   const c128 *SDSGE_RESTRICT eye, i64 *SDSGE_RESTRICT piv,
                   f64 *SDSGE_RESTRICT rcond) {
  for (i64 i = 0; i < n_s; ++i) {
    if (!klein_root_finite(s, i, nd)) {
      return SDSGE_KLEIN_INFINITE_ROOT;
    }
  }

  for (i64 i = 0; i < n_s; ++i) {
    for (i64 j = 0; j < n_s; ++j) {
      z11[i * n_s + j] = z[j * nd + i];
    }
  }

  memcpy(tmp, z11, n_s * n_s * sizeof(c128));
  if (c128_lu_factor_inplace(tmp, piv, n_s) != SDSGE_LU_SUCCESS) {
    return SDSGE_KLEIN_RANK_FAIL;
  }
  c128_lu_solve(tmp, piv, eye, z11i, n_s, n_s);

  const f64 r = 1.0 / (c128_norm1(z11, n_s) * c128_norm1(z11i, n_s));
  if (!(r > 1e-9)) {
    return SDSGE_KLEIN_RANK_FAIL;
  }
  *rcond = r;

  return SDSGE_KLEIN_OK;
}

static inline void
klein_triple_copy(c128 *SDSGE_RESTRICT ds, c128 *SDSGE_RESTRICT dt,
                  c128 *SDSGE_RESTRICT dz, const c128 *SDSGE_RESTRICT ss,
                  const c128 *SDSGE_RESTRICT st, const c128 *SDSGE_RESTRICT sz,
                  const i64 sq) {
  const size_t bytes = (size_t)sq * sizeof(c128);
  memcpy(ds, ss, bytes);
  memcpy(dt, st, bytes);
  memcpy(dz, sz, bytes);
}

static inline void klein_pair_copy(c128 *SDSGE_RESTRICT z11,
                                   c128 *SDSGE_RESTRICT z11i,
                                   const c128 *SDSGE_RESTRICT src11,
                                   const c128 *SDSGE_RESTRICT src11i,
                                   const i64 ssq) {
  const size_t bytes = (size_t)ssq * sizeof(c128);
  memcpy(z11, src11, bytes);
  memcpy(z11i, src11i, bytes);
}

/* Move the diagonal block at `ifst` to `ilst`, 1-based, updating Z with it.
 * `info != 0` means ztgex2 gave up partway and left the factors swapped up to
 * wherever it stopped, so the caller restores rather than continuing. */
static int klein_exchange(sdsge_ztgexc_fn ztgexc, c128 *SDSGE_RESTRICT s,
                          c128 *SDSGE_RESTRICT t, c128 *SDSGE_RESTRICT z,
                          const i64 nd, const i64 ifst, const i64 ilst) {
  const int wantq = 0, wantz = 1; /* Q is unread downstream, Z feeds the rule */
  const int n32 = (int)nd, ldq = 1; /* ldq >= 1 even unreferenced */
  const int ifst32 = (int)ifst;
  int ilst32 = (int)ilst; /* [in,out]: LAPACK reports where it landed */
  int info = 0;
  c128 q_dummy = c128_make(0.0, 0.0);

  ztgexc(&wantq, &wantz, &n32, s, &n32, t, &n32, &q_dummy, &ldq, z, &n32,
         &ifst32, &ilst32, &info);
  return info;
}

arena_size klein_classify_arena_size(const i64 nd, const i64 n_s) {
  return make_sizer(2 * (3 * nd * nd + 2 * n_s * n_s), 0);
}

i64 klein_reorder_argmax(sdsge_ztgexc_fn ztgexc, c128 *SDSGE_RESTRICT s,
                         c128 *SDSGE_RESTRICT t, c128 *SDSGE_RESTRICT z,
                         const i64 nd, const i64 n_s, const i64 sdim,
                         c128 *SDSGE_RESTRICT z11, c128 *SDSGE_RESTRICT z11i,
                         c128 *SDSGE_RESTRICT tmp,
                         const c128 *SDSGE_RESTRICT eye,
                         i64 *SDSGE_RESTRICT piv, f64 *SDSGE_RESTRICT arena) {
  const i64 sq = nd * nd;
  const i64 ssq = n_s * n_s;
  c128 *keep_s = (c128 *)arena;
  c128 *keep_t = keep_s + sq;
  c128 *keep_z = keep_t + sq;
  c128 *c_z11 = keep_z + sq;
  c128 *c_z11i = c_z11 + ssq;

  /* The unsearched block is candidate zero: scored like any other so that a
   * baseline nothing beats needs no special case at the exit. */
  f64 best = 0.0;
  i64 rc = klein_z11_pair(s, z, nd, n_s, c_z11, c_z11i, tmp, eye, piv, &best);
  i64 best_k = 0;
  i64 best_j = 0;
  if (rc == SDSGE_KLEIN_OK) {
    klein_pair_copy(z11, z11i, c_z11, c_z11i, ssq);
  }
  klein_triple_copy(keep_s, keep_t, keep_z, s, t, z, sq);

  for (i64 k = 1; k <= n_s; ++k) {
    for (i64 j = n_s + 1; j <= sdim; ++j) {
      klein_triple_copy(s, t, z, keep_s, keep_t, keep_z, sq);
      /* A pair ztgexc declines to swap is an outcome, not an error. */
      if (klein_exchange(ztgexc, s, t, z, nd, k, n_s) != 0 ||
          klein_exchange(ztgexc, s, t, z, nd, j, n_s) != 0) {
        continue;
      }
      f64 r = 0.0;
      if (klein_z11_pair(s, z, nd, n_s, c_z11, c_z11i, tmp, eye, piv, &r) !=
          SDSGE_KLEIN_OK) {
        continue;
      }
      if (r > best) {
        best = r;
        rc = SDSGE_KLEIN_OK;
        best_k = k;
        best_j = j;
        klein_pair_copy(z11, z11i, c_z11, c_z11i, ssq);
      }
    }
  }

  klein_triple_copy(s, t, z, keep_s, keep_t, keep_z, sq);
  if (best_k > 0) {
    klein_exchange(ztgexc, s, t, z, nd, best_k, n_s);
    klein_exchange(ztgexc, s, t, z, nd, best_j, n_s);
  }
  return rc;
}
