#include "pencil.h"

/* 0 static, 1 pred, 2 both, 3 fwd: the group's position in `order`. */
static inline int pencil_class(const i8 inc) {
  const int lag = (inc & SDSGE_INC_LAG) != 0;
  const int lead = (inc & SDSGE_INC_LEAD) != 0;
  return lag ? (lead ? 2 : 1) : (lead ? 3 : 0);
}

i64 sdsge_pencil_dim(const i8 *SDSGE_RESTRICT incidence, const i64 n_var) {
  i64 n_dynamic = 0;
  i64 n_both = 0;
  for (i64 j = 0; j < n_var; ++j) {
    const int cls = pencil_class(incidence[j]);
    if (cls != 0) {
      ++n_dynamic;
    }
    if (cls == 2) {
      ++n_both;
    }
  }
  return n_dynamic + n_both;
}

i64 sdsge_pencil_partition(const i8 *SDSGE_RESTRICT incidence, const i64 n_var,
                           i64 *SDSGE_RESTRICT order,
                           i64 *SDSGE_RESTRICT n_static,
                           i64 *SDSGE_RESTRICT n_pred,
                           i64 *SDSGE_RESTRICT n_both,
                           i64 *SDSGE_RESTRICT n_fwd) {
  i64 count[4] = {0, 0, 0, 0};

  for (i64 j = 0; j < n_var; ++j) {
    if (incidence[j] == 0) {
      return SDSGE_PENCIL_ABSENT_VAR;
    }
    ++count[pencil_class(incidence[j])];
  }

  /* Group starts, then one stable pass appending each variable to its group. */
  i64 at[4];
  at[0] = 0;
  for (int g = 1; g < 4; ++g) {
    at[g] = at[g - 1] + count[g - 1];
  }
  for (i64 j = 0; j < n_var; ++j) {
    order[at[pencil_class(incidence[j])]++] = j;
  }

  *n_static = count[0];
  *n_pred = count[1];
  *n_both = count[2];
  *n_fwd = count[3];
  return SDSGE_PENCIL_OK;
}

void sdsge_pencil_assemble(const f64 *SDSGE_RESTRICT a,
                           const f64 *SDSGE_RESTRICT b,
                           const f64 *SDSGE_RESTRICT c,
                           const i64 *SDSGE_RESTRICT order, const i64 n_var,
                           const i64 n_static, const i64 n_pred,
                           const i64 n_both, const i64 n_fwd,
                           f64 *SDSGE_RESTRICT E, f64 *SDSGE_RESTRICT D) {
  const i64 nspred = n_pred + n_both;
  const i64 nsfwrd = n_both + n_fwd;
  const i64 ndyn = n_pred + n_both + n_fwd;
  const i64 nd = ndyn + n_both;

  for (i64 k = 0; k < nd * nd; ++k) {
    E[k] = 0.0;
    D[k] = 0.0;
  }

  /* Dynare's signs: A = dF/dprev, B = dF/dcur, C = dF/dfwd, against our
   * b = -dF/dcur and c = -dF/dprev. The pencil is homogeneous, so the flip is
   * global and cancels; what matters is that the three agree. */
  for (i64 r = 0; r < ndyn; ++r) {
    const i64 row = n_static + r; /* the rotated dynamic rows */

    /* D, led columns: C[:, both | fwd] at [nspred, nd). */
    for (i64 k = 0; k < nsfwrd; ++k) {
      D[r * nd + nspred + k] = a[row * n_var + order[n_static + n_pred + k]];
    }
    /* D, current columns of the predetermined block: B[:, pred] at [0, n_pred).
     * The `both` slots stay empty here; the companion rows fill them. */
    for (i64 k = 0; k < n_pred; ++k) {
      D[r * nd + k] = -b[row * n_var + order[n_static + k]];
    }

    /* E, lagged columns: -A[:, pred | both] at [0, nspred). */
    for (i64 k = 0; k < nspred; ++k) {
      E[r * nd + k] = c[row * n_var + order[n_static + k]];
    }
    /* E, current columns of the led block: -B[:, both | fwd] at [nspred, nd). */
    for (i64 k = 0; k < nsfwrd; ++k) {
      E[r * nd + nspred + k] = b[row * n_var + order[n_static + n_pred + k]];
    }
  }

  /* Companion rows: y_both at t is the same object in both column blocks, so
   * the identity in each ties the two halves of the linearization together. */
  for (i64 k = 0; k < n_both; ++k) {
    D[(ndyn + k) * nd + n_pred + k] = 1.0;
    E[(ndyn + k) * nd + nspred + k] = 1.0;
  }
}

arena_size sdsge_pencil_rotate_arena_size(const i64 n_var, const i64 n_static,
                                          const i64 n_cols) {
  /* Static column copy, tau, and a work buffer. LAPACK's blocked path wants
   * nb * max(n_cols, n_var); nb is 64 on any build we will meet, and the
   * matrices here are tens of rows, so one flat bound beats a workspace query
   * per solve. */
  const i64 lwork = 64 * (n_cols > n_var ? n_cols : n_var) + n_var;
  return make_sizer(n_var * n_static + n_static + lwork, 0);
}

i64 sdsge_pencil_rotate_static(sdsge_dgeqrf_fn dgeqrf, sdsge_dormqr_fn dormqr,
                               const f64 *SDSGE_RESTRICT b,
                               const i64 *SDSGE_RESTRICT order,
                               const i64 n_var, const i64 n_static,
                               f64 *const *blocks, const i64 *n_cols,
                               const i64 n_block, f64 *SDSGE_RESTRICT arena) {
  if (n_static <= 0) {
    return SDSGE_PENCIL_OK;
  }

  i64 widest = 0;
  for (i64 k = 0; k < n_block; ++k) {
    if (n_cols[k] > widest) {
      widest = n_cols[k];
    }
  }

  f64 *qr = arena;
  f64 *tau = qr + n_var * n_static;
  f64 *work = tau + n_static;
  const int lwork_i =
      (int)(64 * (widest > n_var ? widest : n_var) + n_var);

  /* Column-major copy of the static columns: the LAPACK view of the row-major
   * block would be its transpose, and the QR wanted is of the block itself. */
  for (i64 j = 0; j < n_static; ++j) {
    const i64 col = order[j];
    for (i64 i = 0; i < n_var; ++i) {
      qr[j * n_var + i] = b[i * n_var + col];
    }
  }

  const int m_i = (int)n_var;
  const int k_i = (int)n_static;
  int info = 0;
  dgeqrf(&m_i, &k_i, qr, &m_i, tau, work, &lwork_i, &info);
  if (info != 0) {
    return SDSGE_PENCIL_QR_FAIL;
  }

  const char side = 'R';
  const char trans = 'N';
  for (i64 blk = 0; blk < n_block; ++blk) {
    if (n_cols[blk] == 0) {
      continue;
    }
    /* C is the column-major (n_cols, n_var) view of the row-major block, so
     * C := C @ Q reads back as Q' @ block. */
    const int c_rows = (int)n_cols[blk];
    dormqr(&side, &trans, &c_rows, &m_i, &k_i, qr, &m_i, tau, blocks[blk],
           &c_rows, work, &lwork_i, &info);
    if (info != 0) {
      return SDSGE_PENCIL_QR_FAIL;
    }
  }
  return SDSGE_PENCIL_OK;
}
