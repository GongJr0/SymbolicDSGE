#include "shocks.h"
#include "../rng/philox.h"
#include <stddef.h>

/* --- shared pieces -------------------------------------------------------- */

/* A replication's stream is selected purely by (entry, rep_idx, stream), never
 * by accumulated state, so the draw is identical under any worker schedule. */
static inline void sdsge_shock_seed(sdsge_philox_state *st,
                                    const sdsge_shock_entry *entry,
                                    const i64 rep_idx, const u64 stream) {

  /* entry->columns is the sorted canonical indices of each shock variable in
   * the group. A variable cannot appear in more than one group, therefore
   * entry->columns[0] is a unique identifier for the group. */
  sdsge_philox_seed(st, entry->key, (u64)entry->columns[0], (u64)rep_idx,
                    stream);
}

/* loc + factor @ v, scattered into the entry's columns, for any family whose
 * variate arrives standardized. */
static void sdsge_shock_apply_affine(const sdsge_shock_plan *plan,
                                     const sdsge_shock_entry *entry,
                                     const f64 *SDSGE_RESTRICT v,
                                     f64 *SDSGE_RESTRICT out) {
  const i64 width = entry->width;
  const i64 n_exog = plan->n_exog;
  const i64 T = plan->T;
  const f64 shock_scale = plan->shock_scale;

  for (i64 t = 0; t < T; ++t) {
    const f64 *SDSGE_RESTRICT v_t = v + t * width;
    f64 *SDSGE_RESTRICT out_t = out + t * n_exog;
    for (i64 i = 0; i < width; ++i) {
      const f64 *SDSGE_RESTRICT factor_row = entry->factor + i * width;
      f64 acc = (entry->loc == NULL) ? 0.0 : entry->loc[i];
      /* factor is lower-triangular on the Cholesky path, but the eigh
       * fallback for a semidefinite covariance is dense, so sum the full row.
       */
      for (i64 j = 0; j < width; ++j) {
        acc += factor_row[j] * v_t[j];
      }
      out_t[entry->columns[i]] = shock_scale * acc;
    }
  }
}

/* --- families ------------------------------------------------------------- */

static void sdsge_shock_draw_normal(const sdsge_shock_plan *plan,
                                    const sdsge_shock_entry *entry,
                                    const i64 rep_idx,
                                    f64 *SDSGE_RESTRICT scratch,
                                    f64 *SDSGE_RESTRICT out) {
  sdsge_philox_state st;

  sdsge_shock_seed(&st, entry, rep_idx, 0);
  sdsge_philox_standard_normal_fill(&st, plan->T * entry->width, scratch);
  sdsge_shock_apply_affine(plan, entry, scratch, out);
}

/* Uniform is univariate by construction, and its standardization is a special
 * case relative to multivariate-supporting distributions. Inlined here.
 *
 * Each draw is transformed on its own, so it goes straight to its strided
 * destination and the family stages nothing. */
static void sdsge_shock_draw_uniform(const sdsge_shock_plan *plan,
                                     const sdsge_shock_entry *entry,
                                     const i64 rep_idx,
                                     f64 *SDSGE_RESTRICT scratch,
                                     f64 *SDSGE_RESTRICT out) {
  const i64 n_exog = plan->n_exog;
  const i64 T = plan->T;
  const i64 column = entry->columns[0];
  const f64 shock_scale = plan->shock_scale;
  const f64 sqrt3 = sqrt(3.0);
  const f64 loc = (entry->loc == NULL) ? 0.0 : entry->loc[0];
  const f64 lo = loc - sqrt3 * entry->factor[0];
  const f64 sc = 2.0 * sqrt3 * entry->factor[0];
  sdsge_philox_state st;

  (void)scratch;
  sdsge_shock_seed(&st, entry, rep_idx, 0);
  for (i64 t = 0; t < T; ++t) {
    out[t * n_exog + column] =
        shock_scale * (lo + sc * sdsge_philox_next_double(&st));
  }
}

static void sdsge_shock_draw_student_t(const sdsge_shock_plan *plan,
                                       const sdsge_shock_entry *entry,
                                       const i64 rep_idx,
                                       f64 *SDSGE_RESTRICT scratch,
                                       f64 *SDSGE_RESTRICT out) {
  const i64 width = entry->width;
  const i64 T = plan->T;
  const f64 df = entry->params.df;
  f64 *SDSGE_RESTRICT v = scratch;
  f64 *SDSGE_RESTRICT g = scratch + T * width;
  sdsge_philox_state st;

  sdsge_shock_seed(&st, entry, rep_idx, 0);
  sdsge_philox_standard_normal_fill(&st, T * width, v);
  sdsge_shock_seed(&st, entry, rep_idx, 1);
  sdsge_philox_chi2_fill(&st, T, g, entry->params);

  for (i64 t = 0; t < T; ++t) {
    const f64 s = sqrt((df - 2.0) / g[t]);
    f64 *SDSGE_RESTRICT v_t = v + t * width;
    for (i64 i = 0; i < width; ++i) {
      v_t[i] *= s;
    }
  }
  sdsge_shock_apply_affine(plan, entry, v, out);
}

static void sdsge_shock_draw_exponential(const sdsge_shock_plan *plan,
                                         const sdsge_shock_entry *entry,
                                         const i64 rep_idx,
                                         f64 *SDSGE_RESTRICT scratch,
                                         f64 *SDSGE_RESTRICT out) {
  f64 *SDSGE_RESTRICT v = scratch;
  sdsge_philox_state st;

  sdsge_shock_seed(&st, entry, rep_idx, 0);
  sdsge_philox_standard_exponential_fill(&st, plan->T, v);
  for (i64 t = 0; t < plan->T; ++t) {
    v[t] -= 1.0; /* center at zero */
  }
  sdsge_shock_apply_affine(plan, entry, v, out);
}

static void sdsge_shock_draw_gamma(const sdsge_shock_plan *plan,
                                   const sdsge_shock_entry *entry,
                                   const i64 rep_idx,
                                   f64 *SDSGE_RESTRICT scratch,
                                   f64 *SDSGE_RESTRICT out) {
  f64 *SDSGE_RESTRICT v = scratch;
  sdsge_philox_state st;

  sdsge_shock_seed(&st, entry, rep_idx, 0);
  sdsge_philox_standard_gamma_fill(&st, plan->T, v, entry->params);
  const f64 a = entry->params.a;
  const f64 invsq_a = 1.0 / sqrt(a);
  for (i64 t = 0; t < plan->T; ++t) {
    v[t] = (v[t] - a) * invsq_a; /* center at zero, scale to var 1 */
  }
  sdsge_shock_apply_affine(plan, entry, v, out);
}

/* Indexed by `native_shock`, in enum order. Draws only: what a family spends is
 * stated in `sdsge_shock_entry_arena_size`, which no caller of this table reads
 * and which runs once per entry at plan time rather than per replication. */
static const sdsge_shock_draw_fn SDSGE_SHOCK_DRAW[] = {
    [SDSGE_SHOCK_NORMAL] = sdsge_shock_draw_normal,
    [SDSGE_SHOCK_UNIFORM] = sdsge_shock_draw_uniform,
    [SDSGE_SHOCK_STUDENT_T] = sdsge_shock_draw_student_t,
    [SDSGE_SHOCK_EXPONENTIAL] = sdsge_shock_draw_exponential,
    [SDSGE_SHOCK_GAMMA] = sdsge_shock_draw_gamma,
};

static void sdsge_shock_apply_path(const sdsge_shock_plan *plan,
                                   const sdsge_shock_entry *entry,
                                   f64 *SDSGE_RESTRICT out) {
  const i64 n_exog = plan->n_exog;
  const i64 width = entry->width;
  const f64 shock_scale = plan->shock_scale;

  for (i64 t = 0; t < plan->T; ++t) {
    for (i64 j = 0; j < width; ++j) {
      out[t * n_exog + entry->columns[j]] =
          shock_scale * entry->path[t * width + j];
    }
  }
}

/* --- entry points --------------------------------------------------------- */

arena_size sdsge_shock_entry_arena_size(const native_shock family,
                                        const i64 width, const i64 T) {
  switch (family) {
  case SDSGE_SHOCK_PATH:
    /* A path is read from `path` and needs no scratch. */
    return make_sizer(0, 0);
  case SDSGE_SHOCK_NORMAL:
    return make_sizer(T * width, 0);
  case SDSGE_SHOCK_UNIFORM:
    /* Transformed one draw at a time into `out`, staging nothing. */
    return make_sizer(0, 0);
  case SDSGE_SHOCK_STUDENT_T:
    /* The Gaussian core, plus the chi-square it is divided by. */
    return make_sizer(T * (width + 1), 0);
  case SDSGE_SHOCK_EXPONENTIAL:
    return make_sizer(T, 0);
  case SDSGE_SHOCK_GAMMA:
    return make_sizer(T, 0);
  }
}

arena_size sdsge_shock_plan_arena_size(const sdsge_shock_plan *plan) {
  arena_size total = make_sizer(0, 0);

  for (i64 i = 0; i < plan->n_entries; ++i) {
    const sdsge_shock_entry *entry = &plan->entries[i];
    total = sdsge_max_arena(total, sdsge_shock_entry_arena_size(
                                       entry->family, entry->width, plan->T));
  }
  return total;
}

void sdsge_shock_draw(const sdsge_shock_plan *plan, const i64 rep_idx,
                      f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out) {
  for (i64 i = 0; i < plan->n_entries; ++i) {
    const sdsge_shock_entry *entry = &plan->entries[i];

    if (entry->family == SDSGE_SHOCK_PATH) {
      sdsge_shock_apply_path(plan, entry, out);
      continue;
    }
    SDSGE_SHOCK_DRAW[entry->family](plan, entry, rep_idx, scratch, out);
  }
}
