#include "shocks.h"
#include <stddef.h>

/* Direct includes (not transitive via shocks.h): the Philox state and its fills
 * are declared in philox.h. Native-include hygiene wants the header that
 * declares each used symbol included at its use site. */
#include "../rng/philox.h"

i64 sdsge_mc_shock_scratch_size(const sdsge_mc_shock_plan *plan) {
  return plan->T * plan->max_width;
}

/* A replication's stream is selected purely by (entry, rep_idx), never by
 * accumulated state, so the draw is identical under any worker schedule. */
static inline void sdsge_mc_shock_seed(sdsge_philox_state *st,
                                       const sdsge_mc_shock_entry *entry,
                                       i64 rep_idx) {
  sdsge_philox_seed(st, entry->key, entry->entry_idx, (u64)rep_idx, 0);
}

/* z @ factor.T, scattered into the entry's columns. Width is the number of
 * exogenous variables one entry drives, so it is small (typically 1 to 3) and
 * the straightforward loop beats any blocking. */
static void sdsge_mc_shock_apply_normal(const sdsge_mc_shock_plan *plan,
                                        const sdsge_mc_shock_entry *entry,
                                        const f64 *SDSGE_RESTRICT z,
                                        f64 *SDSGE_RESTRICT out) {
  const i64 width = entry->width;
  const i64 n_exog = plan->n_exog;
  const f64 shock_scale = plan->shock_scale;
  i64 t;

  for (t = 0; t < plan->T; t++) {
    const f64 *SDSGE_RESTRICT z_t = z + t * width;
    f64 *SDSGE_RESTRICT out_t = out + t * n_exog;
    i64 i;
    for (i = 0; i < width; i++) {
      const f64 *SDSGE_RESTRICT factor_row = entry->factor + i * width;
      f64 acc = (entry->loc == NULL) ? 0.0 : entry->loc[i];
      i64 j;
      /* factor is lower-triangular on the Cholesky path, but the eigh
       * fallback for a semidefinite covariance is dense, so sum the full row.
       */
      for (j = 0; j < width; j++) {
        acc += factor_row[j] * z_t[j];
      }
      out_t[entry->columns[i]] = shock_scale * acc;
    }
  }
}

static void sdsge_mc_shock_apply_uniform(const sdsge_mc_shock_plan *plan,
                                         const sdsge_mc_shock_entry *entry,
                                         const f64 *SDSGE_RESTRICT u,
                                         f64 *SDSGE_RESTRICT out) {
  const i64 n_exog = plan->n_exog;
  const i64 column = entry->columns[0];
  const f64 shock_scale = plan->shock_scale;
  i64 t;

  for (t = 0; t < plan->T; t++) {
    out[t * n_exog + column] = shock_scale * (entry->low + entry->span * u[t]);
  }
}

void sdsge_mc_shock_draw(const sdsge_mc_shock_plan *plan, const i64 rep_idx,
                         f64 *SDSGE_RESTRICT scratch, f64 *SDSGE_RESTRICT out) {
  const i64 total = plan->T * plan->n_exog;
  sdsge_philox_state st;
  i64 i;

  /* An exogenous variable no entry targets stays at zero, matching the Python
   * route, which fills only the columns its spec names. */
  for (i = 0; i < total; i++) {
    out[i] = 0.0;
  }

  for (i = 0; i < plan->n_entries; i++) {
    const sdsge_mc_shock_entry *entry = &plan->entries[i];
    sdsge_mc_shock_seed(&st, entry, rep_idx);

    if (entry->family == SDSGE_MC_SHOCK_UNIFORM) {
      sdsge_philox_standard_uniform_fill(&st, plan->T, scratch);
      sdsge_mc_shock_apply_uniform(plan, entry, scratch, out);
      continue;
    }

    sdsge_philox_standard_normal_fill(&st, plan->T * entry->width, scratch);
    sdsge_mc_shock_apply_normal(plan, entry, scratch, out);
  }
}
