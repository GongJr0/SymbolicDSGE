#include "rng.h"
#include "numpy/random/distributions.h"

void sdsge_rng_standard_normal_fill(bitgen_t *bg, i64 n,
                                    f64 *SDSGE_RESTRICT out) {
  if (n <= 0) {
    return;
  }
  random_standard_normal_fill(bg, (npy_intp)n, out);
}

void sdsge_rng_standard_uniform_fill(bitgen_t *bg, i64 n,
                                     f64 *SDSGE_RESTRICT out) {
  if (n <= 0) {
    return;
  }
  random_standard_uniform_fill(bg, (npy_intp)n, out);
}

void sdsge_rng_standard_exponential_fill(bitgen_t *bg, i64 n,
                                         f64 *SDSGE_RESTRICT out) {
  if (n <= 0) {
    return;
  }
  random_standard_exponential_fill(bg, (npy_intp)n, out);
}

void sdsge_rng_standard_gamma_fill(bitgen_t *bg, i64 n, f64 *SDSGE_RESTRICT out,
                                   sdsge_sampler_params params) {
  i64 i;
  if (n <= 0) {
    return;
  }
  for (i = 0; i < n; i++) {
    out[i] = random_standard_gamma(bg, params.a);
  }
}

void sdsge_rng_chi2_fill(bitgen_t *bg, i64 n, f64 *SDSGE_RESTRICT out,
                         sdsge_sampler_params params) {
  if (n <= 0) {
    return;
  }
  for (i64 i = 0; i < n; i++) {
    out[i] = random_chisquare(bg, params.df);
  }
}

void sdsge_rng_beta_fill(bitgen_t *bg, i64 n, f64 *SDSGE_RESTRICT out,
                         sdsge_sampler_params params) {
  if (n <= 0) {
    return;
  }
  for (i64 i = 0; i < n; i++) {
    out[i] = random_beta(bg, params.beta.a, params.beta.b);
  }
}
