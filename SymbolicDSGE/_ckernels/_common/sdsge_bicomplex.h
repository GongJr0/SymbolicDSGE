#ifndef SDSGE_BICOMPLEX_H
#define SDSGE_BICOMPLEX_H

#include "sdsge_common.h"
#include "sdsge_complex.h"

typedef struct {
  c128 a;
  c128 b;
} bc256;

static inline bc256 bc256_make(const c128 a, const c128 b) {
  bc256 result;
  result.a = a;
  result.b = b;
  return result;
}

static inline f64 bc256_real(const bc256 x) { return x.a.re; }
static inline f64 bc256_i(const bc256 x) { return x.a.im; }
static inline f64 bc256_j(const bc256 x) { return x.b.re; }
static inline f64 bc256_ij(const bc256 x) { return x.b.im; }

static inline bc256 bc256_from_real(const f64 re) {
  return bc256_make(c128_from_real(re), c128_from_real(0.0));
}

static inline bc256 bc256_from_complex(const c128 a) {
  return bc256_make(a, c128_from_real(0.0));
}

static inline bc256 bc256_add(const bc256 x, const bc256 y) {
  return bc256_make(c128_add(x.a, y.a), c128_add(x.b, y.b));
}

static inline bc256 bc256_sub(const bc256 x, const bc256 y) {
  return bc256_make(c128_sub(x.a, y.a), c128_sub(x.b, y.b));
}

static inline bc256 bc256_neg(const bc256 x) {
  return bc256_make(c128_neg(x.a), c128_neg(x.b));
}

static inline bc256 bc256_i_conj(const bc256 x) {
  return bc256_make(c128_conj(x.a), c128_conj(x.b));
}

static inline bc256 bc256_j_conj(const bc256 x) {
  return bc256_make(x.a, c128_neg(x.b));
}

static inline bc256 bc256_conj(const bc256 x) {
  return bc256_make(c128_conj(x.a), c128_neg(c128_conj(x.b)));
}

static inline bc256 bc256_mul(const bc256 x, const bc256 y) {
  return bc256_make(c128_sub(c128_mul(x.a, y.a), c128_mul(x.b, y.b)),
                    c128_add(c128_mul(x.a, y.b), c128_mul(x.b, y.a)));
}

static inline bc256 bc256_div(const bc256 x, const bc256 y) {
  c128 denom = c128_add(c128_mul(y.a, y.a), c128_mul(y.b, y.b));
  return bc256_make(
      c128_div(c128_add(c128_mul(x.a, y.a), c128_mul(x.b, y.b)), denom),
      c128_div(c128_sub(c128_mul(x.b, y.a), c128_mul(x.a, y.b)), denom));
}

static inline bc256 bc256_real_scale(const bc256 x, const f64 s) {
  return bc256_make(c128_real_scale(x.a, s), c128_real_scale(x.b, s));
}

static inline void bc256_proj(const bc256 x, c128 *SDSGE_RESTRICT p1,
                              c128 *SDSGE_RESTRICT p2) {
  const c128 iz2 = c128_i_mul(x.b);
  *p1 = c128_sub(x.a, iz2);
  *p2 = c128_add(x.a, iz2);
}

static inline bc256 bc256_reconst(const c128 a, const c128 b) {
  bc256 result;
  result.a = c128_real_scale(c128_add(a, b), 0.5);
  result.b = c128_real_scale(c128_i_mul(c128_sub(a, b)), 0.5);
  return result;
}

static inline bc256 bc256_exp(const bc256 x) {
  c128 p1, p2;
  bc256_proj(x, &p1, &p2);
  return bc256_reconst(c128_exp(p1), c128_exp(p2));
}

static inline bc256 bc256_log(const bc256 x) {
  c128 p1, p2;
  bc256_proj(x, &p1, &p2);
  return bc256_reconst(c128_log(p1), c128_log(p2));
}

static inline bc256 bc256_spow(const bc256 x, const f64 p) {
  c128 p1, p2;
  bc256_proj(x, &p1, &p2);
  return bc256_reconst(
      c128_spow(p1, p),
      c128_spow(p2, p)); // Undefined for x <= 0; prefer chained multiplication
                         // for integer powers.
}

static inline bc256 bc256_cpow(const bc256 x, const bc256 y) {
  c128 px1, px2, py1, py2;
  bc256_proj(x, &px1, &px2);
  bc256_proj(y, &py1, &py2);
  return bc256_reconst(c128_cpow(px1, py1), c128_cpow(px2, py2));
}

#endif /* SDSGE_BICOMPLEX_H */
