#ifndef SDSGE_COMPLEX_H
#define SDSGE_COMPLEX_H
#include "sdsge_common.h"

/* Simple complex number implementation with 64+64 bit representation matching
 * numpy's complex128 type. */

typedef struct {
  f64 re;
  f64 im;
} c128;

typedef struct {
  c128 *lu;
  i64 *piv;
  i64 n;
  i64 err;
} c128_lu;

/* Arithmetic. Defined `static inline` in the header so every translation unit
 * inlines them; math.h (fabs/hypot) comes in via sdsge_common.h. */

static inline c128 c128_make(const f64 re, const f64 im) {
  c128 result;
  result.re = re;
  result.im = im;
  return result;
}

static inline c128 c128_from_real(const f64 re) { return c128_make(re, 0.0); }

static inline c128 c128_add(const c128 a, const c128 b) {
  return c128_make(a.re + b.re, a.im + b.im);
}

static inline c128 c128_sub(const c128 a, const c128 b) {
  return c128_make(a.re - b.re, a.im - b.im);
}

static inline c128 c128_neg(const c128 a) { return c128_make(-a.re, -a.im); }

static inline c128 c128_conj(const c128 a) { return c128_make(a.re, -a.im); }

static inline c128 c128_mul(const c128 a, const c128 b) {
  return c128_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

static inline c128 c128_div(const c128 a, const c128 b) {
  c128 result;
  f64 r;
  if (fabs(b.re) >= fabs(b.im)) {
    r = b.im / b.re;
    result.re = (a.re + a.im * r) / (b.re + b.im * r);
    result.im = (a.im - a.re * r) / (b.re + b.im * r);
  } else {
    r = b.re / b.im;
    result.re = (a.im + a.re * r) / (b.im + b.re * r);
    result.im = (-a.re + a.im * r) / (b.im + b.re * r);
  }
  return result;
}

static inline f64 c128_abs(const c128 a) { return hypot(a.re, a.im); }

static inline f64 c128_abs2(const c128 a) { return a.re * a.re + a.im * a.im; }

static inline f64 c128_real(const c128 a) { return a.re; }

static inline f64 c128_imag(const c128 a) { return a.im; }

/* Linalg */
void c128_matmul(const c128 *SDSGE_RESTRICT A, const c128 *SDSGE_RESTRICT B,
                 const i64 m, const i64 n, const i64 p,
                 c128 *SDSGE_RESTRICT out);

c128_lu c128_lu_factor(const c128 *A, const i64 n);
i64 c128_lu_factor_inplace(c128 *SDSGE_RESTRICT A, i64 *SDSGE_RESTRICT pivot,
                           const i64 n);
void c128_lu_free(c128_lu *lu);

/* Solve (L U) X = B for X(n,m) from a factorization (LU(n,n),
 * piv(n)) produced by c128_lu_factor[_inplace]. B(n,m) is the
 * RHS; X(n,m) is the output and must not alias B/LU. m == 1 is a
 * vector solve; B == identity gives the inverse. */
void c128_lu_solve(const c128 *SDSGE_RESTRICT LU, const i64 *SDSGE_RESTRICT piv,
                   const c128 *SDSGE_RESTRICT B, c128 *SDSGE_RESTRICT X,
                   const i64 n, const i64 m);

i64 c128_solve(const c128 *SDSGE_RESTRICT A, const c128 *SDSGE_RESTRICT B,
               const i64 n, const i64 m, c128 *SDSGE_RESTRICT X);

i64 c128_inv(const c128 *SDSGE_RESTRICT A, const i64 n,
             c128 *SDSGE_RESTRICT Ainv);

/* Second-Order Perturbation Arithmetic */
static inline c128 c128_real_scale(const c128 a, const f64 s) {
  return c128_make(a.re * s, a.im * s);
}

static inline c128 c128_i_mul(const c128 a) {
  return c128_make(-a.im, a.re); // (x, y) = (-y, x)
}

static inline c128 c128_exp(const c128 a) {
  const f64 e = exp(a.re);
  return c128_make(e * cos(a.im), e * sin(a.im));
}

static inline c128 c128_log(const c128 a) {
  return c128_make(log(c128_abs(a)), atan2(a.im, a.re));
}

static inline c128 c128_spow(const c128 a, const f64 p) {
  return c128_exp(
      c128_real_scale(c128_log(a), p)); // Undefined for a <= 0; prefer chained
                                        // multiplication for integer powers.
}

static inline c128 c128_cpow(const c128 a, const c128 b) {
  return c128_exp(c128_mul(b, c128_log(a)));
}

/* ERROR CODES */
#define SDSGE_LU_SUCCESS 0
#define SDSGE_LU_ALLOC_FAIL -1
#define SDSGE_LU_SINGULAR -2

#endif /* SDSGE_COMPLEX_H */
