#include "sdsge_complex.h"
#include <math.h>

c128 c128_make(f64 re, f64 im) {
  c128 result;
  result.re = re;
  result.im = im;
  return result;
}

c128 c128_from_real(f64 re) { return c128_make(re, 0.0); }

c128 c128_add(c128 a, c128 b) { return c128_make(a.re + b.re, a.im + b.im); }

c128 c128_sub(c128 a, c128 b) { return c128_make(a.re - b.re, a.im - b.im); }

c128 c128_mul(c128 a, c128 b) {
  return c128_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

c128 c128_div(c128 a, c128 b) {
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

f64 c128_abs(c128 a) { return hypot(a.re, a.im); }

f64 c128_real(c128 a) { return a.re; }

f64 c128_imag(c128 a) { return a.im; }
