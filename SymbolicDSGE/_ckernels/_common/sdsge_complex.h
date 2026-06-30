#ifndef SDSGE_COMPLEX_H
#define SDSGE_COMPLEX_H
#include "sdsge_common.h"

/* Simple complex number implementation with 64+64 bit representation matching
 * numpy's complex128 type. */

typedef struct {
  f64 re;
  f64 im;
} c128;

c128 c128_make(f64 re, f64 im);
c128 c128_from_real(f64 re);
c128 c128_add(c128 a, c128 b);
c128 c128_sub(c128 a, c128 b);
c128 c128_mul(c128 a, c128 b);
c128 c128_div(c128 a, c128 b);
f64 c128_abs(c128 a);
f64 c128_real(c128 a);
f64 c128_imag(c128 a);

#endif /* SDSGE_COMPLEX_H */
