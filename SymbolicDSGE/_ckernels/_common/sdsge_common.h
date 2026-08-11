#ifndef SDSGE_COMMON_H
#define SDSGE_COMMON_H

#include <math.h>
#include <stdint.h>

/* Shared low-level definitions for SymbolicDSGE native kernels. Plain C on f64*
 * buffers; no CPython or NumPy API. */

/* Architecture-agnostic numeric types. Use these instead of bare
 * `long`/`int`/`double`: `long` is 32-bit on Windows (LLP64) and 64-bit
 * elsewhere (LP64). All counts and indices are i64. */
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef double f64;

/* Nearest IEEE-754 doubles; TWO_PI is exactly 2.0 * numpy's pi. */
#define TWO_PI 6.283185307179586
#define PI 3.141592653589793
#define SQRT2 1.4142135623730951

/* Status codes for the sdsge_linalg primitives. */
#define SDSGE_OK 0
#define SDSGE_NOT_PD -1

/* Inline min/max for i64 and f64. */
static inline i64 min_i64(i64 a, i64 b) { return (a < b) ? a : b; }
static inline i64 max_i64(i64 a, i64 b) { return (a > b) ? a : b; }
static inline f64 min_f64(f64 a, f64 b) { return (a < b) ? a : b; }
static inline f64 max_f64(f64 a, f64 b) { return (a > b) ? a : b; }

/* float/int arena size descriptor and constructor. */
typedef struct {
  i64 n_float;
  i64 n_int;
} arena_size;

static inline arena_size make_sizer(i64 n_float, i64 n_int) {
  return (arena_size){.n_float = n_float, .n_int = n_int};
}

/* Componentwise max: the stages run one after another off the same arena. */
static inline arena_size sdsge_max_arena(const arena_size a,
                                         const arena_size b) {
  return make_sizer(max_i64(a.n_float, b.n_float), max_i64(a.n_int, b.n_int));
}

/* Flat buffer `any(x[i] == 0)` and `any(x[i] != 0)` checks. Returns 1 if the
 * respective condition is satisfied for any element, 0 otherwise. */
static inline i8 sdsge_any_zero(const f64 *x, i64 n) {
  for (i64 i = 0; i < n; ++i) {
    if (x[i] == 0.0) {
      return 1;
    }
  }
  return 0;
}

static inline i8 sdsge_any_nonzero(const f64 *x, i64 n) {
  for (i64 i = 0; i < n; ++i) {
    if (x[i] != 0.0) {
      return 1;
    }
  }
  return 0;
}
/* Portable `restrict`. */
#if defined(__GNUC__) || defined(__clang__)
#define SDSGE_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define SDSGE_RESTRICT __restrict
#else
#define SDSGE_RESTRICT
#endif

#endif /* SDSGE_COMMON_H */
