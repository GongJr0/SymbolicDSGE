#ifndef SDSGE_COMMON_H
#define SDSGE_COMMON_H

#include <math.h>
#include <stdint.h>

/* Shared low-level definitions for SymbolicDSGE native kernels.
 *
 * Numeric primitives reused across subsystems (matmul, cholesky, triangular
 * solves, dot products) will live alongside this header in sdsge_linalg.{c,h}
 * once the kalman port lands. Everything here is plain C operating on f64*
 * buffers -- no CPython or NumPy API. */

/* Architecture-agnostic numeric types. Use these everywhere instead of bare
 * `long`/`int`/`double` so width and indexing semantics are identical across
 * the whole wheel matrix -- notably `long` is 32-bit on Windows (LLP64) but
 * 64-bit on Linux/macOS (LP64). All counts and indices are i64. */
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

/* 2*pi as the nearest IEEE-754 double (== 2.0 * numpy's pi, exactly); shared by
 * the kalman / distribution / transform kernels. */
#define TWO_PI 6.283185307179586

/* Portable `restrict` qualifier (C99 `restrict` is not in C++ and is spelled
 * differently by MSVC). */
#if defined(__GNUC__) || defined(__clang__)
#define SDSGE_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define SDSGE_RESTRICT __restrict
#else
#define SDSGE_RESTRICT
#endif

#endif /* SDSGE_COMMON_H */
