#ifndef SDSGE_COMMON_H
#define SDSGE_COMMON_H

/* Shared low-level definitions for SymbolicDSGE native kernels.
 *
 * Numeric primitives reused across subsystems (matmul, cholesky, triangular
 * solves, dot products) will live alongside this header in sdsge_linalg.{c,h}
 * once the kalman port lands. Everything here is plain C operating on double*
 * buffers -- no CPython or NumPy API. */

/* Portable `restrict` qualifier (C99 `restrict` is not in C++ and is spelled
 * differently by MSVC). */
#if defined(__GNUC__) || defined(__clang__)
#  define SDSGE_RESTRICT __restrict__
#elif defined(_MSC_VER)
#  define SDSGE_RESTRICT __restrict
#else
#  define SDSGE_RESTRICT
#endif

#endif /* SDSGE_COMMON_H */
