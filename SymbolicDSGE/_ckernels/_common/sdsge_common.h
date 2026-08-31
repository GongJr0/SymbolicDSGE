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

/* Status conventions, shared by every kernel in the tree.
 *
 * Zero is success, and it is the only success. Positive is a non-error outcome
 * that the caller has to branch on (an iteration cap, a fallback, a halt).
 * Negative is a failure, and only failures are numbered.
 *
 * A failure code is `-(module * 100 + code)`, where the module is a code family
 * (the header that owns it, not the directory) and the code counts from 1
 * within it. Families are disjoint, so a code carries its own provenance:
 * forwarding one verbatim through a caller in another module stays unambiguous,
 * and a message can never be attributed to the wrong kernel. The magnitude
 * reads as its family, so `-5xx` is klein_solve and `-9xx` is steady_state. `rc
 * < 0` remains a valid failure test without any table.
 *
 * A module number is assigned once and never reused, and a new family takes the
 * next free one. Reusing a retired number would make an old code readable as a
 * new family, which is the failure this scheme exists to prevent.
 *
 *   1 sdsge_common      2 core             3 klein_postproc   4 klein_qz
 *   5 klein_solve       6 pencil           7 residual_path    8 second_order
 *   9 steady_state     10 diag            11 kalman          12 mc_runner
 *  13 mc_transforms    14 occbin          15 regression      16 mcmc
 *  17 optim
 *
 * Codes are only for failures. A positive status (an iteration cap, a fallback,
 * a halt, occbin's Dynare-mapped outcomes) stays outside the scheme, as does
 * data that happens to be an int: the SDSGE_INC_* incidence bits, the prior
 * distribution and transform enums, the regression criteria. */
#define SDSGE_OK 0
#define SDSGE_NOT_PD -101

/* LU status, shared by the real (sdsge_linalg.h) and complex (sdsge_complex.h)
 * factorizations: one failure vocabulary for one algorithm. */
#define SDSGE_LU_SUCCESS 0
#define SDSGE_LU_ALLOC_FAIL -102
#define SDSGE_LU_SINGULAR -103

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

static inline arena_size add_arena(const arena_size a, const arena_size b) {
  return make_sizer(a.n_float + b.n_float, a.n_int + b.n_int);
}

/* Componentwise max: the stages run one after another off the same arena. */
static inline arena_size sdsge_max_arena(const arena_size a,
                                         const arena_size b) {
  return make_sizer(max_i64(a.n_float, b.n_float), max_i64(a.n_int, b.n_int));
}

/* Widest lane a layout may describe. Fixed so the descriptor below carries its
 * entries inline; a lane that outgrows the cap fails the build where its buffer
 * count is declared rather than writing past the entries it was given. */
#define SDSGE_ARENA_MAX_BUFFERS 24

/* float/int arena offset descriptor: the lane boundary closing each buffer.
 * Entry i is one past the end of buffer i, so buffer i spans
 * `[i ? entry[i - 1] : 0, entry[i])` and the last entry is the lane total. The
 * opening boundary is the arena itself and is not stored, because a slot whose
 * value is always zero is one every walk has to step over to reach a buffer. A
 * buffer a configuration leaves out repeats the entry before it, so walking
 * produces nothing for it rather than the walk having to ask whether it is
 * there. A total is a read, never a sum.
 *
 * A layout writes the entries in order, each one the entry before it plus the
 * width of the buffer it closes. An entry means the same thing the moment it is
 * written, so the descriptor is never half-built in some other unit. A layout
 * writes every entry its lane counts, including the ones it has nothing to put
 * in; none of them carry a default it could lean on instead.
 *
 * The entries live inline, which makes this a value the way `arena_size` is: a
 * copy is a snapshot rather than an alias, a return outlives the frame that
 * built it, and `const` reaches the entries instead of stopping at a pointer.
 * The buffer counts ride along, so a walk bounds itself on the descriptor it
 * was handed rather than on a constant each call site names a second time.
 */
typedef struct {
  i64 foffset[SDSGE_ARENA_MAX_BUFFERS]; // (n_float_buffers, ): buffer end
  i64 ioffset[SDSGE_ARENA_MAX_BUFFERS]; // (n_int_buffers, ): buffer end
  i64 n_float_buffers;
  i64 n_int_buffers;
} arena_offset;

/* Offsets for lanes of the given buffer counts, entries left unwritten. Zeroing
 * them would be a pass over slots a layout is about to overwrite anyway, and
 * the slots past a lane's count are never read. */
static inline arena_offset make_offset(const i64 n_float_buffers,
                                       const i64 n_int_buffers) {
  arena_offset off;
  off.n_float_buffers = n_float_buffers;
  off.n_int_buffers = n_int_buffers;
  return off;
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
