#include "philox.h"

/* Direct includes (not transitive via philox.h): `bitgen_t` and the fill
 * functions are declared here, `memcpy` there. Native-include hygiene wants
 * the header that declares each used symbol included at its use site. Resolved
 * at link time against `npyrandom.lib`. */
#include <string.h>

#include "numpy/random/distributions.h"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

/* Random123 Philox4x64 constants: two multipliers and the two Weyl increments
 * that bump the key between rounds. */
#define PHILOX_M0 0xD2E7470EE14C6C93ULL
#define PHILOX_M1 0xCA5A826395121157ULL
#define PHILOX_W0 0x9E3779B97F4A7C15ULL
#define PHILOX_W1 0xBB67AE8584CAA73BULL

/* Full 64x64 -> 128 multiply. clang and GCC have the native 128-bit type, MSVC
 * has the intrinsic, and the fallback splits into 32-bit halves so an unknown
 * compiler in the wheel matrix still builds. */
static inline u64 sdsge_mulhilo64(u64 a, u64 b, u64 *hi) {
#if defined(__SIZEOF_INT128__)
  __uint128_t product = (__uint128_t)a * (__uint128_t)b;
  *hi = (u64)(product >> 64);
  return (u64)product;
#elif defined(_MSC_VER)
  return _umul128(a, b, hi);
#else
  u64 a_lo = a & 0xFFFFFFFFULL;
  u64 a_hi = a >> 32;
  u64 b_lo = b & 0xFFFFFFFFULL;
  u64 b_hi = b >> 32;

  u64 lo_lo = a_lo * b_lo;
  u64 hi_lo = a_hi * b_lo;
  u64 lo_hi = a_lo * b_hi;
  u64 hi_hi = a_hi * b_hi;

  u64 carry = ((lo_lo >> 32) + (hi_lo & 0xFFFFFFFFULL) + (lo_hi & 0xFFFFFFFFULL)) >> 32;
  *hi = hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + carry;
  return a * b;
#endif
}

/* One Philox4x64 round: two multiplies, then a fixed permutation of the four
 * counter words mixing in the key. */
static inline void sdsge_philox_round(u64 ctr[4], const u64 key[2]) {
  u64 hi0;
  u64 hi1;
  u64 lo0 = sdsge_mulhilo64(PHILOX_M0, ctr[0], &hi0);
  u64 lo1 = sdsge_mulhilo64(PHILOX_M1, ctr[2], &hi1);

  u64 out0 = hi1 ^ ctr[1] ^ key[0];
  u64 out1 = lo1;
  u64 out2 = hi0 ^ ctr[3] ^ key[1];
  u64 out3 = lo0;

  ctr[0] = out0;
  ctr[1] = out1;
  ctr[2] = out2;
  ctr[3] = out3;
}

/* The bijection itself: 10 rounds over a private copy, so the caller's counter
 * stays put and only the block index advances between blocks. */
static void sdsge_philox_block(const u64 ctr_in[4], const u64 key_in[2],
                               u64 out[4]) {
  u64 ctr[4];
  u64 key[2];
  int round;

  memcpy(ctr, ctr_in, sizeof(ctr));
  memcpy(key, key_in, sizeof(key));

  for (round = 0; round < SDSGE_PHILOX_ROUNDS; round++) {
    if (round > 0) {
      key[0] += PHILOX_W0;
      key[1] += PHILOX_W1;
    }
    sdsge_philox_round(ctr, key);
  }

  memcpy(out, ctr, sizeof(ctr));
}

void sdsge_philox_seed(sdsge_philox_state *st, u64 key0, u64 key1, u64 stream0,
                       u64 stream1) {
  st->key[0] = key0;
  st->key[1] = key1;
  st->ctr[0] = 0;
  st->ctr[1] = 0;
  st->ctr[2] = stream0;
  st->ctr[3] = stream1;
  st->buffer[0] = 0;
  st->buffer[1] = 0;
  st->buffer[2] = 0;
  st->buffer[3] = 0;
  st->buffer_pos = 4;
}

u64 sdsge_philox_next_u64(sdsge_philox_state *st) {
  if (st->buffer_pos >= 4) {
    /* Advance the block counter over its low 128 bits, leaving the stream
     * words untouched. That is 2^128 blocks per stream before wraparound. */
    st->ctr[0] += 1;
    if (st->ctr[0] == 0) {
      st->ctr[1] += 1;
    }
    sdsge_philox_block(st->ctr, st->key, st->buffer);
    st->buffer_pos = 0;
  }
  return st->buffer[st->buffer_pos++];
}

/* bitgen_t vtable over the state above. numpy's transforms reach the engine
 * only through these, so binding them is all it takes to reuse npyrandom. */

static u64 sdsge_philox_bg_next_u64(void *st) {
  return sdsge_philox_next_u64((sdsge_philox_state *)st);
}

static u32 sdsge_philox_bg_next_u32(void *st) {
  return (u32)(sdsge_philox_next_u64((sdsge_philox_state *)st) >> 32);
}

static f64 sdsge_philox_bg_next_double(void *st) {
  /* The standard 53-bit construction, identical to what numpy's own bit
   * generators use, so the uniform transform sees the distribution it
   * expects. */
  return (f64)(sdsge_philox_next_u64((sdsge_philox_state *)st) >> 11) *
         (1.0 / 9007199254740992.0);
}

static u64 sdsge_philox_bg_next_raw(void *st) {
  return sdsge_philox_next_u64((sdsge_philox_state *)st);
}

/* The `bitgen_t` lives on the caller's stack for the duration of one fill; it
 * is a pure view over `st`, holding no state of its own. */
static void sdsge_philox_bind(bitgen_t *bg, sdsge_philox_state *st) {
  bg->state = (void *)st;
  bg->next_uint64 = sdsge_philox_bg_next_u64;
  bg->next_uint32 = sdsge_philox_bg_next_u32;
  bg->next_double = sdsge_philox_bg_next_double;
  bg->next_raw = sdsge_philox_bg_next_raw;
}

void sdsge_philox_standard_normal_fill(sdsge_philox_state *st, i64 n,
                                       f64 *SDSGE_RESTRICT out) {
  bitgen_t bg;
  if (n <= 0) {
    return;
  }
  sdsge_philox_bind(&bg, st);
  random_standard_normal_fill(&bg, (npy_intp)n, out);
}

void sdsge_philox_standard_uniform_fill(sdsge_philox_state *st, i64 n,
                                        f64 *SDSGE_RESTRICT out) {
  bitgen_t bg;
  if (n <= 0) {
    return;
  }
  sdsge_philox_bind(&bg, st);
  random_standard_uniform_fill(&bg, (npy_intp)n, out);
}
