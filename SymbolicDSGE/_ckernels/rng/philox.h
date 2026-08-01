#ifndef SDSGE_PHILOX_H
#define SDSGE_PHILOX_H

#include "../_common/sdsge_common.h"

/* Counter-based engine for the Monte Carlo shock draw (issue #374). Seeking to
 * a replication is an assignment, so each worker holds its own state on the
 * stack and nothing here allocates or coordinates. The fills wrap this state in
 * a `bitgen_t`, keeping numpy's compiled transforms.
 *
 * A SECOND engine, separate from the borrowed PCG64 bridge in rng.h, which is
 * bit-pinned by tests/ckernels/test_rng_parity.py. Nothing here touches it. */

/* Philox4x64-10, the standard round count. */
#define SDSGE_PHILOX_ROUNDS 10

typedef struct {
  u64 key[2];
  /* ctr[0..1] is the block counter advanced per 4 draws; ctr[2..3] identifies
   * the stream and is never advanced. */
  u64 ctr[4];
  u64 buffer[4];
  i64 buffer_pos; /* 4 means exhausted; the next draw refills. */
} sdsge_philox_state;

/* Seed a stream. Any distinct combination of the four words yields an
 * independent sequence (zero is fine) and always replays the same draws. */
void sdsge_philox_seed(sdsge_philox_state *st, u64 key0, u64 key1, u64 stream0,
                       u64 stream1);

/* One raw 64-bit draw advancing `st`. */
u64 sdsge_philox_next_u64(sdsge_philox_state *st);

/* Fill `out[0..n)` with standard normal draws (mean 0, var 1) advancing `st`.
 * The transform is numpy's ziggurat, linked from npyrandom. */
void sdsge_philox_standard_normal_fill(sdsge_philox_state *st, i64 n,
                                       f64 *SDSGE_RESTRICT out);

/* Fill `out[0..n)` with standard uniform draws in [0, 1) advancing `st`. The
 * transform is numpy's, linked from npyrandom. */
void sdsge_philox_standard_uniform_fill(sdsge_philox_state *st, i64 n,
                                        f64 *SDSGE_RESTRICT out);

#endif /* SDSGE_PHILOX_H */
