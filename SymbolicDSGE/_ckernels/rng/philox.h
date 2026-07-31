#ifndef SDSGE_PHILOX_H
#define SDSGE_PHILOX_H

#include "../_common/sdsge_common.h"

/* Counter-based engine for the Monte Carlo shock draw (issue #374).
 *
 * The MC runner parallelizes over replications and hands each step only its
 * `rep_idx`, so a shock stream has to be derivable from that index alone:
 * seekable, allocation-free, and identical regardless of which worker runs the
 * replication or how many workers exist. A borrowed numpy PCG64 cannot do
 * this. The `bitgen_t` ABI exposes only the `next_*` function pointers, with
 * no reseed and no jump, so C could only ever share one stream across workers,
 * which is both a data race and order-dependent.
 *
 * Philox is counter-based: the state IS a key plus a counter, so seeking to
 * replication `i` is an assignment rather than a seeding pass. Reseeding costs
 * nothing, distinct `(key, stream)` pairs give independent streams by
 * construction, and no coordination between workers is needed.
 *
 * This preserves the engine/transform split rng.h describes, with the engine
 * swapped: the fills below wrap this state in a `bitgen_t` and hand it to
 * npyrandom, so the uniform and ziggurat-normal transforms are still numpy's
 * compiled implementations. Only the bit source changes.
 *
 * Note that this is a SECOND engine, deliberately separate from the borrowed
 * PCG64 bridge in rng.h. That bridge exists to be bit-identical to numpy and
 * is pinned by tests/ckernels/test_rng_parity.py; nothing here touches it.
 *
 * The state is small and fully public so callers can hold it on the stack
 * inside an OpenMP loop body. Nothing here allocates, and numpy's headers stay
 * out of this header (philox.c alone includes them), matching rng.h. */

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

/* Seed a stream. `key0`/`key1` and `stream0`/`stream1` jointly select it; any
 * distinct combination yields an independent sequence, with no restriction on
 * the values (zero is fine). The block counter starts fresh, so seeding the
 * same four words always replays the same draws. */
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
