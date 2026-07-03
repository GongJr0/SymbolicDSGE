#ifndef SDSGE_SPIKE_H
#define SDSGE_SPIKE_H
#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

/* Portability guard for the cfunc-leaf architecture (issue #248). A hand-written
 * C translation unit that invokes a numba @cfunc through its raw address,
 * mirroring how the perturbation preproc driver calls the model-residual leaf.
 * It asserts, on every platform we build wheels for, that:
 *   - a numba @cfunc is callable from native code by pointer,
 *   - with the GIL released,
 *   - without numba's runtime (NRT) being reachable inside the call.
 * The whole preproc design depends on this holding, so tests/ckernels/test_spike.py
 * exercises it across the wheel platform matrix -- on demand via the "cfunc ABI"
 * workflow (.github/workflows/cfunc-abi.yml) and again on each wheel build via
 * cibuildwheel's per-platform test-command. Keep it. */

typedef void (*spike_residual_fn)(const c128 *a, const c128 *b, c128 *out, i64 n);

void spike_call(spike_residual_fn fn, const c128 *a, const c128 *b, c128 *out,
                i64 n);

#endif /* SDSGE_SPIKE_H */
