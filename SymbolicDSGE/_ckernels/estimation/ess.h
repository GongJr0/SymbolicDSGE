#ifndef SDSGE_ESS_H
#define SDSGE_ESS_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

#define SDSGE_ESS_OK 0
#define SDSGE_ESS_EBADSHAPE -701
#define SDSGE_ESS_EUDEFVARIANCE -702
#define SDSGE_ESS_ENODECAY -703
#define SDSGE_ESS_ENONPOSTAU -704
/* FFT-based autocorrelation. `arena` is 2 * next_pow2(2*n-1) floats
 * and out is n floats. Returns all acorrs as a length-n vector. */
arena_size sdsge_acorr_fft_arena_size(const i64 n);
int sdsge_acorr_fft(const f64 *SDSGE_RESTRICT x, const i64 n,
                    f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT out);

arena_size sdsge_ess_1d_arena_size(const i64 n);
int sdsge_ess_1d(const f64 *SDSGE_RESTRICT x, const i64 n,
                 f64 *SDSGE_RESTRICT arena, f64 *SDSGE_RESTRICT out);

#endif // !SDSGE_ESS_H
