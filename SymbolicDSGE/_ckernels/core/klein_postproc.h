#ifndef SDSGE_KLEIN_POSTPROC_H
#define SDSGE_KLEIN_POSTPROC_H
#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

arena_size klein_postproc_arena_size(i64 n_s, i64 n_cs);

i64 klein_postproc(const c128 *SDSGE_RESTRICT s, const c128 *SDSGE_RESTRICT t,
                   const c128 *SDSGE_RESTRICT z, const i64 n_s, const i64 n_cs,
                   c128 *SDSGE_RESTRICT f, c128 *SDSGE_RESTRICT p,
                   i64 *SDSGE_RESTRICT stab, c128 *SDSGE_RESTRICT eig,
                   f64 *SDSGE_RESTRICT arena, i64 *SDSGE_RESTRICT iarena);

#define SDSGE_KLEIN_POSTPROC_SUCCESS 0
#define SDSGE_KLEIN_POSTPROC_SINGULAR -2
#define SDSGE_KLEIN_POSTPROC_INVALID -3

/* Stamped into *stab up front; 2 is not a stability status the solve returns. */
#define SDSGE_KLEIN_STAB_UNSET 2
#endif /* SDSGE_KLEIN_POSTPROC_H */
