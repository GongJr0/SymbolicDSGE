#ifndef SDSGE_KLEIN_POSTPROC_H
#define SDSGE_KLEIN_POSTPROC_H
#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"

i64 klein_postproc(const c128 *SDSGE_RESTRICT s, const c128 *SDSGE_RESTRICT t,
                   const c128 *SDSGE_RESTRICT z, const i64 n_s, const i64 n_cs,
                   c128 *SDSGE_RESTRICT f, c128 *SDSGE_RESTRICT p,
                   i64 *SDSGE_RESTRICT stab, c128 *SDSGE_RESTRICT eig);

#define SDSGE_KLEIN_POSTPROC_SUCCESS 0
#define SDSGE_KLEIN_POSTPROC_ALLOC_FAIL -1
#define SDSGE_KLEIN_POSTPROC_SINGULAR -2
#define SDSGE_KLEIN_POSTPROC_INVALID -3
#endif /* SDSGE_KLEIN_POSTPROC_H */
