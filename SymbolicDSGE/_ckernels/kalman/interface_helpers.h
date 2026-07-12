#ifndef SDSGE_INTERFACE_HELPERS_H
#define SDSGE_INTERFACE_HELPERS_H

#include "../_common/sdsge_common.h"
#include "../_common/sdsge_linalg.h"

void sdsge_cov_from_unconstrained(const f64 *SDSGE_RESTRICT z,
                                  const f64 *SDSGE_RESTRICT std, const i64 K,
                                  f64 *SDSGE_RESTRICT scratch_M,
                                  f64 *SDSGE_RESTRICT out);

#endif /* SDSGE_INTERFACE_HELPERS_H */
