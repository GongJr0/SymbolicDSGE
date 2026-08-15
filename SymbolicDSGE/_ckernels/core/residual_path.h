#ifndef SDSGE_RESIDUAL_PATH_H
#define SDSGE_RESIDUAL_PATH_H
#include "klein_preproc.h" /* sdsge_residual_fn, c128, i64, f64 */

/* Evaluate the model residual @cfunc over a simulated path. For each step t,
 * calls resid(&fwd[t], &cur[t], &prev[t], &eps[t], par, out) and stores
 * Re(out[k]) into residuals[t*n_eq + k]. States/params are complex128 (real,
 * zero imaginary). `prev` is n_steps*n_var and `eps` is n_steps*n_exog, both
 * dated like the rest of the path. */
i64 sdsge_residual_path(sdsge_residual_fn resid, const c128 *SDSGE_RESTRICT cur,
                        const c128 *SDSGE_RESTRICT fwd,
                        const c128 *SDSGE_RESTRICT prev,
                        const c128 *SDSGE_RESTRICT eps,
                        const c128 *SDSGE_RESTRICT par, i64 n_steps, i64 n_var,
                        i64 n_exog, i64 n_eq, f64 *SDSGE_RESTRICT residuals);

#define SDSGE_RESIDUAL_PATH_OK 0
#define SDSGE_RESIDUAL_PATH_ALLOC_FAIL -701

#endif /* SDSGE_RESIDUAL_PATH_H */
