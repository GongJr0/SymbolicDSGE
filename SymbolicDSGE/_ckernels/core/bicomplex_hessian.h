#ifndef SDSGE_BICOMPLEX_HESSIAN_H
#define SDSGE_BICOMPLEX_HESSIAN_H
#include "../_common/sdsge_common.h"
#include "../_common/sdsge_complex.h"
#include "../_common/sdsge_bicomplex.h"

/* Second-order preproc: the residual Hessian F_xx via the bicomplex step.
 * `residual` is the numba @cfunc built with BicomplexOps, invoked by pointer
 * over bc256 buffers (each bc256 == complex128[2], the cfunc's per-element view).
 *
 * z = (fwd, cur) stacked, 2*n_var wide. Perturbing z_i on the i-unit (a.im) and
 * z_j on the j-unit (b.re) -- both units on the same variable when i == j -- the
 * residual's ij component (b.im) / h^2 is d^2 F / dz_i dz_j. The driver only
 * constructs perturbations and extracts the ij component; the bicomplex
 * arithmetic lives in the numba cfunc.
 *
 * `hessian` is (n_eq, 2*n_var, 2*n_var) row-major f64, symmetric in the last
 * two. Levels only (no log-linear wrapping at order > 1). */

typedef void (*bc_residual_fn)(const bc256 *fwd, const bc256 *cur,
                               const bc256 *par, bc256 *out);

i64 sdsge_bicomplex_hessian(bc_residual_fn residual, const f64 *SDSGE_RESTRICT ss,
                            const f64 *SDSGE_RESTRICT par, i64 n_var, i64 n_par,
                            i64 n_eq, f64 step, f64 *SDSGE_RESTRICT hessian);

#define SDSGE_HESSIAN_OK 0
#define SDSGE_HESSIAN_ALLOC_FAIL -1

#endif /* SDSGE_BICOMPLEX_HESSIAN_H */
