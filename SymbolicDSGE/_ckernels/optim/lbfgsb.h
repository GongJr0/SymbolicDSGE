#ifndef SDSGE_OPTIM_LBFGSB_H
#define SDSGE_OPTIM_LBFGSB_H

#include <math.h>

#include "shim.h"

/* Compat header for the vendored L-BFGS-B kernel (optim/lbfgsb.c, from scipy's
 * BSD-3 netlib v3.0 translation). The kernel was retyped to our i64/f64 and its
 * linear-algebra calls point straight at the self-contained primitives in
 * shim.h (sdsge_shim_*); no external library, no indirection layer. */

void setulb(i64 n, i64 m, f64 *x, f64 *l, f64 *u, i64 *nbd, f64 *f, f64 *g,
            f64 factr, f64 pgtol, f64 *wa, i64 *iwa, i64 *task, i64 *lsave,
            i64 *isave, f64 *dsave, i64 maxls, i64 *ln_task);

#endif /* SDSGE_OPTIM_LBFGSB_H */
