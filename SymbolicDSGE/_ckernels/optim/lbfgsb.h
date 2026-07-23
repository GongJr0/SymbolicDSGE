#ifndef SDSGE_OPTIM_LBFGSB_H
#define SDSGE_OPTIM_LBFGSB_H

#include <math.h>

#include "blas_backend.h"

/* scipy's optimize/src/lbfgsb.c is vendored byte-for-byte (BSD-3, netlib v3.0
 * translation). This header stands in for its original lbfgsb.h +
 * blaslapack_declarations.h: it supplies CBLAS_INT and the BLAS_FUNC macro so the
 * source stays unmodified, but BLAS_FUNC now dispatches through a runtime backend
 * vtable (capsule pointers or self-contained shims) instead of a linked BLAS
 * symbol. That single macro redirection is the whole transpile. */

typedef blas_int CBLAS_INT;

/* Backend for the current optimize, set once by the driver before the setulb
 * reverse-communication loop. One active optimize per thread (the vendored
 * kernel keeps no other global state). */
extern const sdsge_blas_ops *sdsge_optim_blas;
void sdsge_optim_set_blas(const sdsge_blas_ops *ops);

#define BLAS_FUNC(name) (sdsge_optim_blas->name)

void setulb(CBLAS_INT n, CBLAS_INT m, double *x, double *l, double *u,
            CBLAS_INT *nbd, double *f, double *g, double factr, double pgtol,
            double *wa, CBLAS_INT *iwa, CBLAS_INT *task, CBLAS_INT *lsave,
            CBLAS_INT *isave, double *dsave, CBLAS_INT maxls, CBLAS_INT *ln_task);

#endif /* SDSGE_OPTIM_LBFGSB_H */
