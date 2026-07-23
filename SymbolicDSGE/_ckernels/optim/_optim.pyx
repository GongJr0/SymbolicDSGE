# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython surface for the native optimizer module.

Exposes the L-BFGS-B driver over synthetic benchmark objectives (Rosenbrock,
ill-conditioned quadratic, separable double well) with a selectable BLAS backend
(``"capsule"`` = scipy cython_blas/lapack pointers, ``"shim"`` = self-contained).
This is the entry point for the scipy parity tests and the capsule-vs-shim bench;
the estimation objective wires in separately (issue #330).
"""

from libc.stdint cimport int64_t

from cpython.pycapsule cimport PyCapsule_GetName, PyCapsule_GetPointer

import numpy as np
import scipy.linalg.cython_blas as _cblas
import scipy.linalg.cython_lapack as _clapack


cdef extern from "blas_backend.h":
    ctypedef struct sdsge_blas_ops:
        void *dcopy
        void *daxpy
        void *dscal
        void *ddot
        void *dnrm2
        void *dpotrf
        void *dtrtrs
    void sdsge_blas_ops_shim(sdsge_blas_ops *ops)
    void sdsge_blas_ops_from_ptrs(sdsge_blas_ops *ops, void *dcopy, void *daxpy,
                                  void *dscal, void *ddot, void *dnrm2,
                                  void *dpotrf, void *dtrtrs)


cdef extern from "optim.h":
    ctypedef double (*sdsge_objective_fn)(const double *x, void *ctx) noexcept nogil

    ctypedef struct sdsge_lbfgsb_options:
        int64_t m
        int64_t maxiter
        int64_t maxfun
        int64_t maxls
        double factr
        double pgtol
        double fd_step

    ctypedef struct sdsge_lbfgsb_result:
        int64_t status
        int64_t nfev
        int64_t nit
        double fun
        int success
        const char *message

    int64_t sdsge_lbfgsb(sdsge_objective_fn obj, void *obj_ctx,
                         const sdsge_blas_ops *blas, int64_t n, double *x,
                         const double *lo, const double *hi, const int64_t *nbd,
                         const sdsge_lbfgsb_options *opt,
                         sdsge_lbfgsb_result *out) nogil


# --- Benchmark objectives (match sdsge_objective_fn) --------------------------
cdef struct ObjCtx:
    int64_t n
    const double *p   # problem params (may be NULL)


cdef double _obj_rosenbrock(const double *x, void *ctx) noexcept nogil:
    cdef ObjCtx *c = <ObjCtx *>ctx
    cdef int64_t n = c.n, i
    cdef double s = 0.0, t1, t2
    for i in range(n - 1):
        t1 = x[i + 1] - x[i] * x[i]
        t2 = 1.0 - x[i]
        s += 100.0 * t1 * t1 + t2 * t2
    return s


cdef double _obj_quad(const double *x, void *ctx) noexcept nogil:
    # p = [d_0..d_{n-1}, xstar_0..xstar_{n-1}]; f = 0.5 * sum d_i (x_i - xstar_i)^2
    cdef ObjCtx *c = <ObjCtx *>ctx
    cdef int64_t n = c.n, i
    cdef const double *d = c.p
    cdef const double *xs = c.p + n
    cdef double s = 0.0, e
    for i in range(n):
        e = x[i] - xs[i]
        s += 0.5 * d[i] * e * e
    return s


cdef double _obj_double_well(const double *x, void *ctx) noexcept nogil:
    # separable double well; minima at x_i = +/-1 in each coordinate
    cdef ObjCtx *c = <ObjCtx *>ctx
    cdef int64_t n = c.n, i
    cdef double s = 0.0, t
    for i in range(n):
        t = x[i] * x[i] - 1.0
        s += t * t
    return s


cdef sdsge_objective_fn _objective(str name) except NULL:
    if name == "rosenbrock":
        return _obj_rosenbrock
    if name == "quad":
        return _obj_quad
    if name == "double_well":
        return _obj_double_well
    raise ValueError(f"unknown objective {name!r}")


cdef void* _cap(mod, str name):
    cap = mod.__pyx_capi__[name]
    return PyCapsule_GetPointer(cap, PyCapsule_GetName(cap))


cdef void _fill_backend(sdsge_blas_ops *ops, str backend):
    if backend == "shim":
        sdsge_blas_ops_shim(ops)
    elif backend == "capsule":
        sdsge_blas_ops_from_ptrs(
            ops,
            _cap(_cblas, "dcopy"), _cap(_cblas, "daxpy"), _cap(_cblas, "dscal"),
            _cap(_cblas, "ddot"), _cap(_cblas, "dnrm2"),
            _cap(_clapack, "dpotrf"), _cap(_clapack, "dtrtrs"),
        )
    else:
        raise ValueError(f"backend must be 'capsule' or 'shim', got {backend!r}")


def run_lbfgsb(
    str objective,
    double[::1] x0,
    str backend="shim",
    bounds=None,
    object params=None,
    int m=10,
    int maxiter=15000,
    int maxfun=15000,
    int maxls=20,
    double factr=1e7,
    double pgtol=1e-5,
    double fd_step=0.0,
):
    """Minimize a benchmark objective with native L-BFGS-B. ``bounds`` is None
    (unbounded) or a length-n list of (lo, hi) with None for a missing side.
    ``params`` feeds the parametrized objectives (``quad``). Returns a result
    dict mirroring the lean native struct."""
    cdef int64_t n = x0.shape[0]
    x = np.array(x0, dtype=np.float64, copy=True)
    cdef double[::1] xv = x

    cdef double[::1] pv
    cdef ObjCtx ctx
    ctx.n = n
    ctx.p = NULL
    if params is not None:
        pv = np.ascontiguousarray(params, dtype=np.float64)
        ctx.p = &pv[0]

    # Bounds -> lo/hi/nbd (scipy map: none=0, lower=1, both=2, upper=3).
    cdef double[::1] lo = np.zeros(n, dtype=np.float64)
    cdef double[::1] hi = np.zeros(n, dtype=np.float64)
    cdef int64_t[::1] nbd = np.zeros(n, dtype=np.int64)
    cdef int has_bounds = bounds is not None
    cdef int i
    if has_bounds:
        for i in range(n):
            lb, ub = bounds[i]
            has_lo = lb is not None
            has_hi = ub is not None
            if has_lo:
                lo[i] = lb
            if has_hi:
                hi[i] = ub
            nbd[i] = (2 if has_hi else 1) if has_lo else (3 if has_hi else 0)

    cdef sdsge_blas_ops ops
    _fill_backend(&ops, backend)

    cdef sdsge_objective_fn obj = _objective(objective)

    cdef sdsge_lbfgsb_options opt
    opt.m = m
    opt.maxiter = maxiter
    opt.maxfun = maxfun
    opt.maxls = maxls
    opt.factr = factr
    opt.pgtol = pgtol
    opt.fd_step = fd_step

    cdef sdsge_lbfgsb_result res
    cdef const int64_t *nbd_ptr = &nbd[0] if has_bounds else NULL

    with nogil:
        sdsge_lbfgsb(obj, <void *>&ctx, &ops, n, &xv[0], &lo[0], &hi[0], nbd_ptr,
                     &opt, &res)

    return {
        "x": x,
        "fun": res.fun,
        "nfev": int(res.nfev),
        "nit": int(res.nit),
        "success": bool(res.success),
        "status": int(res.status),
        "message": (<bytes>res.message).decode() if res.message != NULL else "",
    }
