import numpy as np
from numpy import float64
from libc.stdint cimport int64_t

cdef extern from "transforms.h":
    # LOG TRANSFORM
    void sdsge_log_fwd(const double *x, double *y) nogil
    void sdsge_log_fwd_arr(const double *x, double *y, int64_t n) nogil

    void sdsge_log_inv(const double *y, double *x) nogil
    void sdsge_log_inv_arr(const double *y, double *x, int64_t n) nogil

    void sdsge_log_grad_fwd(const double *x, double *y) nogil
    void sdsge_log_grad_fwd_arr(const double *x, double *y, int64_t n) nogil

    void sdsge_log_grad_inv(const double *y, double *x) nogil
    void sdsge_log_grad_inv_arr(const double *y, double *x, int64_t n) nogil

    void sdsge_log_ldet_abs_jac_fwd(const double *x, double *y) nogil
    void sdsge_log_ldet_abs_jac_fwd_arr(const double *x, double *y, int64_t n) nogil

    void sdsge_log_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_log_ldet_abs_jac_inv_arr(const double *y, double *x, int64_t n) nogil

    void sdsge_log_grad_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_log_grad_ldet_abs_jac_inv_arr(const double *y,
                                             double *x, int64_t n) nogil

    # LOGIT TRANSFORM
    void sdsge_logit_fwd(const double *x, double *y) nogil
    void sdsge_logit_fwd_arr(const double *x,
                             double *y, int64_t n) nogil

    void sdsge_logit_inv(const double *y, double *x) nogil
    void sdsge_logit_inv_arr(const double *y,
                             double *x, int64_t n) nogil

    void sdsge_logit_grad_fwd(const double *x, double *y) nogil
    void sdsge_logit_grad_fwd_arr(const double *x,
                                  double *y, int64_t n) nogil

    void sdsge_logit_grad_inv(const double *y, double *x) nogil
    void sdsge_logit_grad_inv_arr(const double *y,
                                  double *x, int64_t n) nogil

    void sdsge_logit_ldet_abs_jac_fwd(const double *x, double *y) nogil
    void sdsge_logit_ldet_abs_jac_fwd_arr(const double *x,
                                          double *y, int64_t n) nogil

    void sdsge_logit_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_logit_ldet_abs_jac_inv_arr(const double *y,
                                          double *x, int64_t n) nogil

    void sdsge_logit_grad_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_logit_grad_ldet_abs_jac_inv_arr(const double *y,
                                               double *x, int64_t n) nogil

# LOG TRANSFORM


def log_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_log_fwd(&xx, &yy)
        return float64(yy)

    cdef int64_t n
    cdef double[::1] xv, yv
    arr = np.ascontiguousarray(x, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    xv = arr.reshape(-1)
    yv = out.reshape(-1)
    with nogil:
        sdsge_log_fwd_arr(&xv[0], &yv[0], n)
    return out


def log_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_log_inv(&yy, &xx)
        return float64(xx)

    cdef int64_t n
    cdef double[::1] yv, xv
    arr = np.ascontiguousarray(y, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    yv = arr.reshape(-1)
    xv = out.reshape(-1)
    with nogil:
        sdsge_log_inv_arr(&yv[0], &xv[0], n)
    return out


def log_grad_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_log_grad_fwd(&xx, &yy)
        return float64(yy)

    cdef int64_t n
    cdef double[::1] xv, yv
    arr = np.ascontiguousarray(x, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    xv = arr.reshape(-1)
    yv = out.reshape(-1)
    with nogil:
        sdsge_log_grad_fwd_arr(&xv[0], &yv[0], n)
    return out


def log_grad_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_log_grad_inv(&yy, &xx)
        return float64(xx)

    cdef int64_t n
    cdef double[::1] yv, xv
    arr = np.ascontiguousarray(y, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    yv = arr.reshape(-1)
    xv = out.reshape(-1)
    with nogil:
        sdsge_log_grad_inv_arr(&yv[0], &xv[0], n)
    return out


def log_ldet_abs_jac_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_log_ldet_abs_jac_fwd(&xx, &yy)
        return float64(yy)

    cdef int64_t n
    cdef double[::1] xv, yv
    arr = np.ascontiguousarray(x, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    xv = arr.reshape(-1)
    yv = out.reshape(-1)
    with nogil:
        sdsge_log_ldet_abs_jac_fwd_arr(&xv[0], &yv[0], n)
    return out


def log_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_log_ldet_abs_jac_inv(&yy, &xx)
        return float64(xx)

    cdef int64_t n
    cdef double[::1] yv, xv
    arr = np.ascontiguousarray(y, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    yv = arr.reshape(-1)
    xv = out.reshape(-1)
    with nogil:
        sdsge_log_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


def log_grad_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_log_grad_ldet_abs_jac_inv(&yy, &xx)
        return float64(xx)

    cdef int64_t n
    cdef double[::1] yv, xv
    arr = np.ascontiguousarray(y, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    yv = arr.reshape(-1)
    xv = out.reshape(-1)
    with nogil:
        sdsge_log_grad_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


# LOGIT TRANSFORM

def logit_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_logit_fwd(&xx, &yy)
        return float64(yy)

    cdef int64_t n
    cdef double[::1] xv, yv
    arr = np.ascontiguousarray(x, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    xv = arr.reshape(-1)
    yv = out.reshape(-1)
    with nogil:
        sdsge_logit_fwd_arr(&xv[0], &yv[0], n)
    return out


def logit_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_logit_inv(&yy, &xx)
        return float64(xx)

    cdef int64_t n
    cdef double[::1] yv, xv
    arr = np.ascontiguousarray(y, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    yv = arr.reshape(-1)
    xv = out.reshape(-1)
    with nogil:
        sdsge_logit_inv_arr(&yv[0], &xv[0], n)
    return out


def logit_grad_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_logit_grad_fwd(&xx, &yy)
        return float64(yy)

    cdef int64_t n
    cdef double[::1] xv, yv
    arr = np.ascontiguousarray(x, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    xv = arr.reshape(-1)
    yv = out.reshape(-1)
    with nogil:
        sdsge_logit_grad_fwd_arr(&xv[0], &yv[0], n)
    return out


def logit_grad_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_logit_grad_inv(&yy, &xx)
        return float64(xx)

    cdef int64_t n
    cdef double[::1] yv, xv
    arr = np.ascontiguousarray(y, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    yv = arr.reshape(-1)
    xv = out.reshape(-1)
    with nogil:
        sdsge_logit_grad_inv_arr(&yv[0], &xv[0], n)
    return out


def logit_ldet_abs_jac_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_logit_ldet_abs_jac_fwd(&xx, &yy)
        return float64(yy)

    cdef int64_t n
    cdef double[::1] xv, yv
    arr = np.ascontiguousarray(x, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    xv = arr.reshape(-1)
    yv = out.reshape(-1)
    with nogil:
        sdsge_logit_ldet_abs_jac_fwd_arr(&xv[0], &yv[0], n)
    return out


def logit_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_logit_ldet_abs_jac_inv(&yy, &xx)
        return float64(xx)

    cdef int64_t n
    cdef double[::1] yv, xv
    arr = np.ascontiguousarray(y, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    yv = arr.reshape(-1)
    xv = out.reshape(-1)
    with nogil:
        sdsge_logit_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


def logit_grad_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_logit_grad_ldet_abs_jac_inv(&yy, &xx)
        return float64(xx)

    cdef int64_t n
    cdef double[::1] yv, xv
    arr = np.ascontiguousarray(y, dtype=float64)
    out = np.empty_like(arr)
    n = arr.size
    if n == 0:
        return out
    yv = arr.reshape(-1)
    xv = out.reshape(-1)
    with nogil:
        sdsge_logit_grad_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out
