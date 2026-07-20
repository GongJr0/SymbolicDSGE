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

    # PROBIT TRANSFORM
    void sdsge_probit_fwd(const double *x, double *y) nogil
    void sdsge_probit_fwd_arr(const double *x,
                              double *y, int64_t n) nogil

    void sdsge_probit_inv(const double *y, double *x) nogil
    void sdsge_probit_inv_arr(const double *y,
                              double *x, int64_t n) nogil

    void sdsge_probit_grad_fwd(const double *x, double *y) nogil
    void sdsge_probit_grad_fwd_arr(const double *x,
                                   double *y, int64_t n) nogil

    void sdsge_probit_grad_inv(const double *y, double *x) nogil
    void sdsge_probit_grad_inv_arr(const double *y,
                                   double *x, int64_t n) nogil

    void sdsge_probit_ldet_abs_jac_fwd(const double *x, double *y) nogil
    void sdsge_probit_ldet_abs_jac_fwd_arr(const double *x,
                                           double *y, int64_t n) nogil

    void sdsge_probit_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_probit_ldet_abs_jac_inv_arr(const double *y,
                                           double *x, int64_t n) nogil

    void sdsge_probit_grad_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_probit_grad_ldet_abs_jac_inv_arr(const double *y,
                                                double *x, int64_t n) nogil

    # AFFINE LOGIT TRANSFORM
    void sdsge_aff_logit_fwd(const double *x, double *y,
                             const double *a, const double *b) nogil
    void sdsge_aff_logit_fwd_arr(const double *x, double *y, int64_t n,
                                 const double *a, const double *b) nogil

    void sdsge_aff_logit_inv(const double *y, double *x,
                             const double *a, const double *b) nogil
    void sdsge_aff_logit_inv_arr(const double *y, double *x, int64_t n,
                                 const double *a, const double *b) nogil

    void sdsge_aff_logit_grad_fwd(const double *x, double *y,
                                  const double *a, const double *b) nogil
    void sdsge_aff_logit_grad_fwd_arr(const double *x, double *y, int64_t n,
                                      const double *a, const double *b) nogil

    void sdsge_aff_logit_grad_inv(const double *y, double *x,
                                  const double *a, const double *b) nogil
    void sdsge_aff_logit_grad_inv_arr(const double *y, double *x, int64_t n,
                                      const double *a, const double *b) nogil

    void sdsge_aff_logit_ldet_abs_jac_fwd(const double *x, double *y,
                                          const double *a, const double *b) nogil
    void sdsge_aff_logit_ldet_abs_jac_fwd_arr(const double *x, double *y,
                                              int64_t n, const double *a,
                                              const double *b) nogil

    void sdsge_aff_logit_ldet_abs_jac_inv(const double *y, double *x,
                                          const double *a, const double *b) nogil
    void sdsge_aff_logit_ldet_abs_jac_inv_arr(const double *y, double *x,
                                              int64_t n, const double *a,
                                              const double *b) nogil

    void sdsge_aff_logit_grad_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_aff_logit_grad_ldet_abs_jac_inv_arr(const double *y,
                                                   double *x, int64_t n) nogil

    # AFFINE PROBIT TRANSFORM
    void sdsge_aff_probit_fwd(const double *x, double *y,
                              const double *a, const double *b) nogil
    void sdsge_aff_probit_fwd_arr(const double *x, double *y, int64_t n,
                                  const double *a, const double *b) nogil

    void sdsge_aff_probit_inv(const double *y, double *x,
                              const double *a, const double *b) nogil
    void sdsge_aff_probit_inv_arr(const double *y, double *x, int64_t n,
                                  const double *a, const double *b) nogil

    void sdsge_aff_probit_grad_fwd(const double *x, double *y,
                                   const double *a, const double *b) nogil
    void sdsge_aff_probit_grad_fwd_arr(const double *x, double *y, int64_t n,
                                       const double *a, const double *b) nogil

    void sdsge_aff_probit_grad_inv(const double *y, double *x,
                                   const double *a, const double *b) nogil
    void sdsge_aff_probit_grad_inv_arr(const double *y, double *x, int64_t n,
                                       const double *a, const double *b) nogil

    void sdsge_aff_probit_ldet_abs_jac_fwd(const double *x, double *y,
                                           const double *a, const double *b) nogil
    void sdsge_aff_probit_ldet_abs_jac_fwd_arr(const double *x, double *y,
                                               int64_t n, const double *a,
                                               const double *b) nogil

    void sdsge_aff_probit_ldet_abs_jac_inv(const double *y, double *x,
                                           const double *a, const double *b) nogil
    void sdsge_aff_probit_ldet_abs_jac_inv_arr(const double *y, double *x,
                                               int64_t n, const double *a,
                                               const double *b) nogil

    void sdsge_aff_probit_grad_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_aff_probit_grad_ldet_abs_jac_inv_arr(const double *y,
                                                    double *x, int64_t n) nogil

    # SOFTPLUS TRANSFORM
    void sdsge_softplus_fwd(const double *x, double *y) nogil
    void sdsge_softplus_fwd_arr(const double *x, double *y, int64_t n) nogil

    void sdsge_softplus_inv(const double *y, double *x) nogil
    void sdsge_softplus_inv_arr(const double *y, double *x, int64_t n) nogil

    void sdsge_softplus_grad_fwd(const double *x, double *y) nogil
    void sdsge_softplus_grad_fwd_arr(const double *x, double *y, int64_t n) nogil

    void sdsge_softplus_grad_inv(const double *y, double *x) nogil
    void sdsge_softplus_grad_inv_arr(const double *y, double *x, int64_t n) nogil

    void sdsge_softplus_ldet_abs_jac_fwd(const double *x, double *y) nogil
    void sdsge_softplus_ldet_abs_jac_fwd_arr(const double *x, double *y,
                                             int64_t n) nogil

    void sdsge_softplus_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_softplus_ldet_abs_jac_inv_arr(const double *y, double *x,
                                             int64_t n) nogil

    void sdsge_softplus_grad_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_softplus_grad_ldet_abs_jac_inv_arr(const double *y, double *x,
                                                  int64_t n) nogil

    # LOWER BOUNDED TRANSFORM
    void sdsge_lower_fwd(const double *x, double *y, const double *low) nogil
    void sdsge_lower_fwd_arr(const double *x, double *y, int64_t n,
                             const double *low) nogil

    void sdsge_lower_inv(const double *y, double *x, const double *low) nogil
    void sdsge_lower_inv_arr(const double *y, double *x, int64_t n,
                             const double *low) nogil

    void sdsge_lower_grad_fwd(const double *x, double *y, const double *low) nogil
    void sdsge_lower_grad_fwd_arr(const double *x, double *y, int64_t n,
                                  const double *low) nogil

    void sdsge_lower_grad_inv(const double *y, double *x) nogil
    void sdsge_lower_grad_inv_arr(const double *y, double *x, int64_t n) nogil

    void sdsge_lower_ldet_abs_jac_fwd(const double *x, double *y,
                                      const double *low) nogil
    void sdsge_lower_ldet_abs_jac_fwd_arr(const double *x, double *y, int64_t n,
                                          const double *low) nogil

    void sdsge_lower_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_lower_ldet_abs_jac_inv_arr(const double *y, double *x,
                                          int64_t n) nogil

    void sdsge_lower_grad_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_lower_grad_ldet_abs_jac_inv_arr(const double *y, double *x,
                                               int64_t n) nogil

    # UPPER BOUNDED TRANSFORM
    void sdsge_upper_fwd(const double *x, double *y, const double *high) nogil
    void sdsge_upper_fwd_arr(const double *x, double *y, int64_t n,
                             const double *high) nogil

    void sdsge_upper_inv(const double *y, double *x, const double *high) nogil
    void sdsge_upper_inv_arr(const double *y, double *x, int64_t n,
                             const double *high) nogil

    void sdsge_upper_grad_fwd(const double *x, double *y, const double *high) nogil
    void sdsge_upper_grad_fwd_arr(const double *x, double *y, int64_t n,
                                  const double *high) nogil

    void sdsge_upper_grad_inv(const double *y, double *x) nogil
    void sdsge_upper_grad_inv_arr(const double *y, double *x, int64_t n) nogil

    void sdsge_upper_ldet_abs_jac_fwd(const double *x, double *y,
                                      const double *high) nogil
    void sdsge_upper_ldet_abs_jac_fwd_arr(const double *x, double *y, int64_t n,
                                          const double *high) nogil

    void sdsge_upper_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_upper_ldet_abs_jac_inv_arr(const double *y, double *x,
                                          int64_t n) nogil

    void sdsge_upper_grad_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_upper_grad_ldet_abs_jac_inv_arr(const double *y, double *x,
                                               int64_t n) nogil

    # TANH TRANSFORM
    void sdsge_tanh_fwd(const double *x, double *y) nogil
    void sdsge_tanh_fwd_arr(const double *x, double *y, int64_t n) nogil

    void sdsge_tanh_inv(const double *y, double *x) nogil
    void sdsge_tanh_inv_arr(const double *y, double *x, int64_t n) nogil

    void sdsge_tanh_grad_fwd(const double *x, double *y) nogil
    void sdsge_tanh_grad_fwd_arr(const double *x, double *y, int64_t n) nogil

    void sdsge_tanh_grad_inv(const double *y, double *x) nogil
    void sdsge_tanh_grad_inv_arr(const double *y, double *x, int64_t n) nogil

    void sdsge_tanh_ldet_abs_jac_fwd(const double *x, double *y) nogil
    void sdsge_tanh_ldet_abs_jac_fwd_arr(const double *x, double *y,
                                         int64_t n) nogil

    void sdsge_tanh_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_tanh_ldet_abs_jac_inv_arr(const double *y, double *x,
                                         int64_t n) nogil

    void sdsge_tanh_grad_ldet_abs_jac_inv(const double *y, double *x) nogil
    void sdsge_tanh_grad_ldet_abs_jac_inv_arr(const double *y, double *x,
                                              int64_t n) nogil

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


# PROBIT TRANSFORM

def probit_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_probit_fwd(&xx, &yy)
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
        sdsge_probit_fwd_arr(&xv[0], &yv[0], n)
    return out


def probit_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_probit_inv(&yy, &xx)
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
        sdsge_probit_inv_arr(&yv[0], &xv[0], n)
    return out


def probit_grad_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_probit_grad_fwd(&xx, &yy)
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
        sdsge_probit_grad_fwd_arr(&xv[0], &yv[0], n)
    return out


def probit_grad_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_probit_grad_inv(&yy, &xx)
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
        sdsge_probit_grad_inv_arr(&yv[0], &xv[0], n)
    return out


def probit_ldet_abs_jac_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_probit_ldet_abs_jac_fwd(&xx, &yy)
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
        sdsge_probit_ldet_abs_jac_fwd_arr(&xv[0], &yv[0], n)
    return out


def probit_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_probit_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_probit_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


def probit_grad_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_probit_grad_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_probit_grad_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


# AFFINE LOGIT TRANSFORM

def aff_logit_fwd(x, low, high):
    cdef double xx, yy
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_aff_logit_fwd(&xx, &yy, &aa, &bb)
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
        sdsge_aff_logit_fwd_arr(&xv[0], &yv[0], n, &aa, &bb)
    return out


def aff_logit_inv(y, low, high):
    cdef double yy, xx
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_aff_logit_inv(&yy, &xx, &aa, &bb)
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
        sdsge_aff_logit_inv_arr(&yv[0], &xv[0], n, &aa, &bb)
    return out


def aff_logit_grad_fwd(x, low, high):
    cdef double xx, yy
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_aff_logit_grad_fwd(&xx, &yy, &aa, &bb)
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
        sdsge_aff_logit_grad_fwd_arr(&xv[0], &yv[0], n, &aa, &bb)
    return out


def aff_logit_grad_inv(y, low, high):
    cdef double yy, xx
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_aff_logit_grad_inv(&yy, &xx, &aa, &bb)
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
        sdsge_aff_logit_grad_inv_arr(&yv[0], &xv[0], n, &aa, &bb)
    return out


def aff_logit_ldet_abs_jac_fwd(x, low, high):
    cdef double xx, yy
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_aff_logit_ldet_abs_jac_fwd(&xx, &yy, &aa, &bb)
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
        sdsge_aff_logit_ldet_abs_jac_fwd_arr(&xv[0], &yv[0], n, &aa, &bb)
    return out


def aff_logit_ldet_abs_jac_inv(y, low, high):
    cdef double yy, xx
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_aff_logit_ldet_abs_jac_inv(&yy, &xx, &aa, &bb)
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
        sdsge_aff_logit_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n, &aa, &bb)
    return out


def aff_logit_grad_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_aff_logit_grad_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_aff_logit_grad_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


# AFFINE PROBIT TRANSFORM

def aff_probit_fwd(x, low, high):
    cdef double xx, yy
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_aff_probit_fwd(&xx, &yy, &aa, &bb)
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
        sdsge_aff_probit_fwd_arr(&xv[0], &yv[0], n, &aa, &bb)
    return out


def aff_probit_inv(y, low, high):
    cdef double yy, xx
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_aff_probit_inv(&yy, &xx, &aa, &bb)
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
        sdsge_aff_probit_inv_arr(&yv[0], &xv[0], n, &aa, &bb)
    return out


def aff_probit_grad_fwd(x, low, high):
    cdef double xx, yy
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_aff_probit_grad_fwd(&xx, &yy, &aa, &bb)
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
        sdsge_aff_probit_grad_fwd_arr(&xv[0], &yv[0], n, &aa, &bb)
    return out


def aff_probit_grad_inv(y, low, high):
    cdef double yy, xx
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_aff_probit_grad_inv(&yy, &xx, &aa, &bb)
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
        sdsge_aff_probit_grad_inv_arr(&yv[0], &xv[0], n, &aa, &bb)
    return out


def aff_probit_ldet_abs_jac_fwd(x, low, high):
    cdef double xx, yy
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_aff_probit_ldet_abs_jac_fwd(&xx, &yy, &aa, &bb)
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
        sdsge_aff_probit_ldet_abs_jac_fwd_arr(&xv[0], &yv[0], n, &aa, &bb)
    return out


def aff_probit_ldet_abs_jac_inv(y, low, high):
    cdef double yy, xx
    cdef double aa = <double> low
    cdef double bb = <double> high
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_aff_probit_ldet_abs_jac_inv(&yy, &xx, &aa, &bb)
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
        sdsge_aff_probit_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n, &aa, &bb)
    return out


def aff_probit_grad_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_aff_probit_grad_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_aff_probit_grad_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


# SOFTPLUS TRANSFORM

def softplus_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_softplus_fwd(&xx, &yy)
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
        sdsge_softplus_fwd_arr(&xv[0], &yv[0], n)
    return out


def softplus_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_softplus_inv(&yy, &xx)
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
        sdsge_softplus_inv_arr(&yv[0], &xv[0], n)
    return out


def softplus_grad_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_softplus_grad_fwd(&xx, &yy)
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
        sdsge_softplus_grad_fwd_arr(&xv[0], &yv[0], n)
    return out


def softplus_grad_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_softplus_grad_inv(&yy, &xx)
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
        sdsge_softplus_grad_inv_arr(&yv[0], &xv[0], n)
    return out


def softplus_ldet_abs_jac_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_softplus_ldet_abs_jac_fwd(&xx, &yy)
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
        sdsge_softplus_ldet_abs_jac_fwd_arr(&xv[0], &yv[0], n)
    return out


def softplus_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_softplus_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_softplus_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


def softplus_grad_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_softplus_grad_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_softplus_grad_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


# LOWER BOUNDED TRANSFORM

def lower_fwd(x, low):
    cdef double xx, yy
    cdef double ll = <double> low
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_lower_fwd(&xx, &yy, &ll)
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
        sdsge_lower_fwd_arr(&xv[0], &yv[0], n, &ll)
    return out


def lower_inv(y, low):
    cdef double yy, xx
    cdef double ll = <double> low
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_lower_inv(&yy, &xx, &ll)
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
        sdsge_lower_inv_arr(&yv[0], &xv[0], n, &ll)
    return out


def lower_grad_fwd(x, low):
    cdef double xx, yy
    cdef double ll = <double> low
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_lower_grad_fwd(&xx, &yy, &ll)
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
        sdsge_lower_grad_fwd_arr(&xv[0], &yv[0], n, &ll)
    return out


def lower_grad_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_lower_grad_inv(&yy, &xx)
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
        sdsge_lower_grad_inv_arr(&yv[0], &xv[0], n)
    return out


def lower_ldet_abs_jac_fwd(x, low):
    cdef double xx, yy
    cdef double ll = <double> low
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_lower_ldet_abs_jac_fwd(&xx, &yy, &ll)
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
        sdsge_lower_ldet_abs_jac_fwd_arr(&xv[0], &yv[0], n, &ll)
    return out


def lower_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_lower_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_lower_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


def lower_grad_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_lower_grad_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_lower_grad_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


# UPPER BOUNDED TRANSFORM

def upper_fwd(x, high):
    cdef double xx, yy
    cdef double hh = <double> high
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_upper_fwd(&xx, &yy, &hh)
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
        sdsge_upper_fwd_arr(&xv[0], &yv[0], n, &hh)
    return out


def upper_inv(y, high):
    cdef double yy, xx
    cdef double hh = <double> high
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_upper_inv(&yy, &xx, &hh)
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
        sdsge_upper_inv_arr(&yv[0], &xv[0], n, &hh)
    return out


def upper_grad_fwd(x, high):
    cdef double xx, yy
    cdef double hh = <double> high
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_upper_grad_fwd(&xx, &yy, &hh)
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
        sdsge_upper_grad_fwd_arr(&xv[0], &yv[0], n, &hh)
    return out


def upper_grad_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_upper_grad_inv(&yy, &xx)
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
        sdsge_upper_grad_inv_arr(&yv[0], &xv[0], n)
    return out


def upper_ldet_abs_jac_fwd(x, high):
    cdef double xx, yy
    cdef double hh = <double> high
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_upper_ldet_abs_jac_fwd(&xx, &yy, &hh)
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
        sdsge_upper_ldet_abs_jac_fwd_arr(&xv[0], &yv[0], n, &hh)
    return out


def upper_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_upper_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_upper_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


def upper_grad_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_upper_grad_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_upper_grad_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


# TANH TRANSFORM

def tanh_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_tanh_fwd(&xx, &yy)
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
        sdsge_tanh_fwd_arr(&xv[0], &yv[0], n)
    return out


def tanh_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_tanh_inv(&yy, &xx)
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
        sdsge_tanh_inv_arr(&yv[0], &xv[0], n)
    return out


def tanh_grad_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_tanh_grad_fwd(&xx, &yy)
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
        sdsge_tanh_grad_fwd_arr(&xv[0], &yv[0], n)
    return out


def tanh_grad_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_tanh_grad_inv(&yy, &xx)
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
        sdsge_tanh_grad_inv_arr(&yv[0], &xv[0], n)
    return out


def tanh_ldet_abs_jac_fwd(x):
    cdef double xx, yy
    if not isinstance(x, (np.ndarray, list, tuple)):
        xx = <double> x
        with nogil:
            sdsge_tanh_ldet_abs_jac_fwd(&xx, &yy)
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
        sdsge_tanh_ldet_abs_jac_fwd_arr(&xv[0], &yv[0], n)
    return out


def tanh_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_tanh_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_tanh_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out


def tanh_grad_ldet_abs_jac_inv(y):
    cdef double yy, xx
    if not isinstance(y, (np.ndarray, list, tuple)):
        yy = <double> y
        with nogil:
            sdsge_tanh_grad_ldet_abs_jac_inv(&yy, &xx)
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
        sdsge_tanh_grad_ldet_abs_jac_inv_arr(&yv[0], &xv[0], n)
    return out
