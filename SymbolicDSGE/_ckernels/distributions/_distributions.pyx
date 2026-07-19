# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Thin Cython shim for the native distribution kernels.

No numeric logic here -- only argument marshalling and the C call. The numba
references now live in the test oracles (tests/_oracles/distributions.py), so
this is the sole production path and the parity tests pin them together.

``ndtri_as241`` / ``erfinv_from_as241`` are the Wichura AS 241 inverse-normal
primitives from ``_common/as241.c``. The vectorized ``ndtri_as241_into`` owns its
own casting: it takes an untyped array, forces a C-contiguous float64 copy, and
returns a new array shaped like the input. The precise NumPy types live in the
``.pyi`` stub, not the signature.
"""

import numpy as np
from libc.stdint cimport int64_t


cdef extern from "as241.h":
    double sdsge_ndtri_as241(double p) nogil
    double sdsge_erfinv_from_as241(double y) nogil
    void sdsge_ndtri_as241_into(const double *p, int64_t n, double *out) nogil


def ndtri_as241(p):
    """Inverse standard-normal CDF (probit). ``-inf`` at 0, ``+inf`` at 1."""
    cdef double pp = <double> p
    cdef double out
    with nogil:
        out = sdsge_ndtri_as241(pp)
    return out


def erfinv_from_as241(y):
    """Inverse error function, ``erfinv(y) = ndtri(0.5*(y+1)/sqrt(2))``."""
    cdef double yy = <double> y
    cdef double out
    with nogil:
        out = sdsge_erfinv_from_as241(yy)
    return out


def ndtri_as241_into(p):
    """Elementwise ndtri; returns a new float64 array shaped like ``p``."""
    arr = np.ascontiguousarray(p, dtype=np.float64)
    out = np.empty_like(arr)
    cdef int64_t n = arr.size
    if n == 0:
        return out
    cdef double[::1] pv = arr.reshape(-1)
    cdef double[::1] ov = out.reshape(-1)
    with nogil:
        sdsge_ndtri_as241_into(&pv[0], n, &ov[0])
    return out
