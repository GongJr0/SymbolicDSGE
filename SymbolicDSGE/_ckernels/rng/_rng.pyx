# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython surface for the native RNG bridge (issue #328).

Unwraps a numpy ``Generator``'s ``bit_generator.capsule`` into the borrowed
``bitgen_t*`` and hands it to the native fill kernels, so native draws advance
numpy's own PCG64 state. The draws are bit-identical to ``rng.standard_normal()``
/ ``rng.random()`` (the transform is numpy's, linked from ``npyrandom``); the
engine is numpy's live object, reached through the capsule pointer.

Lifetime: ``rng`` is an argument, so the Python owner of the borrowed pointer is
held for the whole call, including the ``nogil`` fill. The ``nogil`` block touches
only the raw pointer, never the Python object.
"""

from libc.stdint cimport int64_t

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

import numpy as np

from numpy.random cimport bitgen_t


cdef extern from "rng.h":
    void sdsge_rng_standard_normal_fill(bitgen_t *bg, int64_t n,
                                        double *out) nogil
    void sdsge_rng_standard_uniform_fill(bitgen_t *bg, int64_t n,
                                         double *out) nogil


# numpy tags the BitGenerator capsule with this exact name; PyCapsule_GetPointer
# rejects any mismatch, so a wrong/foreign capsule can't be dereferenced.
cdef const char *_CAPSULE_NAME = b"BitGenerator"


cdef bitgen_t *_bitgen_ptr(object rng) except NULL:
    """Borrow the ``bitgen_t*`` from a numpy ``Generator``. Caller must keep
    ``rng`` alive for as long as the pointer is used."""
    capsule = rng.bit_generator.capsule
    if not PyCapsule_IsValid(capsule, _CAPSULE_NAME):
        raise ValueError(
            "rng must be a numpy Generator exposing a valid BitGenerator capsule."
        )
    return <bitgen_t *>PyCapsule_GetPointer(capsule, _CAPSULE_NAME)


def standard_normal(object rng, int64_t n):
    """``n`` standard normal draws advancing ``rng``'s own PCG64 state.

    Bit-identical to ``rng.standard_normal(n)`` on the same generator state.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    cdef bitgen_t *bg = _bitgen_ptr(rng)
    cdef double[::1] outv = out
    with nogil:
        sdsge_rng_standard_normal_fill(bg, n, &outv[0])
    return out


def standard_uniform(object rng, int64_t n):
    """``n`` standard uniform draws in [0, 1) advancing ``rng``'s own PCG64 state.

    Bit-identical to ``rng.random(n)`` on the same generator state.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    cdef bitgen_t *bg = _bitgen_ptr(rng)
    cdef double[::1] outv = out
    with nogil:
        sdsge_rng_standard_uniform_fill(bg, n, &outv[0])
    return out
