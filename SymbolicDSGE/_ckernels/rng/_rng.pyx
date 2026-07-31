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

from libc.stdint cimport int64_t, uint64_t

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

import numpy as np

from numpy.random cimport bitgen_t


cdef extern from "rng.h":
    void sdsge_rng_standard_normal_fill(bitgen_t *bg, int64_t n,
                                        double *out) nogil
    void sdsge_rng_standard_uniform_fill(bitgen_t *bg, int64_t n,
                                         double *out) nogil


cdef extern from "philox.h":
    ctypedef struct sdsge_philox_state:
        pass

    void sdsge_philox_seed(sdsge_philox_state *st, uint64_t key0,
                           uint64_t key1, uint64_t stream0,
                           uint64_t stream1) nogil
    uint64_t sdsge_philox_next_u64(sdsge_philox_state *st) nogil
    void sdsge_philox_standard_normal_fill(sdsge_philox_state *st, int64_t n,
                                           double *out) nogil
    void sdsge_philox_standard_uniform_fill(sdsge_philox_state *st, int64_t n,
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


# --- Philox surface --------------------------------------------------------
# The counter-based engine the Monte Carlo shock draw runs on. Native callers
# use the C kernels directly (the state is stack-held inside the hot loop);
# these wrappers exist so the engine's reproducibility can be driven and pinned
# from Python. Draws here are NOT numpy-parity: the engine is ours.


def philox_standard_normal(uint64_t key0, uint64_t key1, uint64_t stream0,
                           uint64_t stream1, int64_t n):
    """``n`` standard normal draws from the Philox stream ``(key, stream)``.

    Seeding the same four words always replays the same draws, and distinct
    words give independent streams.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    cdef sdsge_philox_state st
    cdef double[::1] outv = out
    with nogil:
        sdsge_philox_seed(&st, key0, key1, stream0, stream1)
        sdsge_philox_standard_normal_fill(&st, n, &outv[0])
    return out


def philox_standard_uniform(uint64_t key0, uint64_t key1, uint64_t stream0,
                            uint64_t stream1, int64_t n):
    """``n`` standard uniform draws in [0, 1) from the stream ``(key, stream)``."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    cdef sdsge_philox_state st
    cdef double[::1] outv = out
    with nogil:
        sdsge_philox_seed(&st, key0, key1, stream0, stream1)
        sdsge_philox_standard_uniform_fill(&st, n, &outv[0])
    return out


def philox_raw(uint64_t key0, uint64_t key1, uint64_t stream0,
               uint64_t stream1, int64_t n):
    """``n`` raw 64-bit draws from the stream ``(key, stream)``.

    The engine's own output, ahead of any distribution transform.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    out = np.empty(n, dtype=np.uint64)
    if n == 0:
        return out
    cdef sdsge_philox_state st
    cdef uint64_t[::1] outv = out
    cdef int64_t i
    with nogil:
        sdsge_philox_seed(&st, key0, key1, stream0, stream1)
        for i in range(n):
            outv[i] = sdsge_philox_next_u64(&st)
    return out
