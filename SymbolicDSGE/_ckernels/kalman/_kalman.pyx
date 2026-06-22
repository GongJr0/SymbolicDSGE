# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython shim for the kalman native kernels (placeholder).

Declare the C prototypes with ``cdef extern from "kalman.h"`` and add one ``def``
wrapper per entrypoint that maps NumPy memoryviews to ``double*``.
"""
