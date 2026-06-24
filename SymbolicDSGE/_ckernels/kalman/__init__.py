"""Native kalman kernels (linear filter hot loop).

Re-exports the compiled ``_kalman`` extension. If it is not built, importing
this module raises ``ImportError`` and the consumer (``SymbolicDSGE.kalman``)
falls back to its numba kernels.
"""

from ._kalman import kalman_hot_loop as kalman_hot_loop
