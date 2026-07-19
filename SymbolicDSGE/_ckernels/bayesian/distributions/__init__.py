"""Native distribution kernels.

Re-exports the compiled ``_distributions`` extension. Currently the Wichura
AS 241 inverse-normal primitives (``ndtri_as241`` / ``erfinv_from_as241``); more
distribution kernels land here as they are ported.
"""

from ._distributions import (
    erfinv_from_as241,
    ndtri_as241,
    ndtri_as241_into,
)

__all__ = [
    "erfinv_from_as241",
    "ndtri_as241",
    "ndtri_as241_into",
]
