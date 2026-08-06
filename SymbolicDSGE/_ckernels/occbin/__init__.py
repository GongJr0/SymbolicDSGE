"""Native OccBin kernels (regime conditions evaluated over a simulated path).

Re-exports the compiled ``_occbin`` extension, which is mandatory: if it is not
built, importing this module (and the library) raises ``ImportError``.
"""

from ._occbin import MAX_CONSTRAINTS, constraint_path

__all__ = [
    "MAX_CONSTRAINTS",
    "constraint_path",
]
