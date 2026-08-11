"""Native OccBin kernels: per-regime pencils and the guess-and-verify solve."""

from ._occbin import (
    occbin_solve,
    regime_pencil,
)

__all__ = [
    "occbin_solve",
    "regime_pencil",
]
