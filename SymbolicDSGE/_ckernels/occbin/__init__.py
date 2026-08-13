"""Native OccBin kernels: per-regime pencils and the guess-and-verify solve."""

from ._occbin import (
    occbin_sim,
    occbin_solve1,
    regime_pencil,
)

__all__ = [
    "occbin_sim",
    "occbin_solve1",
    "regime_pencil",
]
