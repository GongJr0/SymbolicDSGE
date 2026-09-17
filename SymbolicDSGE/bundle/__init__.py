"""``.sdsge`` bundle serialization and container (UI-independent)."""

from .builder import BundleBuilder
from .container import write_bundle
from .loader import LoadedBundle, LoadedEstimation, LoadedMC, load_bundle
from .manifest import (
    SDSGE_FORMAT_VERSION,
)

__all__ = [
    "SDSGE_FORMAT_VERSION",
    # container
    "write_bundle",
    # build / load
    "BundleBuilder",
    "load_bundle",
    "LoadedBundle",
    "LoadedEstimation",
    "LoadedMC",
]
