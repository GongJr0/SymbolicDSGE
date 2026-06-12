"""``.sdsge`` bundle serialization and container (UI-independent)."""

from .builder import BundleBuilder
from .container import BundleArchive, write_bundle
from .loader import LoadedBundle, LoadedEstimation, LoadedMC, build_from
from .manifest import (
    SDSGE_FORMAT_VERSION,
    Manifest,
    Member,
    ShockGeneration,
    SimSpec,
)
from .parquet import (
    collapse_columns,
    csv_to_json,
    from_parquet,
    from_parquet_columns,
    to_parquet,
    trace_to_json,
)

__all__ = [
    # parquet seam
    "to_parquet",
    "from_parquet",
    "csv_to_json",
    "trace_to_json",
    "from_parquet_columns",
    "collapse_columns",
    # manifest
    "Manifest",
    "Member",
    "SimSpec",
    "ShockGeneration",
    "SDSGE_FORMAT_VERSION",
    # container
    "write_bundle",
    "BundleArchive",
    # build / load
    "BundleBuilder",
    "build_from",
    "LoadedBundle",
    "LoadedEstimation",
    "LoadedMC",
]
