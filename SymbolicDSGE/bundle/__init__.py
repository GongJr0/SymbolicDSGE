"""``.sdsge`` bundle serialization and container (UI-independent)."""

from .builder import BundleBuilder
from .container import BundleArchive, write_bundle
from .loader import LoadedBundle, LoadedEstimation, LoadedMC, build_from
from ..core.shock_generators import ShockParameters
from .manifest import (
    SDSGE_FORMAT_VERSION,
    Manifest,
    Member,
    SimSpec,
)
from .parquet import (
    arrays_from_parquet,
    arrays_to_parquet,
    collapse_columns,
    csv_to_columns,
    csv_to_json,
    frame_to_json,
    from_parquet,
    from_parquet_columns,
    to_parquet,
    trace_to_csv,
    trace_to_json,
)

__all__ = [
    # parquet seam
    "to_parquet",
    "from_parquet",
    "csv_to_json",
    "csv_to_columns",
    "trace_to_json",
    "trace_to_csv",
    "frame_to_json",
    "from_parquet_columns",
    "collapse_columns",
    "arrays_to_parquet",
    "arrays_from_parquet",
    # manifest
    "Manifest",
    "Member",
    "SimSpec",
    "ShockParameters",
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
