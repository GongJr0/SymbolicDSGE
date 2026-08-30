"""``.sdsge`` bundle serialization and container (UI-independent)."""

from .builder import BundleBuilder
from .container import BundleArchive, write_bundle
from .loader import LoadedBundle, LoadedEstimation, LoadedMC, build_from
from .manifest import (
    SDSGE_FORMAT_VERSION,
    Manifest,
    Member,
)
from .parquet import (
    arrays_to_parquet,
    collapse_columns,
    columns_from_parquet,
    columns_to_parquet,
    csv_to_columns,
    csv_to_json,
    frame_to_json,
    from_parquet,
    from_parquet_columns,
    to_parquet,
    trace_to_csv,
)

__all__ = [
    # parquet seam
    "to_parquet",
    "from_parquet",
    "columns_to_parquet",
    "columns_from_parquet",
    "csv_to_json",
    "csv_to_columns",
    "trace_to_csv",
    "frame_to_json",
    "from_parquet_columns",
    "collapse_columns",
    "arrays_to_parquet",
    # manifest
    "Manifest",
    "Member",
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
