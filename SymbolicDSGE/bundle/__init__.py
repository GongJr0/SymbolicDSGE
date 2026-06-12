"""``.sdsge`` bundle serialization (UI-independent)."""

from .parquet import csv_to_json, from_parquet, to_parquet, trace_to_json

__all__ = ["to_parquet", "from_parquet", "csv_to_json", "trace_to_json"]
