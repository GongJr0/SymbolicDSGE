"""Parquet serialization for ``.sdsge`` bundles.

A single gateway, :func:`to_parquet`, encodes newline-delimited JSON to Parquet
bytes via the ``parquet-engine`` extension. Everything that needs Parquet output
produces NDJSON first (:func:`csv_to_json` for raw data files, :func:`trace_to_json`
for MCMC / Monte Carlo pipeline traces) and passes it through :func:`to_parquet`.

This lives in the core library (not behind the ``[ui]`` extra) so a bundle can be
produced without the UI dependencies.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

import parquet_engine

__all__ = ["to_parquet", "from_parquet", "csv_to_json", "trace_to_json"]


def to_parquet(
    json_data: bytes | str,
    *,
    encodings: Mapping[str, str] | None = None,
    compression_level: int = 3,
) -> bytes:
    """Encode newline-delimited JSON into Parquet bytes.

    The single seam every bundle member's bulk data flows through. ``encodings``
    optionally pins a per-column Parquet encoding (``"bss"``/``"byte_stream_split"``,
    ``"dictionary"``, ``"plain"``); columns left out are inferred (float ->
    byte-stream-split, otherwise dictionary).
    """
    if isinstance(json_data, str):
        json_data = json_data.encode("utf-8")
    enc = dict(encodings) if encodings is not None else None
    return cast(bytes, parquet_engine.encode(json_data, enc, compression_level))


def from_parquet(data: bytes) -> str:
    """Decode Parquet bytes back into newline-delimited JSON."""
    return cast(str, parquet_engine.decode(data))


def csv_to_json(data: bytes | str, *, dialect: str = "excel") -> bytes:
    """Convert a CSV (header row + data rows) into newline-delimited JSON.

    Per-column type inference keeps numeric data numeric (so the engine can pick
    byte-stream-split for float columns): a column is emitted as ``int`` if every
    non-empty cell parses as an integer, else ``float`` if every cell parses as a
    finite float, else left as a string. Empty cells and non-finite floats become
    JSON ``null``.
    """
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    reader = csv.reader(io.StringIO(text), dialect=dialect)
    rows = list(reader)
    if not rows:
        return b""

    header, body = rows[0], rows[1:]
    n_cols = len(header)
    columns: list[list[str | None]] = [[] for _ in range(n_cols)]
    for row in body:
        for j in range(n_cols):
            cell = row[j] if j < len(row) else ""
            columns[j].append(cell if cell != "" else None)

    converters = [_column_converter(col) for col in columns]
    out = io.BytesIO()
    for i in range(len(body)):
        obj = {header[j]: converters[j](columns[j][i]) for j in range(n_cols)}
        out.write(json.dumps(obj, allow_nan=False).encode("utf-8"))
        out.write(b"\n")
    return out.getvalue()


def trace_to_json(columns: Mapping[str, Any]) -> bytes:
    """Convert columnar trace data into newline-delimited JSON.

    Each value is a 1-D array, or a 2-D array ``(n, k)`` that is expanded into
    columns ``f"{name}.{j}"`` for ``j`` in ``range(k)``. Used for MCMC posterior
    samples and Monte Carlo pipeline result traces. All columns must share the
    leading length ``n``; non-finite floats become JSON ``null``.
    """
    if not columns:
        return b""

    flat: dict[str, NDArray[Any]] = {}
    length: int | None = None
    for name, value in columns.items():
        arr = np.asarray(value)
        if arr.ndim == 1:
            expanded = {name: arr}
        elif arr.ndim == 2:
            expanded = {f"{name}.{j}": arr[:, j] for j in range(arr.shape[1])}
        else:
            raise ValueError(
                f"trace column {name!r} must be 1-D or 2-D, got {arr.ndim}-D"
            )
        for key, col in expanded.items():
            if length is None:
                length = int(col.shape[0])
            elif col.shape[0] != length:
                raise ValueError(
                    f"trace columns must share length; {key!r} has {col.shape[0]}, "
                    f"expected {length}"
                )
            flat[key] = col

    assert length is not None
    names = list(flat)
    out = io.BytesIO()
    for i in range(length):
        obj = {name: _json_scalar(flat[name][i]) for name in names}
        out.write(json.dumps(obj, allow_nan=False).encode("utf-8"))
        out.write(b"\n")
    return out.getvalue()


def _column_converter(values: list[str | None]) -> Callable[[str | None], Any]:
    non_null = [v for v in values if v is not None]
    if non_null and all(_is_int(v) for v in non_null):
        return lambda v: None if v is None else int(v)
    if non_null and all(_is_float(v) for v in non_null):
        return _to_float_or_none
    return lambda v: v


def _is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _to_float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value.item() if hasattr(value, "item") else value
