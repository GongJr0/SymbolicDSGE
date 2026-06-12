from __future__ import annotations

import json

import numpy as np
import pytest

from SymbolicDSGE.bundle import csv_to_json, from_parquet, to_parquet, trace_to_json


def _records(ndjson: bytes) -> list[dict]:
    return [json.loads(line) for line in ndjson.splitlines() if line.strip()]


def test_to_parquet_round_trip_via_ndjson() -> None:
    rows = [{"x": 0.5, "y": 1, "label": "a"}, {"x": -0.5, "y": 2, "label": "b"}]
    ndjson = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()

    parquet = to_parquet(ndjson)
    assert isinstance(parquet, bytes) and len(parquet) > 0
    assert _records(from_parquet(parquet)) == rows


def test_csv_to_json_infers_column_types_and_nulls() -> None:
    csv_text = "x,n,label\n0.5,1,a\n,2,b\n1.5,3,\n"
    records = _records(csv_to_json(csv_text))

    assert records == [
        {"x": 0.5, "n": 1, "label": "a"},
        {"x": None, "n": 2, "label": "b"},
        {"x": 1.5, "n": 3, "label": None},
    ]
    # x is float, n is int (preserved through types)
    assert isinstance(records[0]["x"], float)
    assert isinstance(records[0]["n"], int)


def test_csv_round_trips_through_parquet() -> None:
    csv_text = "x,y\n0.1,10\n0.2,20\n0.3,30\n"
    parquet = to_parquet(csv_to_json(csv_text))
    assert _records(from_parquet(parquet)) == [
        {"x": 0.1, "y": 10},
        {"x": 0.2, "y": 20},
        {"x": 0.3, "y": 30},
    ]


def test_trace_to_json_expands_2d_and_nulls_nonfinite() -> None:
    columns = {
        "theta": np.array([[1.0, 2.0], [3.0, np.nan]]),  # (n=2, k=2)
        "logpost": np.array([-1.5, -2.5]),
        "status": np.array([0, -5], dtype=np.int64),
    }
    records = _records(trace_to_json(columns))

    assert records == [
        {"theta.0": 1.0, "theta.1": 2.0, "logpost": -1.5, "status": 0},
        {"theta.0": 3.0, "theta.1": None, "logpost": -2.5, "status": -5},
    ]
    assert isinstance(records[0]["status"], int)


def test_trace_to_json_round_trips_through_parquet() -> None:
    rng = np.random.default_rng(0)
    columns = {
        "samples": rng.normal(size=(50, 3)),  # MCMC-like (n_draws, n_params)
        "logpost": rng.normal(size=50),
    }
    parquet = to_parquet(trace_to_json(columns))
    records = _records(from_parquet(parquet))
    assert len(records) == 50
    np.testing.assert_allclose(
        [r["samples.0"] for r in records], columns["samples"][:, 0]
    )
    np.testing.assert_allclose([r["logpost"] for r in records], columns["logpost"])


def test_trace_to_json_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        trace_to_json({"a": np.zeros(3), "b": np.zeros(4)})


def test_encodings_override_is_forwarded() -> None:
    ndjson = trace_to_json({"x": np.random.default_rng(1).normal(size=500)})
    # explicit BSS vs dictionary should both round-trip and differ in size
    bss = to_parquet(ndjson, encodings={"x": "bss"})
    dictionary = to_parquet(ndjson, encodings={"x": "dictionary"})
    assert len(bss) < len(dictionary)
    assert _records(from_parquet(bss)) == _records(from_parquet(dictionary))
