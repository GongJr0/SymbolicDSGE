from __future__ import annotations

import json

import numpy as np
import pytest

from SymbolicDSGE.bundle import (
    arrays_to_parquet,
    collapse_columns,
    columns_from_parquet,
    columns_to_parquet,
    csv_to_json,
    from_parquet,
    to_parquet,
)


def _records(ndjson: bytes) -> list[dict]:
    return [json.loads(line) for line in ndjson.splitlines() if line.strip()]


def _restore(data: bytes, shapes: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """Rebuild N-D arrays the way the loader does, from the shape manifest."""
    columns = collapse_columns(columns_from_parquet(data))
    return {
        name: np.asarray(columns[name]).reshape(tuple(shape))
        for name, shape in shapes.items()
    }


def test_arrays_to_parquet_round_trips_3d_per_rep() -> None:
    rng = np.random.default_rng(0)
    states = rng.normal(size=(4, 5, 2))  # n_rep, T, k
    observables = rng.normal(size=(4, 5, 3))

    data, shapes = arrays_to_parquet({"states": states, "observables": observables})
    assert shapes == {"states": [4, 5, 2], "observables": [4, 5, 3]}

    restored = _restore(data, shapes)
    np.testing.assert_allclose(restored["states"], states)
    np.testing.assert_allclose(restored["observables"], observables)


def test_arrays_to_parquet_round_trips_mixed_2d_and_1d() -> None:
    states = np.arange(15, dtype=float).reshape(5, 3)  # shared (T, k)
    vector = np.linspace(0.0, 1.0, 5)  # 1-D (T,)

    data, shapes = arrays_to_parquet({"states": states, "raw:eps": vector})
    assert shapes == {"states": [5, 3], "raw:eps": [5]}

    restored = _restore(data, shapes)
    np.testing.assert_allclose(restored["states"], states)
    np.testing.assert_allclose(restored["raw:eps"], vector)


def test_arrays_to_parquet_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one array"):
        arrays_to_parquet({})


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


def test_columns_to_parquet_expands_2d_and_keeps_dtypes() -> None:
    columns = {
        "theta": np.array([[1.0, 2.0], [3.0, np.nan]]),  # (n=2, k=2)
        "logpost": np.array([-1.5, -2.5]),
        "status": np.array([0, -5], dtype=np.int64),
    }
    out = columns_from_parquet(columns_to_parquet(columns))

    assert list(out) == ["theta.0", "theta.1", "logpost", "status"]
    assert out["status"].dtype == np.int64
    assert out["theta.0"].dtype == np.float64
    np.testing.assert_array_equal(out["status"], [0, -5])
    np.testing.assert_array_equal(out["theta.0"], [1.0, 3.0])
    # The binary path stores a non-finite float as itself, not as a null.
    assert out["theta.1"][0] == 2.0
    assert np.isnan(out["theta.1"][1])


def test_columns_to_parquet_round_trips_exactly() -> None:
    rng = np.random.default_rng(0)
    columns = {
        "samples": rng.normal(size=(50, 3)),  # MCMC-like (n_draws, n_params)
        "logpost": rng.normal(size=50),
    }
    out = columns_from_parquet(columns_to_parquet(columns))
    for j in range(3):
        np.testing.assert_array_equal(out[f"samples.{j}"], columns["samples"][:, j])
    np.testing.assert_array_equal(out["logpost"], columns["logpost"])


def test_columns_to_parquet_preserves_infinities() -> None:
    columns = {"x": np.array([np.inf, -np.inf, 0.0])}
    out = columns_from_parquet(columns_to_parquet(columns))
    np.testing.assert_array_equal(out["x"], columns["x"])


def test_columns_from_parquet_returns_writable_arrays() -> None:
    out = columns_from_parquet(columns_to_parquet({"x": np.zeros(4)}))
    out["x"][0] = 1.0  # a loaded trace is no more frozen than a computed one
    assert out["x"][0] == 1.0


def test_columns_to_parquet_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="share length"):
        columns_to_parquet({"a": np.zeros(3), "b": np.zeros(4)})


def test_columns_to_parquet_rejects_non_numeric() -> None:
    with pytest.raises(TypeError, match="not numeric"):
        columns_to_parquet({"label": np.array(["a", "b"])})


def test_encodings_override_is_forwarded() -> None:
    column = {"x": np.random.default_rng(1).normal(size=500)}
    # explicit BSS vs dictionary should both round-trip and differ in size
    bss = columns_to_parquet(column, encodings={"x": "bss"})
    dictionary = columns_to_parquet(column, encodings={"x": "dictionary"})
    assert len(bss) < len(dictionary)
    np.testing.assert_array_equal(
        columns_from_parquet(bss)["x"], columns_from_parquet(dictionary)["x"]
    )
