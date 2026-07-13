"""Branch coverage for parquet NDJSON/CSV helpers (no engine needed)."""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.bundle import parquet as P


def test_arrays_to_parquet_guards():
    with pytest.raises(ValueError, match="at least one array"):
        P.arrays_to_parquet({})
    with pytest.raises(ValueError, match="at least 1-D"):
        P.arrays_to_parquet({"x": 5.0})  # 0-D


def test_empty_input_returns():
    assert P.csv_to_json("") == b""
    assert P.trace_to_json({}) == b""
    assert P.frame_to_json({}) == b""
    assert P.trace_to_csv({}) == b""
    assert P._parse_csv_columns("") == ([], [])


def test_dimension_errors():
    bad = {"x": np.zeros((2, 2, 2))}
    with pytest.raises(ValueError, match="must be 1-D or 2-D"):
        P.trace_to_json(bad)
    with pytest.raises(ValueError, match="must be 1-D or 2-D"):
        P.trace_to_csv(bad)


def test_frame_to_json_length_mismatch():
    with pytest.raises(ValueError, match="must share length"):
        P.frame_to_json({"a": [1, 2], "b": [1]})


def test_csv_scalar_branches():
    assert P._csv_scalar(None) == ""
    assert P._csv_scalar(np.int64(3)) == "3"
    # np.bool_ (not Python bool, which is an int subclass) hits the bool branch
    assert P._csv_scalar(np.bool_(True)) == "True"
    assert P._csv_scalar(np.bool_(False)) == "False"
    assert P._csv_scalar(float("inf")) == ""
    assert P._csv_scalar(2.5) == repr(2.5)
    # numpy scalar with .item that is not float/int/bool
    assert P._csv_scalar(np.complex128(1 + 2j)) == str(complex(1 + 2j))
    assert P._csv_scalar("plain") == "plain"


def test_json_scalar_branches():
    assert P._json_scalar(np.bool_(True)) is True
    assert P._json_scalar(np.int64(4)) == 4
    assert P._json_scalar(2.0) == 2.0
    assert P._json_scalar(float("nan")) is None
    assert P._json_scalar("s") == "s"
