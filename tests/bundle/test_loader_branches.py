"""Branch coverage for bundle.loader error paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from SymbolicDSGE.bundle import loader as L


def test_load_columns_rejects_unknown_format():
    archive = SimpleNamespace(read=lambda path: b"")
    member = SimpleNamespace(path="data.json", format="json")
    with pytest.raises(ValueError, match="neither 'parquet' nor 'csv'"):
        L._load_columns(archive, member)


def test_stack_observed_without_y_or_columns():
    member = SimpleNamespace(path="obs.parquet", columns=None)
    with pytest.raises(ValueError, match="no 'y.\\*' columns"):
        L._stack_observed({"a": [1.0, 2.0]}, member)
