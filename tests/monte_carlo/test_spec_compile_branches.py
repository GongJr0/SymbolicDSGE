"""Branch coverage for monte_carlo.spec_compile helper functions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from SymbolicDSGE.monte_carlo import spec_compile as SC


def test_step_type_requires_step_type():
    with pytest.raises(ValueError, match="no step_type"):
        SC._step_type(SimpleNamespace(step_type=None, name="s"))
    assert SC._step_type(SimpleNamespace(step_type="wald", name="s")) == "wald"


def test_shock_dict_branches():
    assert SC._shock_dict({"dist": "norm"}) == {"dist": "norm"}
    with pytest.raises(TypeError, match="raw shock array"):
        SC._shock_dict(np.zeros(3))
    with pytest.raises(TypeError, match="shock generator"):
        SC._shock_dict(lambda s: s)


def test_jsonable_branches():
    assert SC._jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert SC._jsonable(np.int64(4)) == 4
    assert SC._jsonable({"a": (1, 2)}) == {"a": [1, 2]}


def test_raw_model_data_arrays_states_only():
    kwargs = {
        "states": [[1.0, 2.0], [3.0, 4.0]],
        "observables": None,  # skipped
        "raw": {"foo": [1.0, 2.0]},  # legacy payloads are not serialized.
    }
    out = SC.raw_model_data_arrays(kwargs)
    assert set(out) == {"states"}
    assert out["states"].shape == (2, 2)
