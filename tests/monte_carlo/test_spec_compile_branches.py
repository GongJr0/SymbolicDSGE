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


def test_recover_wald_branches():
    # 1-D target, no kind -> target_vector
    out = SC._recover_wald({"target": [1.0, 2.0, 3.0]})
    assert "target_vector" in out
    # explicit kind picks the field
    out2 = SC._recover_wald({"target": [[1.0, 0.0], [0.0, 1.0]], "kind": "joint"})
    assert "target_matrix" in out2
    # matrix target without a kind is an error
    with pytest.raises(ValueError, match="must store the Wald kind"):
        SC._recover_wald({"target": [[1.0, 0.0], [0.0, 1.0]]})


def test_jsonable_branches():
    assert SC._jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert SC._jsonable(np.int64(4)) == 4
    assert SC._jsonable({"a": (1, 2)}) == {"a": [1, 2]}


def test_raw_model_data_arrays_states_and_raw():
    kwargs = {
        "states": [[1.0, 2.0], [3.0, 4.0]],
        "observables": None,  # skipped
        "raw": {"foo": [1.0, 2.0]},
    }
    out = SC.raw_model_data_arrays(kwargs)
    assert set(out) == {"states", "raw:foo"}
    assert out["states"].shape == (2, 2)


def test_recover_one_source_optional_fields():
    selector = SimpleNamespace(
        source_step="sim",
        field="observables",
        columns=["a", "b"],
        burn_in=5,
        drop_initial=2,
    )
    out = SC._recover_one_source(
        selector,
        source_key="source",
        field_key="field",
        columns_key="columns",
    )
    assert out["source"] == "sim"
    assert out["field"] == "observables"
    assert out["columns"] == ["a", "b"]
    assert out["burn_in"] == 5
    assert out["drop_initial"] == 2
