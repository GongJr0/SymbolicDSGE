"""Branch coverage for the step serialization helpers in monte_carlo.mc_constructs."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from SymbolicDSGE.core.shock.generators import Shock, ShockPath
from SymbolicDSGE.core.shock.spec import shock_from_json
from SymbolicDSGE.monte_carlo import mc_constructs as MC


def test_jsonable_branches():
    assert MC._jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert MC._jsonable(np.int64(4)) == 4
    assert MC._jsonable({"a": (1, 2)}) == {"a": [1, 2]}
    # Anything declaring `to_dict` is asked for it rather than walked.
    assert MC._jsonable(Shock(dist="norm", seed=0).joint("u"))["dist"] == "norm"


def test_shock_entries_serialize_themselves():
    # An entry names its own shocks, so the spec needs no object key: a Shock
    # flattens its arguments in, a path rides under "path".
    assert Shock(dist="norm", seed=0).joint("u").to_dict() == {
        "target": ("u",),
        "dist": "norm",
        "seed": 0,
        "dist_kwargs": {},
    }
    assert Shock(dist="norm", seed=0).joint("u", "v").to_dict()["target"] == ("u", "v")
    assert ShockPath(np.zeros((2, 1)), "u").to_dict() == {
        "target": ("u",),
        "path": [[0.0], [0.0]],
    }
    # Neither family nor path: a live scipy object has no serialized form.
    with pytest.raises(TypeError, match="serializable"):
        Shock(dist=scipy.stats.norm, seed=0).joint("u").to_dict()


def test_shock_from_json_branches():
    live = Shock(dist="norm", seed=3).joint("u")
    assert shock_from_json(live.to_dict()).seed == 3
    restored = shock_from_json({"target": ["u"], "path": [[0.0], [1.0]]})
    assert isinstance(restored, ShockPath)
    assert restored.path.shape == (2, 1)
    assert restored.target == ("u",)
