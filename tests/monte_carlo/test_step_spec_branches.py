"""Branch coverage for the step serialization helpers in monte_carlo.mc_constructs."""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.core.shock_generators import Shock
from SymbolicDSGE.monte_carlo import mc_constructs as MC


def test_jsonable_branches():
    assert MC._jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert MC._jsonable(np.int64(4)) == 4
    assert MC._jsonable({"a": (1, 2)}) == {"a": [1, 2]}
    # Anything declaring `to_dict` is asked for it rather than walked.
    assert MC._jsonable(Shock(dist="norm", seed=0))["dist"] == "norm"


def test_shock_spec_branches():
    assert MC._shock_spec("u", Shock(dist="norm", seed=0))["seed"] == 0
    # A bare shock path travels as nested lists.
    assert MC._shock_spec("u", np.zeros((2, 1))) == [[0.0], [0.0]]
    with pytest.raises(TypeError, match="callable"):
        MC._shock_spec("u", lambda s: s)


def test_restore_shock_branches():
    live = Shock(dist="norm", seed=3)
    assert MC._restore_shock(live) is live
    assert MC._restore_shock(live.to_dict()).seed == 3
    restored = MC._restore_shock([[0.0], [1.0]])
    assert isinstance(restored, np.ndarray)
    assert restored.shape == (2, 1)
