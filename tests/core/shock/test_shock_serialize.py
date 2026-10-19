from __future__ import annotations

import json

import pytest

from SymbolicDSGE.core.shock.generators import Shock
from SymbolicDSGE.core.shock.spec import serialize_shocks, deserialize_shocks


def test_norm_univariate_round_trips() -> None:
    shock = Shock(dist="norm", seed=3, loc=0.5).joint("u")
    payload = serialize_shocks([shock])[0]
    json.dumps(payload)  # must be JSON-safe

    restored = deserialize_shocks(specs=payload, paths={}, tables={})[0]
    assert serialize_shocks([restored])[0] == payload
    assert (restored.dist, restored.seed, restored.target) == ("norm", 3, ("u",))
    assert restored.dist_kwargs == {"loc": 0.5}


def test_t_distribution_carries_df() -> None:
    shock = Shock(dist="t", seed=None, loc=0.0, df=5.0).joint("u")
    restored = deserialize_shocks(*serialize_shocks([shock]))[0]
    assert restored.seed is None
    assert restored.dist_kwargs == {"loc": 0.0, "df": 5.0}


def test_deserialization_ignores_a_stored_multivar_and_T() -> None:
    # Arity belongs to the entry's own target, which is why a flag written by
    # an older release is read past rather than honored.
    payload = {
        "target": ["u"],
        "dist": "norm",
        "seed": 1,
        "multivar": True,
        "T": 10,
        "dist_kwargs": {"loc": 0.0},
    }
    restored = deserialize_shocks(specs=[payload], paths={}, tables={})[0]
    assert not hasattr(restored, "multivar")
    assert not hasattr(restored, "T")
    assert not any(k in serialize_shocks([restored])[0] for k in ("multivar", "T"))


def test_deserialize_rejects_unknown_dist() -> None:
    with pytest.raises(ValueError, match=r"Unknown shock distribution family:"):
        deserialize_shocks(
            specs=[{"target": ["u"], "dist": "laplace", "seed": 1, "dist_kwargs": {}}],
            paths={},
            tables={},
        )
