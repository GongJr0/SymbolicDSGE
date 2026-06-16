from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import norm

from SymbolicDSGE.core.shock_generators import Shock


def test_norm_univariate_round_trips() -> None:
    shock = Shock(T=8, dist="norm", seed=3, dist_kwargs={"loc": 0.5})
    payload = shock.to_dict()
    json.dumps(payload)  # must be JSON-safe

    restored = Shock.from_dict(payload)
    assert restored.to_dict() == payload
    assert (restored.T, restored.dist, restored.seed) == (8, "norm", 3)
    assert restored.multivar is False
    assert restored.dist_kwargs == {"loc": 0.5}


def test_t_distribution_carries_df() -> None:
    shock = Shock(T=12, dist="t", seed=None, dist_kwargs={"loc": 0.0, "df": 5.0})
    restored = Shock.from_dict(shock.to_dict())
    assert restored.seed is None
    assert restored.dist_kwargs == {"loc": 0.0, "df": 5.0}


def test_multivariate_numpy_kwargs_are_coerced_to_lists() -> None:
    shock = Shock(
        T=6,
        dist="norm",
        multivar=True,
        seed=0,
        dist_kwargs={"mean": np.array([0.0, 1.0])},
    )
    payload = shock.to_dict()
    assert payload["multivar"] is True
    assert payload["dist_kwargs"]["mean"] == [0.0, 1.0]
    json.dumps(payload)


def test_to_dict_rejects_live_scipy_distribution() -> None:
    shock = Shock(T=4, dist=norm, seed=0)
    with pytest.raises(TypeError, match="string-identified"):
        shock.to_dict()


def test_to_dict_rejects_materialized_shock_arr() -> None:
    shock = Shock(T=4, dist="norm", seed=0, shock_arr=np.zeros(4))
    with pytest.raises(ValueError, match="shock_arr"):
        shock.to_dict()


def test_from_dict_rejects_unknown_dist() -> None:
    with pytest.raises(ValueError, match="'norm'/'t'/'uni'"):
        Shock.from_dict({"T": 4, "dist": "cauchy"})
