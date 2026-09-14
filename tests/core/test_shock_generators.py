# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numpy import float64

from scipy.stats import norm

from SymbolicDSGE.core.shock_generators import Shock, abstract_shock_array


def test_abstract_shock_array_is_seed_reproducible():
    s1 = abstract_shock_array(8, 7, norm, loc=0.0, scale=1.0)
    s2 = abstract_shock_array(8, 7, norm, loc=0.0, scale=1.0)
    s3 = abstract_shock_array(8, 8, norm, loc=0.0, scale=1.0)

    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_shock_class_draw_fn_and_rejections():
    sh = Shock(dist="norm", seed=3)
    arr = sh.draw_fn(6, False)(0.5, sh.seed, None)
    assert arr.shape == (6,)

    # Arity is the caller's to state: the same spec draws either way.
    sh_t = Shock(
        dist="t",
        seed=3,
        dist_kwargs={"loc": [0.0, 0.0], "df": 5.0},
    )
    t_arr = sh_t.draw_fn(6, True)(np.eye(2, dtype=float64), sh_t.seed, None)
    assert t_arr.shape == (6, 2)

    with pytest.raises(ValueError, match="scale"):
        Shock(dist="norm", dist_kwargs={"scale": 1.0}).draw_fn(6, False)

    with pytest.raises(ValueError, match="Distribution must be specified"):
        Shock().draw_fn(6, False)


def test_shock_class_dist_resolution_and_custom_dist():
    sh_norm = Shock(dist="norm")
    sh_t = Shock(dist="t")
    sh_uni = Shock(dist="uni")
    assert sh_norm._get_dist(False) is not None
    assert sh_t._get_dist(False) is not None
    assert sh_uni._get_dist(False) is not None

    class CustomDist:
        @staticmethod
        def rvs(size, random_state=None, *args, **kwargs):
            return np.zeros(size, dtype=float64)

    sh_custom = Shock(dist=CustomDist())  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="valid scipy.stats distribution"):
        sh_custom._get_dist(False)
