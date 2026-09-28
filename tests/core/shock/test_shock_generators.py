# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numpy import float64

from scipy.stats import norm, multivariate_normal as mnorm

from SymbolicDSGE.core.shock.generators import Shock, abstract_shock_array


def test_abstract_shock_array_is_seed_reproducible():
    s1 = abstract_shock_array(8, 7, norm, loc=0.0, scale=1.0)
    s2 = abstract_shock_array(8, 7, norm, loc=0.0, scale=1.0)
    s3 = abstract_shock_array(8, 8, norm, loc=0.0, scale=1.0)

    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_shock_class_draw_fn_and_rejections():
    # The draw takes ``(loc, factor, seed)`` and returns ``(T, width)``: the
    # factor carries the scale at either arity, a 1x1 holding a standard
    # deviation or a covariance block's factor.
    sh = Shock(dist=norm, seed=3)
    arr = sh.draw_fn(6, False)(np.zeros(1), np.array([[0.5]]), sh.seed)
    assert arr.shape == (6, 1)

    # Arity is the caller's to state: the same spec draws either way.
    sh_mn = Shock(dist=mnorm, seed=3)
    mn_arr = sh_mn.draw_fn(6, True)(np.zeros(2), np.eye(2, dtype=float64), sh_mn.seed)
    assert mn_arr.shape == (6, 2)

    with pytest.raises(ValueError, match="scale"):
        Shock(dist="norm", dist_kwargs={"scale": 1.0}).draw_fn(6, False)

    with pytest.raises(ValueError, match="A distribution must be specified"):
        Shock(dist=None).draw_fn(6, False)
