# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numpy import float64

from scipy.stats import norm

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
    sh = Shock(dist="norm", seed=3)
    arr = sh.draw_fn(6, False)(np.zeros(1), np.array([[0.5]]), sh.seed)
    assert arr.shape == (6, 1)

    # Arity is the caller's to state: the same spec draws either way.
    sh_t = Shock(dist="t", seed=3, dist_kwargs={"df": 5.0})
    t_arr = sh_t.draw_fn(6, True)(np.zeros(2), np.eye(2, dtype=float64), sh_t.seed)
    assert t_arr.shape == (6, 2)

    with pytest.raises(ValueError, match="scale"):
        Shock(dist="norm", dist_kwargs={"scale": 1.0}).draw_fn(6, False)

    with pytest.raises(ValueError, match="Distribution must be specified"):
        Shock(dist=None).draw_fn(6, False)

    # A linear map of independent uniforms is not uniform in its margins, so a
    # grouped uniform is refused where the arity is known.
    with pytest.raises(NotImplementedError, match="Multivariate uniform"):
        Shock(dist="uni").draw_fn(6, True)


def test_get_dist_takes_only_live_objects():
    # A named family draws through the numpy fast paths, so only a live
    # distribution object ever reaches the scipy route's resolution.
    for name in ("norm", "t", "uni"):
        with pytest.raises(TypeError, match="scipy.stats distribution object"):
            Shock(dist=name)._get_dist(False)

    class CustomDist:
        @staticmethod
        def rvs(size, random_state=None, *args, **kwargs):
            return np.zeros(size, dtype=float64)

    sh_custom = Shock(dist=CustomDist())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scipy.stats distribution object"):
        sh_custom._get_dist(False)
