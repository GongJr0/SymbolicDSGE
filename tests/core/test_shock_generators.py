# type: ignore
from __future__ import annotations

import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.core.shock_generators import (
    Shock,
    abstract_shock_array,
    normal_multivariate_shock_array,
    normal_shock_array,
    shock_placement,
    t_multivariate_shock_array,
    t_shock_array,
    uniform_multivariate_shock_array,
    uniform_shock_array,
)


def test_abstract_shock_array_is_seed_reproducible():
    s1 = normal_shock_array(T=8, seed=7, mu=0.0, sigma=1.0)
    s2 = normal_shock_array(T=8, seed=7, mu=0.0, sigma=1.0)
    s3 = normal_shock_array(T=8, seed=8, mu=0.0, sigma=1.0)

    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_distribution_shock_array_shapes():
    n = normal_shock_array(T=10, seed=1)
    t_uni = t_shock_array(T=10, seed=1, df=5)
    u = uniform_shock_array(T=10, seed=1)
    n_mv = normal_multivariate_shock_array(
        T=10, seed=1, mus=[0.0, 0.0], cov_mat=[[1.0, 0.2], [0.2, 1.0]]
    )
    t_mv = t_multivariate_shock_array(
        T=10, seed=1, df=4, locs=[0.0, 0.0], cov_mat=[[1.0, 0.0], [0.0, 1.0]]
    )

    assert n.shape == (10,)
    assert t_uni.shape == (10,)
    assert u.shape == (10,)
    assert n_mv.shape == (10, 2)
    assert t_mv.shape == (10, 2)


def test_uniform_multivariate_shock_array_is_not_implemented():
    with pytest.raises(NotImplementedError):
        uniform_multivariate_shock_array(
            T=10,
            k=2,
            seed=1,
            locs=[0.0, 0.0],
            cov_mat=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_shock_placement_places_values_in_time_array():
    out = shock_placement(6, {0: 1.0, 3: -0.5})
    assert out.shape == (6,)
    assert out[0] == 1.0
    assert out[3] == -0.5
    assert np.allclose(out[[1, 2, 4, 5]], 0.0)


def test_shock_class_generator_and_assertions():
    sh = Shock(dist="norm", multivar=False, seed=3)
    gen = sh.shock_generator(6)
    arr = gen(0.5)
    assert arr.shape == (6,)

    sh_t = Shock(
        dist="t",
        multivar=True,
        seed=3,
        dist_kwargs={"loc": [0.0, 0.0], "df": 5.0},
    )
    t_arr = sh_t.shock_generator(6)(np.eye(2, dtype=float64))
    assert t_arr.shape == (6, 2)

    with pytest.raises(AssertionError, match="scale"):
        Shock(dist="norm", dist_kwargs={"scale": 1.0}).shock_generator(6)


def test_shock_class_place_shocks_univariate_and_bounds():
    sh = Shock(dist=None, shock_arr=np.zeros(5, dtype=float64))
    out = sh.place_shocks({1: 2.0, 4: -1.0}, 5)
    assert np.array_equal(out, np.array([0.0, 2.0, 0.0, 0.0, -1.0], dtype=float64))

    with pytest.raises(IndexError):
        sh.place_shocks({5: 1.0}, 5)


def test_shock_class_place_shocks_multivariate_and_bounds():
    sh = Shock(dist=None, multivar=True)
    out = sh.place_shocks({(0, 0): 1.0, (2, 1): -0.5}, 4)
    assert out.shape == (4, 2)
    assert out[0, 0] == 1.0
    assert out[2, 1] == -0.5

    with pytest.raises(IndexError):
        sh.place_shocks({(4, 0): 1.0}, 4)
    with pytest.raises(IndexError):
        sh.place_shocks({(0, -1): 1.0}, 4)


def test_shock_class_dist_resolution_and_custom_dist():
    sh_norm = Shock(dist="norm")
    sh_t = Shock(dist="t")
    sh_uni = Shock(dist="uni")
    assert sh_norm._get_dist() is not None
    assert sh_t._get_dist() is not None
    assert sh_uni._get_dist() is not None

    class CustomDist:
        @staticmethod
        def rvs(size, random_state=None, *args, **kwargs):
            return np.zeros(size, dtype=float64)

    sh_custom = Shock(dist=CustomDist())  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="valid scipy.stats distribution"):
        sh_custom._get_dist()
