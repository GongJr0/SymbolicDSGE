# type: ignore
import math

import numpy as np
import pytest
from numpy import float64
from scipy.special import betaln, ndtr, ndtri

from SymbolicDSGE.bayesian.distributions import (
    Beta,
    HalfNormal,
    LKJChol,
    LogNormal,
    TruncNormal,
    Uniform,
)
from SymbolicDSGE.bayesian.distributions._as241 import (
    erfinv_from_as241,
    horner,
    ndtri_as241,
)
from SymbolicDSGE.bayesian.distributions.beta import (
    _grad_logpdf_scalar as beta_grad_logpdf_scalar,
    _grad_logpdf_vectorized as beta_grad_logpdf_vectorized,
    _logpdf_scalar as beta_logpdf_scalar,
    _logpdf_vectorized as beta_logpdf_vectorized,
    _rvs as beta_rvs,
)
from SymbolicDSGE.bayesian.distributions.distribution import (
    Distribution,
    _coerce_rng,
    _scalar_or_array,
    _std_norm_cdf_scalar,
    log_beta,
    x_logy_scalar,
    x_logy_vectorized,
    xlog1py_scalar,
    xlog1py_vectorized,
)
from SymbolicDSGE.bayesian.distributions.half_norm import (
    _grad_logpdf_scalar as halfnorm_grad_logpdf_scalar,
    _grad_logpdf_vectorized as halfnorm_grad_logpdf_vectorized,
    _logpdf_scalar as halfnorm_logpdf_scalar,
    _logpdf_vectorized as halfnorm_logpdf_vectorized,
    _rvs as halfnorm_rvs,
)
from SymbolicDSGE.bayesian.distributions.lkj_chol import (
    _has_unit_row_norms,
    _is_correlation_matrix,
    _is_lower_triangular,
    _is_positive_definite,
    _is_symmetric,
    _log_lkj_normalizer_C,
    _one,
)
from SymbolicDSGE.bayesian.distributions.log_norm import (
    _grad_logpdf_scalar as lognorm_grad_logpdf_scalar,
    _grad_logpdf_vectorized as lognorm_grad_logpdf_vectorized,
    _logpdf_scalar as lognorm_logpdf_scalar,
    _logpdf_vectorized as lognorm_logpdf_vectorized,
    _rvs as lognorm_rvs,
)
from SymbolicDSGE.bayesian.distributions.param_builder import get_dist_params
from SymbolicDSGE.bayesian.distributions.trunc_norm import (
    _grad_logpdf_scalar as truncnorm_grad_logpdf_scalar,
    _grad_logpdf_vectorized as truncnorm_grad_logpdf_vectorized,
    _logpdf_scalar as truncnorm_logpdf_scalar,
    _logpdf_vectorized as truncnorm_logpdf_vectorized,
    _rvs as truncnorm_rvs,
)
from SymbolicDSGE.bayesian.distributions.uniform import (
    _grad_logpdf_scalar as uniform_grad_logpdf_scalar,
    _grad_logpdf_vectorized as uniform_grad_logpdf_vectorized,
    _logpdf_scalar as uniform_logpdf_scalar,
    _logpdf_vectorized as uniform_logpdf_vectorized,
    _rvs as uniform_rvs,
)
from SymbolicDSGE.bayesian.support import Support


class _DummyDistribution(Distribution[float64, np.ndarray]):
    def __repr__(self) -> str:
        return "Dummy"

    def logpdf(self, x):
        return float64(x)

    def grad_logpdf(self, x):
        return x

    def cdf(self, x):
        return x

    def ppf(self, q):
        return q

    def rvs(self, size=1, random_state=None):
        if isinstance(size, int):
            size = (size,)
        return np.zeros(size, dtype=np.float64)

    @property
    def support(self) -> Support:
        return Support(float64(-1.0), float64(1.0))

    @property
    def mean(self) -> float64:
        return float64(0.0)

    @property
    def var(self) -> float64:
        return float64(1.0)

    @property
    def mode(self) -> float64:
        return float64(0.0)


def _valid_corr_and_chol() -> tuple[np.ndarray, np.ndarray]:
    corr = np.array(
        [
            [1.0, 0.2, -0.1],
            [0.2, 1.0, 0.25],
            [-0.1, 0.25, 1.0],
        ],
        dtype=np.float64,
    )
    return corr, np.linalg.cholesky(corr).astype(np.float64)


def test_distribution_helper_functions_cover_scalar_vector_and_rng_branches():
    rng_none = _coerce_rng(None)
    assert rng_none.random() == pytest.approx(np.random.default_rng(0).random())

    rng_seed = _coerce_rng(123)
    assert rng_seed.random() == pytest.approx(np.random.default_rng(123).random())

    rng = np.random.default_rng(321)
    assert _coerce_rng(rng) is rng

    legacy = np.random.RandomState(123)
    legacy_rng = _coerce_rng(legacy)
    assert isinstance(legacy_rng, np.random.Generator)
    with pytest.raises(TypeError, match="Unsupported random_state type"):
        _coerce_rng("bad")

    assert _scalar_or_array(1.5) == pytest.approx(1.5)
    arr = _scalar_or_array([1.0, 2.0])
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, np.array([1.0, 2.0], dtype=np.float64))

    assert x_logy_scalar.py_func(float64(0.0), float64(2.0)) == pytest.approx(0.0)
    assert x_logy_scalar.py_func(float64(2.0), float64(4.0)) == pytest.approx(
        2.0 * np.log(4.0)
    )
    assert np.allclose(
        x_logy_vectorized.py_func(float64(2.0), np.array([1.0, 4.0], dtype=np.float64)),
        np.array([0.0, 2.0 * np.log(4.0)], dtype=np.float64),
    )
    assert np.allclose(
        x_logy_vectorized.py_func(float64(0.0), np.array([1.0, 4.0], dtype=np.float64)),
        np.zeros(2, dtype=np.float64),
    )

    assert xlog1py_scalar.py_func(float64(0.0), float64(0.5)) == pytest.approx(0.0)
    assert xlog1py_scalar.py_func(float64(3.0), float64(0.5)) == pytest.approx(
        3.0 * np.log1p(0.5)
    )
    assert np.allclose(
        xlog1py_vectorized.py_func(
            float64(2.0), np.array([0.0, 0.5], dtype=np.float64)
        ),
        np.array([0.0, 2.0 * np.log1p(0.5)], dtype=np.float64),
    )
    assert np.allclose(
        xlog1py_vectorized.py_func(
            float64(0.0), np.array([0.0, 0.5], dtype=np.float64)
        ),
        np.zeros(2, dtype=np.float64),
    )

    assert log_beta.py_func(float64(2.5), float64(3.0)) == pytest.approx(
        float(betaln(2.5, 3.0))
    )
    assert _std_norm_cdf_scalar.py_func(float64(0.7)) == pytest.approx(float(ndtr(0.7)))

    dummy = _DummyDistribution()
    assert dummy.pdf(float64(0.2)) == pytest.approx(math.exp(0.2))
    assert dummy.is_valid(float64(0.4))
    assert not dummy.is_valid(float64(2.0))
    assert isinstance(dummy._rng(1), np.random.Generator)


def test_as241_helpers_cover_all_branches():
    coeffs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert horner.py_func(coeffs, float64(2.0)) == pytest.approx(17.0)

    assert np.isneginf(ndtri_as241.py_func(float64(0.0)))
    assert np.isposinf(ndtri_as241.py_func(float64(1.0)))
    assert ndtri_as241.py_func(float64(0.5)) == pytest.approx(0.0, abs=1e-12)
    assert ndtri_as241.py_func(float64(0.9)) == pytest.approx(
        float(ndtri(0.9)), rel=1e-6
    )
    assert ndtri_as241.py_func(float64(1e-8)) == pytest.approx(
        float(ndtri(1e-8)), rel=1e-5
    )

    y = float64(0.25)
    expected = ndtri_as241.py_func(float64(0.5 * (y + 1.0)) / np.sqrt(2.0))
    assert erfinv_from_as241.py_func(y) == pytest.approx(expected)


def test_uniform_distribution_helper_and_property_branches():
    dist = Uniform(-2.0, 3.5, 123)

    assert uniform_logpdf_scalar.py_func(
        float64(-2.0), float64(3.5), float64(5.5), float64(0.0)
    ) == pytest.approx(-np.log(5.5))
    assert np.isneginf(
        uniform_logpdf_scalar.py_func(
            float64(-2.0), float64(3.5), float64(5.5), float64(4.0)
        )
    )
    assert np.allclose(
        uniform_logpdf_vectorized.py_func(
            float64(-2.0),
            float64(3.5),
            float64(5.5),
            np.array([-3.0, 0.0], dtype=np.float64),
        ),
        np.array([-np.inf, -np.log(5.5)], dtype=np.float64),
        equal_nan=False,
    )

    assert uniform_grad_logpdf_scalar.py_func(
        float64(-2.0), float64(3.5), float64(0.0)
    ) == pytest.approx(0.0)
    assert np.isneginf(
        uniform_grad_logpdf_scalar.py_func(float64(-2.0), float64(3.5), float64(4.0))
    )
    assert np.allclose(
        uniform_grad_logpdf_vectorized.py_func(
            float64(-2.0), float64(3.5), np.array([-3.0, 0.0], dtype=np.float64)
        ),
        np.array([-np.inf, 0.0], dtype=np.float64),
    )

    assert dist.cdf(float64(-3.0)) == pytest.approx(0.0)
    assert dist.cdf(float64(10.0)) == pytest.approx(1.0)
    assert dist.cdf(float64(0.75)) == pytest.approx((0.75 + 2.0) / 5.5)
    assert np.allclose(
        dist.cdf(np.array([-3.0, 0.75, 10.0], dtype=np.float64)),
        np.array([0.0, (0.75 + 2.0) / 5.5, 1.0], dtype=np.float64),
    )

    assert np.isnan(dist.ppf(float64(-0.1)))
    assert np.isnan(dist.ppf(float64(1.1)))
    assert dist.ppf(float64(0.25)) == pytest.approx(-2.0 + 0.25 * 5.5)
    assert np.allclose(
        dist.ppf(np.array([-0.1, 0.5, 1.1], dtype=np.float64)),
        np.array([np.nan, 0.75, np.nan], dtype=np.float64),
        equal_nan=True,
    )

    draws = uniform_rvs.py_func(
        float64(-2.0),
        float64(3.5),
        (4,),
        np.random.default_rng(123),
    )
    assert draws.shape == (4,)
    assert dist.rvs(3).shape == (3,)
    assert dist.mean == pytest.approx(0.75)
    assert dist.var == pytest.approx((5.5**2) / 12.0)
    assert dist.support == Support(float64(-2.0), float64(3.5))
    with pytest.raises(ValueError, match="unique mode"):
        _ = dist.mode


def test_beta_distribution_helper_and_mode_branches():
    dist = Beta(2.5, 4.0, 123)

    assert beta_logpdf_scalar.py_func(
        float64(2.5), float64(4.0), float64(log_beta.py_func(2.5, 4.0)), float64(0.4)
    ) == pytest.approx(float(dist.logpdf(float64(0.4))))
    assert np.allclose(
        beta_logpdf_vectorized.py_func(
            float64(2.5),
            float64(4.0),
            float64(log_beta.py_func(2.5, 4.0)),
            np.array([0.2, 0.4], dtype=np.float64),
        ),
        np.asarray(dist.logpdf(np.array([0.2, 0.4], dtype=np.float64))),
    )
    assert beta_grad_logpdf_scalar.py_func(
        float64(2.5), float64(4.0), float64(0.4)
    ) == pytest.approx(float(dist.grad_logpdf(float64(0.4))))
    assert np.allclose(
        beta_grad_logpdf_vectorized.py_func(
            float64(2.5), float64(4.0), np.array([0.2, 0.4], dtype=np.float64)
        ),
        np.asarray(dist.grad_logpdf(np.array([0.2, 0.4], dtype=np.float64))),
    )
    assert beta_rvs.py_func(
        float64(2.5), float64(4.0), (3,), np.random.default_rng(123)
    ).shape == (3,)

    assert dist.cdf(float64(-0.1)) == pytest.approx(0.0)
    assert dist.cdf(float64(1.1)) == pytest.approx(1.0)
    assert np.allclose(
        dist.cdf(np.array([-0.1, 0.3, 1.1], dtype=np.float64)),
        np.array([0.0, dist.cdf(float64(0.3)), 1.0], dtype=np.float64),
    )
    assert dist.mean == pytest.approx(2.5 / 6.5)
    assert dist.var == pytest.approx((2.5 * 4.0) / ((6.5**2) * 7.5))
    assert dist.mode == pytest.approx((2.5 - 1.0) / (2.5 + 4.0 - 2.0))
    assert Beta(0.8, 2.5, 123).mode == pytest.approx(0.0)
    assert Beta(2.5, 0.8, 123).mode == pytest.approx(1.0)
    with pytest.raises(ValueError, match="does not have a unique mode"):
        _ = Beta(0.8, 0.9, 123).mode


def test_halfnormal_and_lognormal_property_and_vector_branches():
    half = HalfNormal(1.4, 123)
    assert np.isneginf(halfnorm_logpdf_scalar.py_func(float64(-0.1), float64(1.4)))
    assert halfnorm_logpdf_scalar.py_func(float64(0.4), float64(1.4)) == pytest.approx(
        float(half.logpdf(float64(0.4)))
    )
    assert np.allclose(
        halfnorm_logpdf_vectorized.py_func(
            np.array([-0.1, 0.4], dtype=np.float64), float64(1.4)
        ),
        np.array([-np.inf, float(half.logpdf(float64(0.4)))], dtype=np.float64),
    )
    assert halfnorm_grad_logpdf_scalar.py_func(
        float64(-0.1), float64(1.4)
    ) == pytest.approx(0.0)
    assert np.allclose(
        halfnorm_grad_logpdf_vectorized.py_func(
            np.array([-0.1, 0.4], dtype=np.float64), float64(1.4)
        ),
        np.array([0.0, -0.4 / (1.4**2)], dtype=np.float64),
    )
    assert half.cdf(float64(-0.1)) == pytest.approx(0.0)
    assert half.cdf(np.array([0.4], dtype=np.float64)) == pytest.approx(
        float(half.cdf(float64(0.4)))
    )
    assert half.ppf(np.array([0.5], dtype=np.float64)) == pytest.approx(
        float(half.ppf(float64(0.5)))
    )
    assert halfnorm_rvs.py_func(float64(1.4), (3,), np.random.default_rng(1)).shape == (
        3,
    )
    assert isinstance(half.rng, np.random.Generator)
    assert half.mean > 0.0
    assert half.var > 0.0
    assert half.mode == pytest.approx(0.0)
    assert half.std == pytest.approx(1.4)

    logn = LogNormal(np.log(1.8), 0.45, 123)
    assert lognorm_logpdf_scalar.py_func(
        float64(np.log(1.8)), float64(0.45), float64(1.4)
    ) == pytest.approx(float(logn.logpdf(float64(1.4))))
    assert np.allclose(
        lognorm_logpdf_vectorized.py_func(
            float64(np.log(1.8)),
            float64(0.45),
            np.array([1.4, 2.0], dtype=np.float64),
        ),
        np.asarray(logn.logpdf(np.array([1.4, 2.0], dtype=np.float64))),
    )
    assert lognorm_grad_logpdf_scalar.py_func(
        float64(np.log(1.8)), float64(0.45), float64(1.4)
    ) == pytest.approx(float(logn.grad_logpdf(float64(1.4))))
    assert np.allclose(
        lognorm_grad_logpdf_vectorized.py_func(
            float64(np.log(1.8)),
            float64(0.45),
            np.array([1.4, 2.0], dtype=np.float64),
        ),
        np.asarray(logn.grad_logpdf(np.array([1.4, 2.0], dtype=np.float64))),
    )
    assert logn.cdf(np.array([1.4], dtype=np.float64)) == pytest.approx(
        float(logn.cdf(float64(1.4)))
    )
    assert logn.ppf(np.array([0.5], dtype=np.float64)) == pytest.approx(
        float(logn.ppf(float64(0.5)))
    )
    assert lognorm_rvs.py_func(
        float64(np.log(1.8)), float64(0.45), (3,), np.random.default_rng(1)
    ).shape == (3,)
    assert logn.rvs(2).shape == (2,)
    assert logn.mean > 0.0
    assert logn.var > 0.0
    assert logn.mode > 0.0


def test_truncnormal_helper_and_property_branches():
    dist = TruncNormal(-1.0, 1.0, 0.2, 0.7, 123)
    assert truncnorm_logpdf_scalar.py_func(
        float64(0.1), float64(0.2), float64(0.7), float64(dist._log_norm)
    ) == pytest.approx(float(dist.logpdf(float64(0.1))))
    assert np.allclose(
        truncnorm_logpdf_vectorized.py_func(
            np.array([-0.2, 0.1], dtype=np.float64),
            float64(0.2),
            float64(0.7),
            float64(dist._log_norm),
        ),
        np.asarray(dist.logpdf(np.array([-0.2, 0.1], dtype=np.float64))),
    )
    assert truncnorm_grad_logpdf_scalar.py_func(
        float64(0.1), float64(0.2), float64(0.7)
    ) == pytest.approx(float(dist.grad_logpdf(float64(0.1))))
    assert np.allclose(
        truncnorm_grad_logpdf_vectorized.py_func(
            np.array([-0.2, 0.1], dtype=np.float64),
            float64(0.2),
            float64(0.7),
        ),
        np.asarray(dist.grad_logpdf(np.array([-0.2, 0.1], dtype=np.float64))),
    )
    draws = truncnorm_rvs.py_func(
        float64(0.2),
        float64(0.7),
        float64((-1.0 - 0.2) / 0.7),
        float64((1.0 - 0.2) / 0.7),
        (4,),
        np.random.default_rng(1),
    )
    assert draws.shape == (4,)
    assert np.all((-1.0 <= draws) & (draws <= 1.0))

    assert TruncNormal._scalar_to_std(0.2, 0.7, -1.0, 1.0) == pytest.approx(
        ((-1.0 - 0.2) / 0.7, (1.0 - 0.2) / 0.7)
    )
    assert dist.cdf(np.array([0.1], dtype=np.float64)) == pytest.approx(
        float(dist.cdf(float64(0.1)))
    )
    assert dist.ppf(np.array([0.5], dtype=np.float64)) == pytest.approx(
        float(dist.ppf(float64(0.5)))
    )
    assert dist.mean == pytest.approx(float(dist.mean))
    assert dist.var == pytest.approx(float(dist.var))
    assert dist.mode == pytest.approx(0.2)
    assert TruncNormal(-1.0, 1.0, -5.0, 0.7, 123).mode == pytest.approx(-1.0)
    assert TruncNormal(-1.0, 1.0, 5.0, 0.7, 123).mode == pytest.approx(1.0)


def test_lkj_helper_validation_and_not_implemented_branches():
    corr, L = _valid_corr_and_chol()

    assert _is_symmetric(corr)
    assert not _is_symmetric(np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64))
    assert _is_positive_definite(corr)
    assert not _is_positive_definite(
        np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    )
    assert _is_correlation_matrix(corr)
    assert not _is_correlation_matrix(
        np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    )
    assert _is_lower_triangular(L)
    assert not _is_lower_triangular(corr)
    assert _has_unit_row_norms(L)
    bad_L = L.copy()
    bad_L[2, 2] = 0.5
    assert not _has_unit_row_norms(bad_L)

    with pytest.raises(ValueError, match="K must be >= 1"):
        _log_lkj_normalizer_C(0, 1.0)
    with pytest.raises(ValueError, match="eta must be > 0"):
        _log_lkj_normalizer_C(2, 0.0)
    assert _log_lkj_normalizer_C(1, 2.0) == pytest.approx(0.0)

    sample = _one.py_func(float64(2.0), 3, np.random.default_rng(123))
    assert sample.shape == (3, 3)
    assert _is_lower_triangular(sample)
    assert _has_unit_row_norms(sample)

    dist = LKJChol(eta=2.0, K=3, random_state=None)
    assert dist.logpdf_from_R(corr) == pytest.approx(dist.logpdf(L))
    with pytest.raises(ValueError, match="square matrix"):
        dist.logpdf_from_R(np.array([1.0, 2.0], dtype=np.float64))
    with pytest.raises(ValueError, match="symmetric"):
        dist.logpdf_from_R(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="positive definite"):
        dist.logpdf_from_R(np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="correlation matrix"):
        dist.logpdf_from_R(np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64))

    grad = dist.grad_logpdf(L)
    assert grad.shape == L.shape
    assert grad[0, 0] == pytest.approx(0.0)
    assert dist.rvs(2).shape == (2, 3, 3)
    assert dist.support.contains(np.array([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float64))

    with pytest.raises(NotImplementedError):
        dist.cdf(L)
    with pytest.raises(NotImplementedError):
        dist.ppf(L)
    with pytest.raises(NotImplementedError):
        _ = dist.mean
    with pytest.raises(NotImplementedError):
        _ = dist.var
    with pytest.raises(NotImplementedError):
        _ = dist.mode


def test_distribution_param_builder_error_branch():
    defaults = get_dist_params("normal")
    assert defaults["mean"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Unsupported distribution family"):
        get_dist_params("not_a_distribution")
