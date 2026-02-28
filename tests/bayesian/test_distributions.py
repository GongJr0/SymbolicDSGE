# type: ignore
import numpy as np
import pytest
from numpy import float64

import SymbolicDSGE.bayesian.distributions.lkj_chol as lkj_chol_module
from SymbolicDSGE.bayesian.distributions import (
    Beta,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InvGamma,
    LKJChol,
    LogNormal,
    Normal,
    TruncNormal,
    Uniform,
)
from SymbolicDSGE.bayesian.distributions.distribution import _coerce_rng
from SymbolicDSGE.bayesian.distributions.lkj_chol import _log_lkj_normalizer_C
from SymbolicDSGE.bayesian.support import OutOfSupportError


@pytest.fixture(
    params=[
        (
            "normal",
            lambda: Normal(0.0, 1.0, 123),
            float64(0.2),
            np.inf,
            False,
        ),
        (
            "trunc_normal",
            lambda: TruncNormal(-1.0, 1.0, 0.0, 1.0, 123),
            float64(0.2),
            float64(2.0),
            False,
        ),
        (
            "beta",
            lambda: Beta(2.0, 3.0, 0.0, 1.0, 123),
            float64(0.4),
            float64(-0.1),
            False,
        ),
        (
            "gamma",
            lambda: Gamma(3.0, 0.0, 2.0, 123),
            float64(1.4),
            float64(-0.1),
            False,
        ),
        (
            "inv_gamma",
            lambda: InvGamma(4.0, 0.0, 2.0, 123),
            float64(1.4),
            float64(0.0),
            False,
        ),
        (
            "half_normal",
            lambda: HalfNormal(
                {
                    "loc": 0.0,
                    "scale": 1.0,
                    "transform": "identity",
                    "transform_kwargs": {},
                    "random_state": 123,
                }
            ),
            float64(0.8),
            float64(-0.1),
            False,
        ),
        (
            "half_cauchy",
            lambda: HalfCauchy(0.0, 1.0, 123),
            float64(0.8),
            float64(-0.1),
            True,
        ),
    ],
    ids=lambda case: case[0],
)
def distribution_case(request):
    return request.param


# Central finite difference:
# f'(x) ~= (f(x+h) - f(x-h)) / (2h)
def _central_diff_grad_logpdf(dist, x: float64, h: float64 = float64(1e-6)) -> float64:
    return float64((dist.logpdf(x + h) - dist.logpdf(x - h)) / (2.0 * h))


def test_expected_concrete_distribution_classes():
    distribution_classes = [
        Normal,
        LogNormal,
        HalfNormal,
        TruncNormal,
        HalfCauchy,
        Beta,
        Gamma,
        InvGamma,
        Uniform,
        LKJChol,
    ]
    for cls in distribution_classes:
        assert not bool(cls.__abstractmethods__), f"{cls.__name__} is still abstract"


def test_lkj_chol_is_concrete():
    assert not bool(LKJChol.__abstractmethods__)


def test_distribution_pdf_is_exp_logpdf(distribution_case):
    _, ctor, x_valid, _, _ = distribution_case
    dist = ctor()
    assert np.allclose(dist.pdf(x_valid), np.exp(dist.logpdf(x_valid)))


def test_distribution_cdf_ppf_roundtrip(distribution_case):
    _, ctor, x_valid, _, _ = distribution_case
    dist = ctor()

    q = dist.cdf(x_valid)
    x_back = dist.ppf(q)

    assert 0.0 <= q <= 1.0
    assert np.allclose(x_back, x_valid, atol=1e-5, rtol=1e-5)


def test_distribution_grad_logpdf_matches_finite_difference(distribution_case):
    _, ctor, x_valid, _, _ = distribution_case
    dist = ctor()

    analytic = dist.grad_logpdf(x_valid)
    numeric = _central_diff_grad_logpdf(dist, x_valid)
    assert np.allclose(analytic, numeric, atol=1e-4, rtol=1e-4)


def test_distribution_rvs_output_and_support(distribution_case):
    _, ctor, _, _, _ = distribution_case
    dist = ctor()

    samples = dist.rvs(size=7, random_state=42)
    assert np.shape(samples) == (7,)
    assert dist.support.contains(samples)


def test_distribution_rvs_seed_reproducibility(distribution_case):
    _, ctor, _, _, _ = distribution_case
    dist = ctor()

    # Explicit seed override should make repeated draws identical.
    s1 = np.asarray(dist.rvs(size=8, random_state=2026))
    s2 = np.asarray(dist.rvs(size=8, random_state=2026))
    s3 = np.asarray(dist.rvs(size=8, random_state=2027))

    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)

    # Stored constructor seed should also be reproducible across fresh instances.
    d1 = ctor()
    d2 = ctor()
    c1 = np.asarray(d1.rvs(size=8))
    c2 = np.asarray(d2.rvs(size=8))
    assert np.array_equal(c1, c2)


def test_distribution_rvs_size_parameterization(distribution_case):
    _, ctor, _, _, _ = distribution_case
    dist = ctor()

    sample_1d = np.asarray(dist.rvs(size=5, random_state=99))
    sample_2d = np.asarray(dist.rvs(size=(2, 3), random_state=99))

    assert isinstance(sample_1d, np.ndarray)
    assert isinstance(sample_2d, np.ndarray)
    assert sample_1d.shape == (5,)
    assert sample_2d.shape == (2, 3)


def test_distribution_support_and_validity(distribution_case):
    _, ctor, x_valid, x_invalid, _ = distribution_case
    dist = ctor()

    assert dist.support.contains(x_valid)
    assert dist.is_valid(x_valid)
    assert not dist.is_valid(x_invalid)


def test_distribution_moment_properties(distribution_case):
    _, ctor, _, _, has_nan_moments = distribution_case
    dist = ctor()

    if has_nan_moments:
        assert np.isnan(dist.mean)
        assert np.isnan(dist.var)
    else:
        assert np.isfinite(dist.mean)
        assert np.isfinite(dist.var)

    assert dist.support.contains(dist.mode)


def test_distribution_logpdf_raises_outside_support(distribution_case):
    _, ctor, _, x_invalid, _ = distribution_case
    dist = ctor()

    with pytest.raises(OutOfSupportError):
        dist.logpdf(x_invalid)


def test_coerce_rng_accepts_supported_inputs():
    generator = np.random.default_rng(1)
    random_state = np.random.RandomState(1)

    for seed in (None, 123, generator, random_state):
        out = _coerce_rng(seed)
        assert isinstance(out, np.random.Generator)


def test_coerce_rng_rejects_unsupported_inputs():
    with pytest.raises(TypeError):
        _coerce_rng("bad-seed")


def _valid_lkj_chol_3x3() -> np.ndarray:
    a = float64(0.3)
    b = float64(-0.2)
    c = float64(0.5)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [a, np.sqrt(1.0 - a**2), 0.0],
            [b, c, np.sqrt(1.0 - b**2 - c**2)],
        ],
        dtype=np.float64,
    )


def test_lkj_chol_logpdf_matches_manual_expression():
    d = LKJChol(2.0, 3, 123)
    L = _valid_lkj_chol_3x3()

    diag = np.diag(L)
    expected = float64(
        _log_lkj_normalizer_C(3, 2.0) + 3.0 * np.log(diag[1]) + 2.0 * np.log(diag[2])
    )
    assert np.allclose(d.logpdf(L), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "L",
    [
        np.ones((2, 3), dtype=np.float64),
        np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    ],
    ids=[
        "non-square",
        "not-lower-triangular",
        "non-positive-diagonal",
        "bad-row-norms",
    ],
)
def test_lkj_chol_logpdf_rejects_invalid_cholesky_inputs(L):
    d = LKJChol(1.5, 3, 123)
    with pytest.raises(ValueError):
        d.logpdf(L)


def test_lkj_chol_logpdf_from_R_matches_logpdf_of_cholesky():
    d = LKJChol(2.0, 3, 123)
    L = _valid_lkj_chol_3x3()
    R = L @ L.T
    assert np.allclose(d.logpdf_from_R(R), d.logpdf(L), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "R",
    [
        np.ones((2, 3), dtype=np.float64),
        np.array([[1.0, 0.2], [0.1, 1.0]], dtype=np.float64),
        np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64),
        np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64),
    ],
    ids=["non-square", "non-symmetric", "not-positive-definite", "not-correlation"],
)
def test_lkj_chol_logpdf_from_R_rejects_invalid_inputs(R):
    d = LKJChol(1.2, 2, 123)
    with pytest.raises(ValueError):
        d.logpdf_from_R(R)


def test_lkj_chol_grad_logpdf_matches_implemented_formula():
    eta = float64(2.0)
    K = 3
    d = LKJChol(eta, K, 123)
    L = _valid_lkj_chol_3x3()
    G = d.grad_logpdf(L)

    expected = np.zeros_like(L, dtype=np.float64)
    for k in range(1, K):
        exponent = float64(K - k - 2.0 * (eta - 1.0))
        expected[k, k] = exponent / L[k, k]

    assert np.allclose(G, expected, rtol=1e-12, atol=1e-12)


def test_lkj_chol_undefined_api_raises_not_implemented():
    d = LKJChol(1.5, 3, 123)
    X = _valid_lkj_chol_3x3()

    with pytest.raises(NotImplementedError):
        d.cdf(X)
    with pytest.raises(NotImplementedError):
        d.ppf(X)
    with pytest.raises(NotImplementedError):
        _ = d.mean
    with pytest.raises(NotImplementedError):
        _ = d.var
    with pytest.raises(NotImplementedError):
        _ = d.mode


def test_lkj_chol_rvs_shape_and_reproducibility(monkeypatch):
    # Numba cannot lower captured np.random.Generator in this method on this stack,
    # so we run the exact same algorithm path with a no-op njit in tests.
    monkeypatch.setattr(lkj_chol_module, "njit", lambda fn: fn)

    d = LKJChol(1.5, 3, 123)
    s1 = d.rvs(size=2, random_state=2026)
    s2 = d.rvs(size=2, random_state=2026)
    s3 = d.rvs(size=(2, 1), random_state=2026)

    assert isinstance(s1, np.ndarray)
    assert s1.shape == (2, 3, 3)
    assert s3.shape == (2, 1, 3, 3)
    assert np.array_equal(s1, s2)

    for sample in s1:
        assert np.allclose(sample, np.tril(sample), atol=1e-12, rtol=0.0)
        assert np.all(np.diag(sample) > 0.0)
        for i in range(sample.shape[0]):
            row = sample[i, : i + 1]
            assert np.allclose(np.dot(row, row), 1.0, atol=1e-10, rtol=0.0)
