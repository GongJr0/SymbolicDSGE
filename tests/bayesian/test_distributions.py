# type: ignore
import numpy as np
import pytest
from numpy import float64
from scipy.stats import (
    beta as scipy_beta,
    gamma as scipy_gamma,
    halfcauchy as scipy_halfcauchy,
    halfnorm as scipy_halfnorm,
    invgamma as scipy_invgamma,
    lognorm as scipy_lognorm,
    norm as scipy_norm,
    truncnorm as scipy_truncnorm,
    uniform as scipy_uniform,
)

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
            lambda: Beta(2.0, 3.0, 123),
            float64(0.4),
            float64(-0.1),
            False,
        ),
        (
            "gamma",
            lambda: Gamma(6.0, np.sqrt(12.0), 123),
            float64(1.4),
            float64(-0.1),
            False,
        ),
        (
            "inv_gamma",
            lambda: InvGamma(2.0, 1.0, 123),
            float64(1.4),
            float64(0.0),
            False,
        ),
        (
            "half_normal",
            lambda: HalfNormal(1.0, 123),
            float64(0.8),
            float64(-0.1),
            False,
        ),
        (
            "half_cauchy",
            lambda: HalfCauchy(1.0, 123),
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


def _affine_scipy_logpdf(base_logpdf, x, *, shift: float, stretch: float):
    return base_logpdf((x - shift) / stretch) - np.log(stretch)


def _scaled_scipy_logpdf(base_logpdf, x, *, stretch: float):
    return base_logpdf(x / stretch) - np.log(stretch)


RVS_REFERENCE_SAMPLE_SIZE = 100_000
RVS_QUANTILES = np.array([0.1, 0.5, 0.9], dtype=np.float64)
RVS_MEAN_SIGMAS = 5.0
RVS_VAR_SIGMAS = 6.0
RVS_QUANTILE_SIGMAS = 6.0


@pytest.fixture(
    params=[
        (
            "normal",
            lambda: Normal(1.25, 0.7, 123),
            scipy_norm(loc=1.25, scale=0.7),
            True,
        ),
        (
            "lognormal",
            lambda: LogNormal(np.log(1.8), 0.45, 123),
            scipy_lognorm(s=0.45, scale=1.8),
            True,
        ),
        (
            "halfnormal",
            lambda: HalfNormal(1.4, 123),
            scipy_halfnorm(scale=1.4),
            True,
        ),
        (
            "halfcauchy",
            lambda: HalfCauchy(1.1, 123),
            scipy_halfcauchy(scale=1.1),
            False,
        ),
        (
            "truncnormal",
            lambda: TruncNormal(-1.0, 1.0, 0.0, 1.0, 123),
            scipy_truncnorm(a=-1.0, b=1.0, loc=0.0, scale=1.0),
            True,
        ),
        (
            "uniform",
            lambda: Uniform(-2.0, 3.5, 123),
            scipy_uniform(loc=-2.0, scale=5.5),
            True,
        ),
        (
            "beta",
            lambda: Beta(2.5, 4.0, 123),
            scipy_beta(a=2.5, b=4.0),
            True,
        ),
        (
            "gamma",
            lambda: Gamma(4.5, 1.3, 123),
            scipy_gamma(
                a=Gamma.to_shape(4.5, 1.3),
                scale=Gamma.to_scale(4.5, 1.3),
            ),
            True,
        ),
        (
            "invgamma",
            lambda: InvGamma(3.2, 1.8, 123),
            scipy_invgamma(
                a=InvGamma.to_shape(3.2, 1.8),
                scale=InvGamma.to_scale(3.2, 1.8),
            ),
            True,
        ),
    ],
    ids=lambda case: case[0],
)
def sampler_reference_case(request):
    return request.param


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


@pytest.mark.parametrize(
    ("ctor", "scipy_logpdf", "x_scalar", "x_vector"),
    [
        (
            lambda: Normal(1.25, 0.7, 123),
            lambda x: _affine_scipy_logpdf(
                lambda z: scipy_norm.logpdf(z),
                x,
                shift=1.25,
                stretch=0.7,
            ),
            float64(0.9),
            np.array([-0.2, 0.9, 1.7], dtype=np.float64),
        ),
        (
            lambda: LogNormal(np.log(1.8), 0.45, 123),
            lambda x: scipy_lognorm.logpdf(
                x,
                s=0.45,
                scale=np.exp(np.log(1.8)),
            ),
            float64(1.4),
            np.array([0.4, 1.4, 3.2], dtype=np.float64),
        ),
        (
            lambda: HalfNormal(1.4, 123),
            lambda x: _scaled_scipy_logpdf(
                lambda z: scipy_halfnorm.logpdf(z),
                x,
                stretch=1.4,
            ),
            float64(0.9),
            np.array([0.2, 0.9, 2.5], dtype=np.float64),
        ),
        (
            lambda: HalfCauchy(1.1, 123),
            lambda x: _scaled_scipy_logpdf(
                lambda z: scipy_halfcauchy.logpdf(z),
                x,
                stretch=1.1,
            ),
            float64(0.7),
            np.array([0.1, 0.7, 2.0], dtype=np.float64),
        ),
        (
            lambda: TruncNormal(-1.0, 1.0, 0.0, 1.0, 123),
            lambda x: _affine_scipy_logpdf(
                lambda z: scipy_truncnorm.logpdf(z, a=-1.0, b=1.0),
                x,
                shift=0.0,
                stretch=1.0,
            ),
            float64(0.2),
            np.array([-0.4, 0.2, 0.9], dtype=np.float64),
        ),
        (
            lambda: Uniform(-2.0, 3.5, 123),
            lambda x: _affine_scipy_logpdf(
                lambda z: scipy_uniform.logpdf(z),
                x,
                shift=-2.0,
                stretch=5.5,
            ),
            float64(0.4),
            np.array([-1.5, 0.4, 2.2], dtype=np.float64),
        ),
        (
            lambda: Beta(2.5, 4.0, 123),
            lambda x: scipy_beta.logpdf(x, a=2.5, b=4.0),
            float64(0.3),
            np.array([0.1, 0.3, 0.9], dtype=np.float64),
        ),
        (
            lambda: Gamma(4.5, 1.3, 123),
            lambda x: scipy_gamma.logpdf(
                x / Gamma.to_scale(4.5, 1.3),
                a=Gamma.to_shape(4.5, 1.3),
            )
            - np.log(Gamma.to_scale(4.5, 1.3)),
            float64(2.1),
            np.array([0.4, 2.1, 6.0], dtype=np.float64),
        ),
        (
            lambda: InvGamma(3.2, 1.8, 123),
            lambda x: scipy_invgamma.logpdf(
                x / InvGamma.to_scale(3.2, 1.8),
                a=InvGamma.to_shape(3.2, 1.8),
            )
            - np.log(InvGamma.to_scale(3.2, 1.8)),
            float64(1.1),
            np.array([0.7, 1.1, 3.0], dtype=np.float64),
        ),
    ],
    ids=[
        "normal",
        "lognormal",
        "halfnormal",
        "halfcauchy",
        "truncnormal",
        "uniform",
        "beta",
        "gamma",
        "invgamma",
    ],
)
def test_distribution_logpdf_matches_scipy_counterpart(
    ctor, scipy_logpdf, x_scalar, x_vector
):
    dist = ctor()

    scalar_out = dist.logpdf(x_scalar)
    scalar_expected = float64(scipy_logpdf(x_scalar))
    assert np.allclose(scalar_out, scalar_expected, rtol=1e-12, atol=1e-12)

    vector_out = np.asarray(dist.logpdf(x_vector), dtype=np.float64)
    vector_expected = np.asarray(scipy_logpdf(x_vector), dtype=np.float64)
    assert np.allclose(vector_out, vector_expected, rtol=1e-12, atol=1e-12)


def test_invgamma_mean_std_parameterization_matches_derived_scipy_moments():
    dist = InvGamma(3.2, 1.8, 123)
    a = InvGamma.to_shape(3.2, 1.8)
    beta = InvGamma.to_scale(3.2, 1.8)

    assert np.allclose(dist.mean, 3.2, rtol=1e-12, atol=1e-12)
    assert np.allclose(dist.var, 1.8**2, rtol=1e-12, atol=1e-12)
    assert np.allclose(dist.mean, beta * scipy_invgamma.mean(a=a))
    assert np.allclose(dist.var, (beta**2) * scipy_invgamma.var(a=a))


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


@pytest.mark.parametrize(
    ("name", "ctor"),
    [
        ("normal", lambda: Normal(0.0, 1.0, 123)),
        ("lognormal", lambda: LogNormal(np.log(1.8), 0.45, 123)),
        ("halfnormal", lambda: HalfNormal(1.4, 123)),
        ("halfcauchy", lambda: HalfCauchy(1.1, 123)),
        ("truncnormal", lambda: TruncNormal(-1.0, 1.0, 0.0, 1.0, 123)),
        ("uniform", lambda: Uniform(-2.0, 3.5, 123)),
        ("beta", lambda: Beta(2.5, 4.0, 123)),
        ("gamma", lambda: Gamma(4.5, 1.3, 123)),
        ("invgamma", lambda: InvGamma(3.2, 1.8, 123)),
    ],
    ids=[
        "normal",
        "lognormal",
        "halfnormal",
        "halfcauchy",
        "truncnormal",
        "uniform",
        "beta",
        "gamma",
        "invgamma",
    ],
)
def test_distribution_rvs_seed_zero_overrides_stored_seed(name, ctor):
    seed0 = np.asarray(ctor().rvs(size=8, random_state=0))
    seed0_again = np.asarray(ctor().rvs(size=8, random_state=0))
    stored = np.asarray(ctor().rvs(size=8))

    assert np.array_equal(seed0, seed0_again)
    assert not np.array_equal(seed0, stored)


def test_distribution_rvs_size_parameterization(distribution_case):
    _, ctor, _, _, _ = distribution_case
    dist = ctor()

    sample_1d = np.asarray(dist.rvs(size=5, random_state=99))
    sample_2d = np.asarray(dist.rvs(size=(2, 3), random_state=99))

    assert isinstance(sample_1d, np.ndarray)
    assert isinstance(sample_2d, np.ndarray)
    assert sample_1d.shape == (5,)
    assert sample_2d.shape == (2, 3)


def test_distribution_rvs_empirical_mean_matches_scipy(sampler_reference_case):
    _, ctor, scipy_dist, has_finite_variance = sampler_reference_case
    if not has_finite_variance:
        pytest.skip("Mean-based sampler check requires finite variance.")

    samples = np.asarray(
        ctor().rvs(size=RVS_REFERENCE_SAMPLE_SIZE, random_state=2026),
        dtype=np.float64,
    )
    expected_mean = float64(scipy_dist.mean())
    expected_var = float64(scipy_dist.var())
    mean_se = float64(np.sqrt(expected_var / RVS_REFERENCE_SAMPLE_SIZE))

    assert np.abs(samples.mean() - expected_mean) <= (RVS_MEAN_SIGMAS * mean_se + 1e-12)


def test_distribution_rvs_empirical_variance_matches_scipy(sampler_reference_case):
    _, ctor, scipy_dist, has_finite_variance = sampler_reference_case
    if not has_finite_variance:
        pytest.skip("Variance-based sampler check requires finite fourth moment.")

    samples = np.asarray(
        ctor().rvs(size=RVS_REFERENCE_SAMPLE_SIZE, random_state=2026),
        dtype=np.float64,
    )
    expected_var = float64(scipy_dist.var())
    excess_kurtosis = float64(scipy_dist.stats(moments="k"))
    var_se = float64(
        expected_var
        * np.sqrt((excess_kurtosis + 2.0) / (RVS_REFERENCE_SAMPLE_SIZE - 1))
    )

    assert np.abs(samples.var(ddof=1) - expected_var) <= (
        RVS_VAR_SIGMAS * var_se + 1e-12
    )


def test_distribution_rvs_empirical_quantiles_match_scipy(sampler_reference_case):
    _, ctor, scipy_dist, _ = sampler_reference_case

    samples = np.asarray(
        ctor().rvs(size=RVS_REFERENCE_SAMPLE_SIZE, random_state=2026),
        dtype=np.float64,
    )
    empirical = np.quantile(samples, RVS_QUANTILES)
    expected = np.asarray(scipy_dist.ppf(RVS_QUANTILES), dtype=np.float64)
    density = np.asarray(scipy_dist.pdf(expected), dtype=np.float64)
    quantile_se = np.sqrt(
        RVS_QUANTILES * (1.0 - RVS_QUANTILES) / (RVS_REFERENCE_SAMPLE_SIZE * density**2)
    )

    assert np.all(
        np.abs(empirical - expected) <= RVS_QUANTILE_SIGMAS * quantile_se + 1e-12
    )


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


def test_distribution_repr_returns_class_name(distribution_case):
    _, ctor, _, _, _ = distribution_case
    dist = ctor()
    assert repr(dist) == dist.__class__.__name__


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
    s0 = d.rvs(size=2, random_state=0)
    s0_again = d.rvs(size=2, random_state=0)
    stored = d.rvs(size=2)

    assert isinstance(s1, np.ndarray)
    assert s1.shape == (2, 3, 3)
    assert s3.shape == (2, 1, 3, 3)
    assert np.array_equal(s1, s2)
    assert np.array_equal(s0, s0_again)
    assert not np.array_equal(s0, stored)

    for sample in s1:
        assert np.allclose(sample, np.tril(sample), atol=1e-12, rtol=0.0)
        assert np.all(np.diag(sample) > 0.0)
        for i in range(sample.shape[0]):
            row = sample[i, : i + 1]
            assert np.allclose(np.dot(row, row), 1.0, atol=1e-10, rtol=0.0)
