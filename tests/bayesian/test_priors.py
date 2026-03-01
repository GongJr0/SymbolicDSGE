# type: ignore
import warnings

import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.bayesian.priors import Prior, make_prior
from SymbolicDSGE.bayesian.support import OutOfSupportError, Support
from SymbolicDSGE.bayesian.transforms import AffineLogitTransform, Identity


class _DummyDist:
    def __init__(self, support: Support):
        self._support = support

    @property
    def support(self) -> Support:
        return self._support


class _DummyTransform:
    def __init__(self, support: Support, maps_to: Support):
        self._support = support
        self._maps_to = maps_to

    @property
    def support(self) -> Support:
        return self._support

    @property
    def maps_to(self) -> Support:
        return self._maps_to


class _TrackingDist:
    def __init__(self):
        self.logged_x = None
        self.grad_x = None
        self.logpdf_calls = 0
        self.grad_calls = 0

    def logpdf(self, x):
        self.logpdf_calls += 1
        self.logged_x = x
        return float64(2.0 * x)

    def grad_logpdf(self, x):
        self.grad_calls += 1
        self.grad_x = x
        return float64(11.0 * x)

    def rvs(self, size, random_state=None):
        return np.zeros(size, dtype=np.float64)

    @property
    def support(self) -> Support:
        return Support(
            float64(-np.inf),
            float64(np.inf),
            low_inclusive=False,
            high_inclusive=False,
        )


class _TrackingTransform:
    def __init__(self):
        self.forward_arg = None
        self.inverse_arg = None
        self.logdet_inv_arg = None
        self.grad_inv_arg = None
        self.grad_logdet_inv_arg = None
        self.forward_calls = 0
        self.inverse_calls = 0
        self.logdet_inv_calls = 0
        self.grad_inv_calls = 0
        self.grad_logdet_inv_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        self.forward_arg = x
        return float64(x + 1.0)

    def inverse(self, y):
        self.inverse_calls += 1
        self.inverse_arg = y
        return float64(y - 1.0)

    def grad_inverse(self, y):
        self.grad_inv_calls += 1
        self.grad_inv_arg = y
        return float64(3.0)

    def log_det_abs_jacobian_inverse(self, y):
        self.logdet_inv_calls += 1
        self.logdet_inv_arg = y
        return float64(5.0)

    def grad_log_det_abs_jacobian_inverse(self, y):
        self.grad_logdet_inv_calls += 1
        self.grad_logdet_inv_arg = y
        return float64(7.0)

    @property
    def support(self) -> Support:
        return Support(
            float64(-np.inf),
            float64(np.inf),
            low_inclusive=False,
            high_inclusive=False,
        )

    @property
    def maps_to(self) -> Support:
        return Support(
            float64(-np.inf),
            float64(np.inf),
            low_inclusive=False,
            high_inclusive=False,
        )


def test_make_prior_builds_instances_and_applies_defaults():
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 2.5},
        transform="identity",
    )

    assert isinstance(prior.transform, Identity)
    assert np.allclose(prior.dist.mean, 2.5)
    # `std` default is 1.0, so variance should be 1.0
    assert np.allclose(prior.dist.var, 1.0)


def test_make_prior_passes_transform_kwargs():
    prior = make_prior(
        distribution="uniform",
        parameters={"a": -2.0, "b": 3.0},
        transform="affine_logit",
        transform_kwargs={"low": -2.0, "high": 3.0},
    )

    assert isinstance(prior.transform, AffineLogitTransform)
    assert np.allclose(prior.transform.low, -2.0)
    assert np.allclose(prior.transform.high, 3.0)


def test_make_prior_rejects_unrecognized_distribution_parameter():
    with pytest.raises(ValueError, match="Unrecognized parameters"):
        make_prior(
            distribution="normal",
            parameters={"not_a_param": 1.0},
            transform="identity",
        )


def test_make_prior_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="Unsupported distribution family"):
        make_prior(
            distribution="does_not_exist",
            parameters={},
            transform="identity",
        )


def test_make_prior_rejects_unknown_transform():
    with pytest.raises(ValueError, match="Unsupported transform method"):
        make_prior(
            distribution="normal",
            parameters={},
            transform="does_not_exist",
        )


def test_prior_logpdf_identity_matches_distribution():
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    x = float64(0.2)
    assert np.allclose(prior.logpdf(x), prior.dist.logpdf(x))


def test_prior_grad_logpdf_identity_matches_distribution():
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    x = float64(-0.4)
    assert np.allclose(prior.grad_logpdf(x), prior.dist.grad_logpdf(x))


def test_prior_bounded_methods_raise_outside_distribution_support():
    prior = make_prior(
        distribution="gamma",
        parameters={"a": 3.0, "loc": 0.0, "scale": 2.0},
        transform="identity",
    )
    bad_x = float64(-0.1)
    with pytest.raises(OutOfSupportError):
        prior.logpdf(bad_x)
    with pytest.raises(OutOfSupportError):
        prior.grad_logpdf(bad_x)


def test_prior_logit_prior_accepts_unconstrained_input_domain():
    prior = make_prior(
        distribution="beta",
        parameters={"a": 2.0, "b": 3.0, "loc": 0.0, "scale": 1.0},
        transform="logit",
    )

    z = float64(2.0)
    val = prior.logpdf(z)
    grad = prior.grad_logpdf(z)
    assert np.isfinite(val)
    assert np.isfinite(grad)


def test_prior_rvs_seed_reproducibility_and_size_shape():
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    s1 = np.asarray(prior.rvs(size=(2, 3), random_state=2026))
    s2 = np.asarray(prior.rvs(size=(2, 3), random_state=2026))
    s3 = np.asarray(prior.rvs(size=5, random_state=2026))

    assert s1.shape == (2, 3)
    assert s3.shape == (5,)
    assert np.array_equal(s1, s2)


def test_prior_support_and_maps_to_proxy_underlying_components():
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    assert prior.support == prior.dist.support
    assert prior.maps_to == prior.transform.maps_to


def test_prior_logpdf_uses_inverse_and_adds_inverse_logdet_term():
    dist = _TrackingDist()
    transform = _TrackingTransform()
    prior = Prior(
        dist=dist,  # type: ignore[arg-type]
        transform=transform,  # type: ignore[arg-type]
    )

    z = float64(2.0)
    out = prior.logpdf(z)

    # Expected: dist.logpdf(inverse(z)) + log|dx/dz|
    expected = float64(2.0 * (z - 1.0) + 5.0)
    assert np.allclose(out, expected)

    assert transform.forward_calls == 0
    assert transform.logdet_inv_calls == 1
    assert transform.inverse_calls == 1
    assert transform.inverse_arg == z
    assert transform.logdet_inv_arg == z
    assert dist.logpdf_calls == 1
    assert np.allclose(dist.logged_x, z - 1.0)


def test_prior_grad_logpdf_uses_inverse_chain_rule_and_jacobian_gradient():
    dist = _TrackingDist()
    transform = _TrackingTransform()
    prior = Prior(
        dist=dist,  # type: ignore[arg-type]
        transform=transform,  # type: ignore[arg-type]
    )

    z = float64(4.0)
    out = prior.grad_logpdf(z)

    # Expected: grad_inverse(z) * dist.grad_logpdf(inverse(z)) + grad log|dx/dz|
    expected = float64(3.0 * (11.0 * (z - 1.0)) + 7.0)
    assert np.allclose(out, expected)

    assert transform.inverse_calls == 1
    assert transform.grad_inv_calls == 1
    assert transform.grad_logdet_inv_calls == 1
    assert transform.forward_calls == 0
    assert transform.inverse_arg == z
    assert transform.grad_inv_arg == z
    assert transform.grad_logdet_inv_arg == z
    assert dist.grad_calls == 1
    assert np.allclose(dist.grad_x, z - 1.0)


def test_confirm_bound_match_raises_on_dist_support_vs_transform_maps_to_mismatch():
    prior = make_prior(
        distribution="gamma",
        parameters={"a": 2.0, "loc": 0.0, "scale": 1.0},
        transform="identity",
    )
    with pytest.raises(ValueError, match="does not match transform maps_to"):
        prior._confirm_bound_match()


def test_confirm_bound_match_raises_on_dist_support_vs_transform_support_mismatch():
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="log",
    )
    with pytest.raises(ValueError, match="transform's support function must match"):
        prior._confirm_bound_match()


def test_confirm_bound_match_warns_for_non_finite_matching_support():
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 0.0, "std": 1.0},
        transform="identity",
    )
    with pytest.warns(UserWarning, match="non-finite support"):
        prior._confirm_bound_match()


def test_confirm_bound_match_allows_finite_matching_support_without_warning():
    finite_support = Support(float64(0.0), float64(1.0))
    prior = Prior(
        dist=_DummyDist(support=finite_support),  # type: ignore[arg-type]
        transform=_DummyTransform(  # type: ignore[arg-type]
            support=finite_support,
            maps_to=finite_support,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prior._confirm_bound_match()
    assert not caught
