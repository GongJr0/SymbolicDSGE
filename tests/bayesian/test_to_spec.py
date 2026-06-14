from __future__ import annotations

import pytest

from SymbolicDSGE.bayesian.distributions.lkj_chol import LKJChol
from SymbolicDSGE.bayesian.priors import make_prior
from SymbolicDSGE.bayesian.transforms.affine_logit import AffineLogitTransform
from SymbolicDSGE.bayesian.transforms.affine_probit import AffineProbitTransform
from SymbolicDSGE.bayesian.transforms.cholesky_corr import CholeskyCorrTransform
from SymbolicDSGE.bayesian.transforms.identity import Identity
from SymbolicDSGE.bayesian.transforms.log import LogTransform
from SymbolicDSGE.bayesian.transforms.logit import LogitTransform
from SymbolicDSGE.bayesian.transforms.lower_bounded import LowerBoundedTransform
from SymbolicDSGE.bayesian.transforms.probit import ProbitTransform
from SymbolicDSGE.bayesian.transforms.softplus import SoftplusTransform
from SymbolicDSGE.bayesian.transforms.upper_bounded import UpperBoundedTransform

# (distribution family, parameters, transform, transform_kwargs) — each pairs a
# distribution with a support-compatible transform.
_PRIOR_CASES = [
    ("normal", {"mean": 0.5, "std": 2.0}, "identity", {}),
    ("log_normal", {"mean": 0.1, "std": 0.3}, "log", {}),
    ("half_normal", {"std": 1.5}, "log", {}),
    (
        "trunc_normal",
        {"low": -1.0, "high": 1.0, "mean": 0.0, "std": 1.0},
        "affine_logit",
        {"low": -1.0, "high": 1.0},
    ),
    ("half_cauchy", {"gamma": 2.0}, "log", {}),
    ("beta", {"a": 2.0, "b": 3.0}, "logit", {}),
    ("gamma", {"mean": 2.0, "std": 1.0}, "log", {}),
    ("inv_gamma", {"mean": 2.0, "std": 1.0}, "log", {}),
    ("uniform", {"low": 0.0, "high": 5.0}, "affine_logit", {"low": 0.0, "high": 5.0}),
]


@pytest.mark.parametrize(
    "distribution, parameters, transform, transform_kwargs", _PRIOR_CASES
)
def test_prior_to_spec_round_trips(
    distribution, parameters, transform, transform_kwargs
):
    prior = make_prior(
        distribution=distribution,
        parameters=parameters,
        transform=transform,
        transform_kwargs=transform_kwargs,
    )
    spec = prior.to_spec()

    assert spec.distribution == distribution
    assert spec.parameters == parameters
    assert spec.transform == transform
    assert spec.transform_kwargs == transform_kwargs

    # Rebuilding from the emitted spec yields an identical spec (lossless).
    rebuilt = make_prior(
        distribution=spec.distribution,
        parameters=spec.parameters,
        transform=spec.transform,
        transform_kwargs=spec.transform_kwargs,
    ).to_spec()
    assert rebuilt.to_dict() == spec.to_dict()


def test_trivial_transforms_to_spec():
    assert Identity().to_spec() == ("identity", {})
    assert LogTransform().to_spec() == ("log", {})
    assert SoftplusTransform().to_spec() == ("softplus", {})
    assert LogitTransform().to_spec() == ("logit", {})
    assert ProbitTransform().to_spec() == ("probit", {})


def test_parametrized_transforms_to_spec():
    assert AffineLogitTransform(0.0, 1.0).to_spec() == (
        "affine_logit",
        {"low": 0.0, "high": 1.0},
    )
    assert AffineProbitTransform(-2.0, 2.0).to_spec() == (
        "affine_probit",
        {"low": -2.0, "high": 2.0},
    )
    assert LowerBoundedTransform(0.0).to_spec() == ("lower_bounded", {"low": 0.0})
    assert UpperBoundedTransform(10.0).to_spec() == ("upper_bounded", {"high": 10.0})


def test_lkj_and_cholesky_corr_to_spec():
    assert LKJChol(eta=2.0, K=3, random_state=None).to_spec() == (
        "lkj_chol",
        {"eta": 2.0, "K": 3},
    )
    assert CholeskyCorrTransform(K=3).to_spec() == ("cholesky_corr", {"K": 3})


def test_lkj_prior_to_spec_round_trips():
    prior = make_prior(
        distribution="lkj_chol",
        parameters={"eta": 2.0, "K": 3},
        transform="cholesky_corr",
        transform_kwargs={"K": 3},
    )
    spec = prior.to_spec()
    assert spec.distribution == "lkj_chol"
    assert spec.parameters == {"eta": 2.0, "K": 3}
    assert spec.transform == "cholesky_corr"
    assert spec.transform_kwargs == {"K": 3}
    rebuilt = make_prior(
        distribution=spec.distribution,
        parameters=spec.parameters,
        transform=spec.transform,
        transform_kwargs=spec.transform_kwargs,
    ).to_spec()
    assert rebuilt.to_dict() == spec.to_dict()
