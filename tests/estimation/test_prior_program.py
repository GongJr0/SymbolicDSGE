from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from SymbolicDSGE.bayesian import make_prior
from SymbolicDSGE.estimation.prior_program import build_packed_logprior


def _scalar_prior_cases():
    return (
        (
            "normal_identity",
            make_prior(
                "normal",
                {"mean": 0.5, "std": 1.25, "random_state": 11},
                "identity",
            ),
            0.3,
            -1.1548820845188825,
        ),
        (
            "lognormal_log",
            make_prior(
                "log_normal",
                {"mean": -0.1, "std": 0.8, "random_state": 12},
                "log",
            ),
            -0.4,
            -0.766107481890463,
        ),
        (
            "halfnormal_softplus",
            make_prior(
                "half_normal",
                {"std": 1.7, "random_state": 13},
                "softplus",
            ),
            0.7,
            -1.3701626523251424,
        ),
        (
            "trunc_affine_logit",
            make_prior(
                "trunc_normal",
                {
                    "mean": 0.1,
                    "std": 0.9,
                    "low": -2.0,
                    "high": 2.0,
                    "random_state": 14,
                },
                "affine_logit",
                {"low": -2.0, "high": 2.0},
            ),
            0.2,
            -0.8020791445034389,
        ),
        (
            "halfcauchy_lower",
            make_prior(
                "half_cauchy",
                {"gamma": 2.2, "random_state": 15},
                "lower_bounded",
                {"low": 0.0},
            ),
            -0.2,
            -1.5697480509831347,
        ),
        (
            "beta_logit",
            make_prior(
                "beta",
                {"a": 2.5, "b": 4.0, "random_state": 16},
                "logit",
            ),
            1.1,
            -2.6815598930941342,
        ),
        (
            "uniform_probit",
            make_prior(
                "uniform",
                {"low": 0.0, "high": 1.0, "random_state": 17},
                "probit",
            ),
            -0.3,
            -0.9639385332046726,
        ),
        (
            "invgamma_log",
            make_prior(
                "inv_gamma",
                {"mean": 1.4, "std": 0.6, "random_state": 18},
                "log",
            ),
            0.4,
            -0.07651091863342285,
        ),
        (
            "uniform_affine_probit",
            make_prior(
                "uniform",
                {"low": -2.0, "high": 3.0, "random_state": 19},
                "affine_probit",
                {"low": -2.0, "high": 3.0},
            ),
            -0.6,
            -1.0989385332046726,
        ),
        (
            "trunc_upper",
            make_prior(
                "trunc_normal",
                {
                    "mean": 0.0,
                    "std": 1.0,
                    "low": -np.inf,
                    "high": 2.0,
                    "random_state": 20,
                },
                "upper_bounded",
                {"high": 2.0},
            ),
            -0.2,
            -1.7936241407375653,
        ),
    )


@pytest.mark.parametrize("name,prior,z,expected", _scalar_prior_cases())
def test_packed_scalar_prior_unit_matches_python_golden(name, prior, z, expected):
    packed = build_packed_logprior(
        priors={name: prior},
        param_index={name: 0},
        matrix_blocks={},
        matrix_member_names=set(),
    )

    assert packed is not None
    assert float(prior.logpdf(np.float64(z))) == pytest.approx(
        expected, rel=1e-13, abs=1e-13
    )
    assert float(packed.logpdf(np.asarray([z], dtype=np.float64))) == pytest.approx(
        expected, rel=1e-13, abs=1e-13
    )


def test_packed_scalar_program_matches_python_golden_sum():
    cases = _scalar_prior_cases()
    priors = {name: prior for name, prior, _, _ in cases}
    theta = np.asarray([z for _, _, z, _ in cases], dtype=np.float64)
    expected_parts = [expected for _, _, _, expected in cases]
    expected_total = -12.277551433095528

    packed = build_packed_logprior(
        priors=priors,
        param_index={name: i for i, name in enumerate(priors)},
        matrix_blocks={},
        matrix_member_names=set(),
    )

    assert packed is not None
    for i, (name, prior) in enumerate(priors.items()):
        assert float(prior.logpdf(np.float64(theta[i]))) == pytest.approx(
            expected_parts[i], rel=1e-13, abs=1e-13
        )
    assert sum(expected_parts) == pytest.approx(expected_total, rel=1e-15, abs=1e-15)
    assert float(packed.logpdf(theta)) == pytest.approx(
        expected_total, rel=1e-13, abs=1e-13
    )


def test_packed_lkj_block_unit_matches_python_golden():
    prior = make_prior(
        "lkj_chol",
        {"eta": 1.5, "K": 3, "random_state": 101},
        "cholesky_corr",
    )
    theta = np.asarray([0.25, -0.15, 0.45], dtype=np.float64)
    block = SimpleNamespace(
        dim=3,
        theta_indices=np.asarray([0, 1, 2], dtype=np.int64),
    )
    expected = 0.5643752975616161

    packed = build_packed_logprior(
        priors={"R_corr": prior},
        param_index={"rho10": 0, "rho20": 1, "rho21": 2},
        matrix_blocks={"R_corr": block},
        matrix_member_names={"rho10", "rho20", "rho21"},
    )

    assert packed is not None
    assert float(prior.logpdf(theta)) == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert float(packed.logpdf(theta)) == pytest.approx(expected, rel=1e-13, abs=1e-13)
