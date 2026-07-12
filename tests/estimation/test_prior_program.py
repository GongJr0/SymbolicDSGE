from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from SymbolicDSGE.bayesian import make_prior
from SymbolicDSGE.estimation.prior_program import (
    DistCode,
    N_DIST_PARAMS,
    N_TRANSFORM_PARAMS,
    TransformCode,
    _pack_distribution,
    _pack_transform,
    build_packed_logprior,
)
from _oracles.estimation import (
    _dist_logpdf,
    _evaluate_logprior_program,
    _lkj_chol_logjac,
    _lkj_chol_logpdf_from_z,
    _log_sigmoid_scalar,
    _sigmoid_scalar,
    _softplus_scalar,
    _std_norm_cdf,
    _std_norm_logpdf,
    _transform_inverse_and_logjac,
)


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
        theta_slice=slice(0, 3),
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


def test_packed_logprior_matches_cache_identity_and_rejects_unsupported_specs():
    prior = make_prior(
        "normal",
        {"mean": 0.0, "std": 1.0, "random_state": 1},
        "identity",
    )
    packed = build_packed_logprior(
        priors={"rho": prior},
        param_index={"rho": 0},
        matrix_blocks={},
        matrix_member_names=set(),
    )

    assert (
        build_packed_logprior(
            priors=None,
            param_index={},
            matrix_blocks={},
            matrix_member_names=set(),
        )
        is None
    )
    assert packed is not None
    assert packed.matches({"rho": prior})
    assert not packed.matches(None)
    assert not packed.matches({"other": prior})
    assert not packed.matches(
        {
            "rho": make_prior(
                "normal",
                {"mean": 0.0, "std": 1.0, "random_state": 2},
                "identity",
            )
        }
    )

    block = SimpleNamespace(dim=2, theta_slice=slice(0, 1))
    assert (
        build_packed_logprior(
            priors={"corr": object()},
            param_index={},
            matrix_blocks={"corr": block},
            matrix_member_names=set(),
        )
        is None
    )
    assert (
        build_packed_logprior(
            priors={"corr": prior},
            param_index={},
            matrix_blocks={"corr": block},
            matrix_member_names=set(),
        )
        is None
    )
    assert (
        build_packed_logprior(
            priors={"rho": prior},
            param_index={},
            matrix_blocks={},
            matrix_member_names=set(),
        )
        is None
    )
    assert (
        build_packed_logprior(
            priors={"rho": object()},
            param_index={"rho": 0},
            matrix_blocks={},
            matrix_member_names=set(),
        )
        is None
    )
    assert (
        build_packed_logprior(
            priors={"rho": object()},
            param_index={},
            matrix_blocks={},
            matrix_member_names={"rho"},
        )
        is not None
    )
    assert _pack_distribution(object())[0] is None
    assert _pack_transform(object())[0] is None

    unsupported_dist_prior = make_prior(
        "normal",
        {"mean": 0.0, "std": 1.0, "random_state": 1},
        "identity",
    )
    object.__setattr__(unsupported_dist_prior, "dist", object())
    assert (
        build_packed_logprior(
            priors={"rho": unsupported_dist_prior},
            param_index={"rho": 0},
            matrix_blocks={},
            matrix_member_names=set(),
        )
        is None
    )


def test_prior_program_scalar_transform_helpers_cover_all_branches():
    assert _softplus_scalar(np.float64(2.0)) == pytest.approx(np.log1p(np.exp(2.0)))
    assert _softplus_scalar(np.float64(-2.0)) == pytest.approx(np.log1p(np.exp(-2.0)))
    assert _log_sigmoid_scalar(np.float64(2.0)) == pytest.approx(
        -np.log1p(np.exp(-2.0))
    )
    assert _log_sigmoid_scalar(np.float64(-2.0)) == pytest.approx(
        -2.0 - np.log1p(np.exp(-2.0))
    )
    assert _sigmoid_scalar(np.float64(2.0)) == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))
    assert _sigmoid_scalar(np.float64(-2.0)) == pytest.approx(
        np.exp(-2.0) / (1.0 + np.exp(-2.0))
    )
    assert _std_norm_cdf(np.float64(0.0)) == pytest.approx(0.5)
    assert _std_norm_logpdf(np.float64(0.0)) == pytest.approx(
        -0.5 * np.log(2.0 * np.pi)
    )

    params = np.array([1.0, 3.0, 2.0], dtype=np.float64)
    for code in (
        TransformCode.IDENTITY,
        TransformCode.LOG,
        TransformCode.SOFTPLUS,
        TransformCode.LOGIT,
        TransformCode.PROBIT,
        TransformCode.AFFINE_LOGIT,
        TransformCode.AFFINE_PROBIT,
        TransformCode.LOWER_BOUNDED,
        TransformCode.UPPER_BOUNDED,
    ):
        x, logjac = _transform_inverse_and_logjac(
            code,
            params,
            np.float64(0.25),
        )
        assert np.isfinite(x)
        assert np.isfinite(logjac)

    x, logjac = _transform_inverse_and_logjac(
        999,
        params,
        np.float64(0.25),
    )
    assert np.isnan(x)
    assert np.isnan(logjac)


def test_prior_program_distribution_logpdf_dispatch_and_support_edges():
    # The test only packs/evaluates prior.dist, but Prior now requires the
    # distribution support to contain the transform support, so each family is
    # paired with its natural transform (identity/log/logit/affine_logit).
    _pm1p1 = {"low": -1.0, "high": 1.0}
    scalar_priors = {
        "normal": make_prior(
            "normal", {"mean": 0.0, "std": 1.0, "random_state": 1}, "identity"
        ),
        "log_normal": make_prior(
            "log_normal", {"mean": 0.0, "std": 0.5, "random_state": 1}, "log"
        ),
        "half_normal": make_prior(
            "half_normal", {"std": 1.0, "random_state": 1}, "log"
        ),
        "trunc_normal": make_prior(
            "trunc_normal",
            {"mean": 0.0, "std": 1.0, "low": -1.0, "high": 1.0, "random_state": 1},
            "affine_logit",
            _pm1p1,
        ),
        "half_cauchy": make_prior(
            "half_cauchy", {"gamma": 1.0, "random_state": 1}, "log"
        ),
        "beta": make_prior("beta", {"a": 2.0, "b": 3.0, "random_state": 1}, "logit"),
        "gamma": make_prior(
            "gamma", {"mean": 2.0, "std": 1.0, "random_state": 1}, "log"
        ),
        "inv_gamma": make_prior(
            "inv_gamma", {"mean": 2.0, "std": 1.0, "random_state": 1}, "log"
        ),
        "uniform": make_prior(
            "uniform",
            {"low": -1.0, "high": 1.0, "random_state": 1},
            "affine_logit",
            _pm1p1,
        ),
    }
    valid_x = {
        DistCode.NORMAL: 0.25,
        DistCode.LOG_NORMAL: 1.2,
        DistCode.HALF_NORMAL: 0.25,
        DistCode.TRUNC_NORMAL: 0.25,
        DistCode.HALF_CAUCHY: 0.25,
        DistCode.BETA: 0.25,
        DistCode.GAMMA: 1.5,
        DistCode.INV_GAMMA: 1.5,
        DistCode.UNIFORM: 0.25,
    }

    for prior in scalar_priors.values():
        code, params = _pack_distribution(prior.dist)
        assert code is not None
        out = _dist_logpdf(
            code,
            np.asarray(params, dtype=np.float64),
            np.float64(valid_x[code]),
        )
        assert np.isfinite(out)

    for code, params, bad_x in (
        (DistCode.LOG_NORMAL, [0.0, 1.0, 0.0, 0.0, 0.0], -1.0),
        (DistCode.HALF_NORMAL, [1.0, 0.0, 0.0, 0.0, 0.0], -1.0),
        (DistCode.TRUNC_NORMAL, [0.0, 1.0, -1.0, 1.0, 0.0], 2.0),
        (DistCode.HALF_CAUCHY, [1.0, 0.0, 0.0, 0.0, 0.0], -1.0),
        (DistCode.BETA, [1.0, 1.0, 0.0, 0.0, 0.0], 2.0),
        (DistCode.GAMMA, [1.0, 1.0, 0.0, 0.0, 0.0], -1.0),
        (DistCode.INV_GAMMA, [1.0, 1.0, 0.0, 0.0, 0.0], 0.0),
        (DistCode.UNIFORM, [-1.0, 1.0, 2.0, 0.0, 0.0], 2.0),
        (999, [0.0] * N_DIST_PARAMS, 0.0),
    ):
        assert np.isnan(
            _dist_logpdf(
                code,
                np.asarray(params, dtype=np.float64),
                np.float64(bad_x),
            )
        )


def test_prior_program_lkj_and_evaluator_python_paths_cover_nan_branches():
    z = np.array([0.25, -0.15, 0.45], dtype=np.float64)

    assert np.isfinite(_lkj_chol_logjac(z, 3, z.size))
    assert np.isnan(_lkj_chol_logjac(z[:1], 3, 1))
    assert np.isfinite(
        _lkj_chol_logpdf_from_z(
            z,
            3,
            z.size,
            np.float64(1.5),
            np.float64(0.0),
        )
    )
    assert np.isnan(
        _lkj_chol_logpdf_from_z(
            z[:1],
            3,
            1,
            np.float64(1.5),
            np.float64(0.0),
        )
    )

    theta = np.array([0.25], dtype=np.float64)
    scalar_indices = np.array([0], dtype=np.int64)
    scalar_dist_codes = np.array([DistCode.NORMAL], dtype=np.int64)
    scalar_transform_codes = np.array([TransformCode.IDENTITY], dtype=np.int64)
    scalar_dist_params = np.zeros((1, N_DIST_PARAMS), dtype=np.float64)
    scalar_dist_params[0, 1] = 1.0
    scalar_transform_params = np.zeros((1, N_TRANSFORM_PARAMS), dtype=np.float64)
    empty_i = np.empty((0,), dtype=np.int64)
    empty_f = np.empty((0,), dtype=np.float64)
    empty_m = np.empty((0,), dtype=np.int64)

    assert np.isfinite(
        _evaluate_logprior_program(
            theta,
            scalar_indices,
            scalar_dist_codes,
            scalar_transform_codes,
            scalar_dist_params,
            scalar_transform_params,
            empty_m,
            empty_i,
            empty_i,
            empty_f,
            empty_f,
        )
    )
    assert np.isnan(
        _evaluate_logprior_program(
            np.array([-0.25], dtype=np.float64),
            scalar_indices,
            np.array([DistCode.LOG_NORMAL], dtype=np.int64),
            scalar_transform_codes,
            scalar_dist_params,
            scalar_transform_params,
            empty_m,
            empty_i,
            empty_i,
            empty_f,
            empty_f,
        )
    )
    assert np.isnan(
        _evaluate_logprior_program(
            theta,
            scalar_indices,
            scalar_dist_codes,
            np.array([999], dtype=np.int64),
            scalar_dist_params,
            scalar_transform_params,
            empty_m,
            empty_i,
            empty_i,
            empty_f,
            empty_f,
        )
    )

    assert np.isnan(
        _evaluate_logprior_program(
            np.array([0.1], dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, N_DIST_PARAMS), dtype=np.float64),
            np.empty((0, N_TRANSFORM_PARAMS), dtype=np.float64),
            np.array([0], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([1.5], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
    )

    assert np.isfinite(
        _evaluate_logprior_program(
            np.array([0.25, -0.15, 0.45], dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, N_DIST_PARAMS), dtype=np.float64),
            np.empty((0, N_TRANSFORM_PARAMS), dtype=np.float64),
            np.array([0], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([3], dtype=np.int64),
            np.array([1.5], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
    )
