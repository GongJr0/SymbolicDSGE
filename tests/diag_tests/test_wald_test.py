from __future__ import annotations

import numpy as np

from SymbolicDSGE._diag_tests.distributions import PvalMethod
from SymbolicDSGE._diag_tests.status import TestStatus
from _oracles.diag import (
    jit_fill_centered,
    jit_fill_mean_ax0,
    jit_fill_symmetric_target_vec,
    jit_symmetric_outer_prod_2dim,
)
from SymbolicDSGE._diag_tests.wald_test import (
    ERR_BAD_SHAPE,
    ERR_LINALG,
    OK,
    jit_wald_stat_from_mean_and_cov,
    wald_covariance_hac,
    wald_mean_hac,
    wald_second_moment_hac,
)


def _manual_bartlett_hac(r: np.ndarray, bandwidth: int) -> np.ndarray:
    n = r.shape[0]
    max_lag = min(bandwidth, n - 1)
    omega = r.T @ r / n

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma_l = r[lag:].T @ r[:-lag] / n
        omega = omega + weight * (gamma_l + gamma_l.T)

    return omega


def _quadratic_wald_stat(
    g: np.ndarray,
    target: np.ndarray,
    *,
    center: bool,
    bandwidth: int,
) -> float:
    x = g - g.mean(axis=0) if center else g
    vech_idx = np.triu_indices(target.shape[0])
    quadratic_moments = np.empty((g.shape[0], vech_idx[0].size), dtype=np.float64)

    for i in range(g.shape[0]):
        quadratic_moments[i] = np.outer(x[i], x[i])[vech_idx]

    mean = quadratic_moments.mean(axis=0)
    centered_moments = quadratic_moments - mean
    omega = _manual_bartlett_hac(centered_moments, bandwidth)
    dev = mean - target[vech_idx]
    # Match the production Cholesky solve so the oracle stays same-algorithm.
    L = np.linalg.cholesky(omega)
    solved = np.linalg.solve(L.T, np.linalg.solve(L, dev))
    return float(g.shape[0] * dev @ solved)


def _quadratic_sample() -> np.ndarray:
    return np.ascontiguousarray(
        np.array(
            [
                [1.0, 2.0],
                [2.0, -1.0],
                [0.0, 1.0],
                [3.0, 0.0],
                [-1.0, 2.0],
                [1.5, -0.5],
                [-2.0, 1.0],
                [0.5, 3.0],
            ],
            dtype=np.float64,
        )
    )


def test_wald_mean_hac_uses_right_tail_p_value_method() -> None:
    g = np.ascontiguousarray(
        np.array(
            [
                [1.0, 2.0],
                [2.0, -1.0],
                [0.0, 1.0],
                [3.0, 0.0],
                [-1.0, 2.0],
            ],
            dtype=np.float64,
        )
    )

    out = wald_mean_hac(g, np.zeros(2, dtype=np.float64), bandwidth=0)

    assert out.pval_method is PvalMethod.SF
    assert out.status is TestStatus.OK


def test_wald_mean_hac_rejects_large_mean_deviation() -> None:
    g = np.ascontiguousarray(
        np.array(
            [
                [1.0, 2.0],
                [2.0, -1.0],
                [0.0, 1.0],
                [3.0, 0.0],
                [-1.0, 2.0],
            ],
            dtype=np.float64,
        )
    )

    out = wald_mean_hac(g, np.zeros(2, dtype=np.float64), bandwidth=0, alpha=0.05)

    assert out.df == 2
    assert out.statistic > 0.0
    assert out.pval < 0.05
    assert out.is_significant()


def test_wald_covariance_hac_matches_manual_vech_statistic() -> None:
    g = _quadratic_sample()
    target = np.eye(2, dtype=np.float64)

    out = wald_covariance_hac(g, target, bandwidth=0)
    manual_stat = _quadratic_wald_stat(
        g,
        target,
        center=True,
        bandwidth=0,
    )

    assert out.test_name == "wald_covariance_hac"
    assert out.pval_method is PvalMethod.SF
    assert out.df == 3
    np.testing.assert_allclose(out.statistic, manual_stat, rtol=1e-14, atol=1e-14)


def test_wald_second_moment_hac_matches_manual_vech_statistic() -> None:
    g = _quadratic_sample()
    target = np.eye(2, dtype=np.float64)

    out = wald_second_moment_hac(g, target, bandwidth=0)
    manual_stat = _quadratic_wald_stat(
        g,
        target,
        center=False,
        bandwidth=0,
    )

    assert out.test_name == "wald_second_moment_hac"
    assert out.pval_method is PvalMethod.SF
    assert out.df == 3
    np.testing.assert_allclose(out.statistic, manual_stat, rtol=1e-14, atol=1e-14)


def test_wald_quadratic_hac_rejects_non_symmetric_targets() -> None:
    g = _quadratic_sample()
    target = np.array([[1.0, 0.1], [0.2, 1.0]], dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "covariance matrix.*symmetric"):
        wald_covariance_hac(g, target, bandwidth=0)

    with np.testing.assert_raises_regex(ValueError, "second moment matrix.*symmetric"):
        wald_second_moment_hac(g, target, bandwidth=0)


def test_wald_quadratic_hac_rejects_target_dimension_mismatch() -> None:
    g = _quadratic_sample()
    target = np.eye(3, dtype=np.float64)

    with np.testing.assert_raises_regex(ValueError, "covariance matrix dimension"):
        wald_covariance_hac(g, target, bandwidth=0)

    with np.testing.assert_raises_regex(ValueError, "second moment matrix dimension"):
        wald_second_moment_hac(g, target, bandwidth=0)


def test_jit_wald_stat_from_mean_and_cov_returns_shape_error_for_bad_target() -> None:
    err, stat, df = jit_wald_stat_from_mean_and_cov.py_func(
        np.zeros(2, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(2, dtype=np.float64),
        10,
    )

    assert err == ERR_BAD_SHAPE
    assert np.isnan(stat)
    assert df == 2


def test_jit_wald_stat_from_mean_and_cov_computes_valid_statistic() -> None:
    err, stat, df = jit_wald_stat_from_mean_and_cov.py_func(
        np.array([1.0, 2.0], dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.eye(2, dtype=np.float64),
        3,
    )

    assert err == OK
    assert stat == np.float64(15.0)
    assert df == 2


def test_jit_wald_stat_from_mean_and_cov_returns_linalg_error_for_singular_covariance() -> (
    None
):
    err, stat, df = jit_wald_stat_from_mean_and_cov.py_func(
        np.array([1.0, 0.0], dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        10,
    )

    assert err == ERR_LINALG
    assert ERR_LINALG == TestStatus.LINALG
    assert np.isnan(stat)
    assert df == 2


def test_moment_fill_helpers_write_mean_and_centered_buffers() -> None:
    x = np.array([[1.0, 2.0], [3.0, 0.0], [5.0, 4.0]], dtype=np.float64)
    mean = np.empty(2, dtype=np.float64)
    centered = np.empty_like(x)

    jit_fill_mean_ax0(x, mean)
    jit_fill_centered(x, mean, centered)

    np.testing.assert_allclose(mean, x.mean(axis=0))
    np.testing.assert_allclose(centered, x - x.mean(axis=0))


def test_symmetric_outer_product_reports_bad_output_shape() -> None:
    x = np.arange(6.0, dtype=np.float64).reshape(3, 2)
    out = np.empty((3, 2), dtype=np.float64)

    err = jit_symmetric_outer_prod_2dim(x, out)

    assert err == ERR_BAD_SHAPE


def test_symmetric_outer_product_fills_upper_triangle_moments() -> None:
    x = np.array([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
    out = np.empty((2, 3), dtype=np.float64)

    err = jit_symmetric_outer_prod_2dim(x, out)

    assert err == OK
    np.testing.assert_allclose(
        out,
        np.array([[1.0, 2.0, 4.0], [9.0, -3.0, 1.0]], dtype=np.float64),
    )


def test_symmetric_target_vector_helper_fills_and_rejects_bad_shapes() -> None:
    target = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.float64)
    out = np.empty(3, dtype=np.float64)

    err = jit_fill_symmetric_target_vec(target, out)

    assert err == OK
    np.testing.assert_allclose(out, np.array([1.0, 0.25, 2.0], dtype=np.float64))

    bad_target = np.array([[1.0, 0.1], [0.2, 1.0]], dtype=np.float64)
    assert jit_fill_symmetric_target_vec(bad_target, out) == ERR_BAD_SHAPE

    bad_out = np.empty(2, dtype=np.float64)
    assert jit_fill_symmetric_target_vec(target, bad_out) == ERR_BAD_SHAPE


def test_wald_public_wrappers_validate_targets_and_report_failures() -> None:
    g = _quadratic_sample()

    with np.testing.assert_raises_regex(ValueError, "covariance matrix must be square"):
        wald_covariance_hac(g, np.ones(2, dtype=np.float64), bandwidth=0)

    with np.testing.assert_raises_regex(
        ValueError, "second moment matrix must be square"
    ):
        wald_second_moment_hac(g, np.ones(2, dtype=np.float64), bandwidth=0)

    with np.testing.assert_raises_regex(ValueError, "Wald test failed"):
        wald_mean_hac(
            np.ones((4, 2), dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            bandwidth=0,
        )
