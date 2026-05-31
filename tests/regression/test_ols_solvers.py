from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t

from SymbolicDSGE._diag_tests.distributions import ReferenceDistribution
from SymbolicDSGE.regression.ols import MCRegressionResult as ExportedMCRegressionResult
from SymbolicDSGE.regression.ols import OLSResult as ExportedOLSResult
from SymbolicDSGE.regression.ols import RegressionStatus as ExportedRegressionStatus
from SymbolicDSGE.regression.ols import ols as exported_ols
from SymbolicDSGE.regression.ols.diag_utils import r2, r2_adj, se, se_from_pinv
from SymbolicDSGE.regression.ols.core import ols
from SymbolicDSGE.regression.ols.ols_result import (
    MCRegressionResult,
    OLSResult,
    RegressionStatus,
)
from SymbolicDSGE.regression.ols.solvers import (
    OK,
    RANK_DEFICIENT,
    chol_solve,
    ltsq_solve,
    xtx_xty,
)


def test_chol_solve_returns_factor_for_standard_error_calculation() -> None:
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 1.9, 3.2, 3.9, 5.1], dtype=np.float64)

    coef, L, status = chol_solve(x, y)
    y_hat = x @ coef
    out = se(L, y, y_hat, x)

    sigma2 = ((y - y_hat) ** 2).sum() / (x.shape[0] - x.shape[1])
    expected = np.sqrt(np.diag(np.linalg.inv(x.T @ x) * sigma2))

    assert status == OK
    np.testing.assert_allclose(L @ L.T, x.T @ x)
    np.testing.assert_allclose(out, expected)


def test_ols_package_exports_public_entry_points() -> None:
    assert ExportedOLSResult is OLSResult
    assert ExportedMCRegressionResult is MCRegressionResult
    assert ExportedRegressionStatus is RegressionStatus
    assert exported_ols is ols


def test_ols_core_uses_cholesky_solver_and_default_variable_names() -> None:
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 1.9, 3.2, 3.9, 5.1], dtype=np.float64)

    out = ols(x, y)

    expected_coef = np.linalg.solve(x.T @ x, x.T @ y)
    assert out.variables == ["x0", "x1"]
    assert out.status is RegressionStatus.OK
    np.testing.assert_allclose(out.coefficients, expected_coef)


def test_ols_core_falls_back_to_lstsq_for_rank_deficient_design() -> None:
    x = np.array(
        [
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 6.0],
            [1.0, 4.0, 8.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.1, 2.9, 4.2], dtype=np.float64)

    out = ols(x, y, variables=["const", "x", "two_x"])

    expected_coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    assert out.variables == ["const", "x", "two_x"]
    assert out.status is RegressionStatus.RANK_DEFICIENT
    np.testing.assert_allclose(out.coefficients, expected_coef)


def test_xtx_xty_matches_matrix_products_for_manual_and_blas_paths() -> None:
    x_small = np.array(
        [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]],
        dtype=np.float64,
    )
    y_small = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    G_small, g_small = xtx_xty(x_small, y_small)

    np.testing.assert_allclose(G_small, x_small.T @ x_small)
    np.testing.assert_allclose(g_small, x_small.T @ y_small)

    x_wide = (np.arange(300, dtype=np.float64).reshape(3, 100) + 1.0) / 100.0
    y_wide = np.array([1.0, -1.0, 2.0], dtype=np.float64)

    G_wide, g_wide = xtx_xty(x_wide, y_wide)

    np.testing.assert_allclose(G_wide, x_wide.T @ x_wide)
    np.testing.assert_allclose(g_wide, x_wide.T @ y_wide)


def test_ltsq_solve_uses_empty_factor_placeholder() -> None:
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    coef, L, status = ltsq_solve(x, y)

    assert status == OK
    assert L.shape == (0, 0)
    np.testing.assert_allclose(coef, np.array([1.0, 1.0], dtype=np.float64))


def test_rank_deficient_se_falls_back_to_pseudoinverse() -> None:
    x = np.array(
        [
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 6.0],
            [1.0, 4.0, 8.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.1, 2.9, 4.2], dtype=np.float64)

    coef, L, status = ltsq_solve(x, y)
    y_hat = x @ coef
    out = se(L, y, y_hat, x=x)

    rank = np.linalg.matrix_rank(x)
    sigma2 = ((y - y_hat) ** 2).sum() / (x.shape[0] - rank)
    expected = np.sqrt(np.diag(np.linalg.pinv(x.T @ x) * sigma2))

    assert status == RANK_DEFICIENT
    assert L.shape == (0, 0)
    np.testing.assert_allclose(out, expected)


def test_pseudoinverse_se_returns_nan_without_residual_degrees_of_freedom() -> None:
    x = np.eye(3, dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    out = se_from_pinv(x, y, y)

    assert np.isnan(out).all()


def test_empty_factor_se_requires_design_matrix() -> None:
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    L = np.empty((0, 0), dtype=np.float64)

    with pytest.raises(TypeError):
        se(L, y, y)


def test_r2_and_adjusted_r2_edge_cases() -> None:
    y = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    y_hat = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert r2(y, y_hat) == np.float64(0.0)
    assert r2_adj(np.float64(0.5), n=3, k=2) == np.float64(0.0)


def test_ols_result_exposes_summary_diagnostics_and_f_test() -> None:
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 1.9, 3.2, 3.9, 5.1], dtype=np.float64)
    coef, L, status = chol_solve(x, y)

    out = OLSResult(
        variables=["const", "trend"],
        coefficients=coef,
        y=y,
        x=x,
        status=RegressionStatus(status),
        _L=L,
    )

    expected_y_hat = x @ coef
    expected_r2 = 1 - ((y - expected_y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    expected_r2_adj = 1 - (1 - expected_r2) * (out.n - 1) / (out.n - out.k - 1)
    expected_t = out.coefficients / out.se
    expected_p = 2 * (1 - t.cdf(abs(expected_t), out.n - out.k))

    np.testing.assert_allclose(out.y_hat, expected_y_hat)
    np.testing.assert_allclose(out.t_stat, expected_t)
    np.testing.assert_allclose(out.p_values, expected_p)
    assert out.r2 == pytest.approx(expected_r2)
    assert out.r2_adj == pytest.approx(expected_r2_adj)
    np.testing.assert_allclose(
        out.partial_r2,
        expected_t**2 / (expected_t**2 + out.n - out.k),
    )

    ci = out.confidence_intervals(alpha=0.1)
    assert ci.shape == (2, 2)

    summary = out.summary(alpha=0.1)
    assert list(summary.index) == ["const", "trend"]
    assert list(summary.columns) == [
        "coef",
        "std_err",
        "coef_ci_low",
        "coef_ci_high",
        "t_stat",
        "pval",
        "partial_r2",
    ]

    f_test = out.F_test(alpha=0.1)
    assert f_test.test_name == "F-test"
    assert f_test.dist is ReferenceDistribution.F
    assert f_test.df == (np.float64(out.k), np.float64(out.n - out.k - 1))
    assert np.isfinite(f_test.pval)

    as_dict = out.to_dict()
    assert as_dict["variables"] == ["const", "trend"]
    assert as_dict["status"] == RegressionStatus.OK


def test_rank_deficient_ols_result_uses_pseudoinverse_standard_errors() -> None:
    x = np.array(
        [
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 6.0],
            [1.0, 4.0, 8.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 2.1, 2.9, 4.2], dtype=np.float64)
    coef, L, status = ltsq_solve(x, y)
    out = OLSResult(
        variables=["const", "x", "two_x"],
        coefficients=coef,
        y=y,
        x=x,
        status=RegressionStatus(status),
        _L=L,
    )

    assert out.status is RegressionStatus.RANK_DEFICIENT
    np.testing.assert_allclose(out.se, se_from_pinv(x, y, out.y_hat))


def test_mc_regression_result_computes_vectorized_diagnostics() -> None:
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ],
        dtype=np.float64,
    )
    y0 = np.array([1.0, 1.9, 3.2, 3.9, 5.1], dtype=np.float64)
    y1 = np.array([0.8, 2.2, 2.9, 4.4, 4.8], dtype=np.float64)
    results = (
        ols(x, y0, variables=["const", "trend"]),
        ols(x, y1, variables=["const", "trend"]),
    )

    out = MCRegressionResult.from_results(results)

    np.testing.assert_allclose(
        out.coef_trace,
        np.vstack([result.coefficients for result in results]),
    )
    np.testing.assert_allclose(
        out.coefficients,
        np.vstack([result.coefficients for result in results]),
    )
    np.testing.assert_allclose(
        out.y_hat_trace,
        np.vstack([result.y_hat for result in results]),
    )
    np.testing.assert_allclose(
        out.se_trace,
        np.vstack([result.se for result in results]),
    )
    np.testing.assert_allclose(
        out.t_stat_trace,
        np.vstack([result.t_stat for result in results]),
    )
    np.testing.assert_allclose(
        out.pval_trace,
        np.vstack([result.p_values for result in results]),
    )
    np.testing.assert_allclose(
        out.r2_trace,
        np.asarray([result.r2 for result in results], dtype=np.float64),
    )
    np.testing.assert_allclose(
        out.r2_adj_trace,
        np.asarray([result.r2_adj for result in results], dtype=np.float64),
    )
    np.testing.assert_allclose(
        out.partial_r2_trace,
        np.vstack([result.partial_r2 for result in results]),
    )
    np.testing.assert_allclose(
        out.F_stat_trace,
        np.asarray([result.F_test().statistic for result in results]),
    )
    np.testing.assert_allclose(
        out.F_pval_trace,
        np.asarray([result.F_test().pval for result in results]),
    )

    ci = out.confidence_intervals(alpha=0.1)
    assert ci.shape == (2, 2, 2)
    np.testing.assert_allclose(ci[0], results[0].confidence_intervals(alpha=0.1))

    summary = out.summary(alpha=0.1)
    assert list(summary.index.names) == ["rep", "variable"]
    assert list(summary.columns) == [
        "coef",
        "std_err",
        "coef_ci_low",
        "coef_ci_high",
        "t_stat",
        "pval",
        "partial_r2",
    ]

    f_test = out.F_test(alpha=0.1)
    assert f_test.test_name == "F-test"
    assert f_test.dist is ReferenceDistribution.F
    assert f_test.df == (np.float64(out.k), np.float64(out.n - out.k - 1))
    np.testing.assert_allclose(f_test.statistic_trace, out.F_stat_trace)

    as_dict = out.to_dict()
    assert as_dict["variables"] == ["const", "trend"]
    assert as_dict["status_trace"] == (RegressionStatus.OK, RegressionStatus.OK)


def test_mc_regression_result_falls_back_to_per_rep_se_for_rank_deficient_runs() -> (
    None
):
    x = np.array(
        [
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [1.0, 3.0, 6.0],
            [1.0, 4.0, 8.0],
        ],
        dtype=np.float64,
    )
    results = (
        ols(x, np.array([1.0, 2.1, 2.9, 4.2], dtype=np.float64)),
        ols(x, np.array([0.8, 2.4, 3.2, 3.9], dtype=np.float64)),
    )

    out = MCRegressionResult.from_results(results)

    assert out.status_trace == (
        RegressionStatus.RANK_DEFICIENT,
        RegressionStatus.RANK_DEFICIENT,
    )
    np.testing.assert_allclose(
        out.se_trace,
        np.vstack([result.se for result in results]),
    )


def test_mc_regression_result_validates_compatible_runs() -> None:
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    first = ols(x, np.array([1.0, 2.0, 3.0], dtype=np.float64), variables=["c", "x"])
    second = ols(
        x,
        np.array([1.5, 2.5, 3.5], dtype=np.float64),
        variables=["const", "trend"],
    )

    with pytest.raises(ValueError, match="incompatible variables"):
        MCRegressionResult.from_results((first, second))
