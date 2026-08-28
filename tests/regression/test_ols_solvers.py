from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t

from SymbolicDSGE._diag_tests.distributions import ReferenceDistribution
from SymbolicDSGE._diag_tests.status import TestStatus
from SymbolicDSGE.regression import (
    MCRegressionResult as ExportedMCRegressionResult,
)
from SymbolicDSGE.regression.ols import OLSResult as ExportedOLSResult
from SymbolicDSGE.regression.ols import RegressionStatus as ExportedRegressionStatus
from SymbolicDSGE.regression.ols import ols as exported_ols
from SymbolicDSGE.regression.ols.diag_utils import (
    r2,
    r2_adj,
    se,
    se_from_cholesky,
    se_from_pinv,
)
from SymbolicDSGE.regression.ols.core import ols
from SymbolicDSGE.regression.enums import RegressionStatus
from SymbolicDSGE.regression.ols.ols_result import OLSResult
from SymbolicDSGE.regression.result import MCRegressionResult
from SymbolicDSGE.regression.ols.solvers import (
    OK,
    RANK_DEFICIENT,
    chol_solve,
    ltsq_solve,
    xtx_xty,
)


def _mc_regression_from_ols(results: tuple[OLSResult, ...]) -> MCRegressionResult:
    first = results[0]
    return MCRegressionResult(
        kind="ols",
        variables=first.variables,
        coef_trace=np.vstack([result.coefficients for result in results]),
        ssr_trace=np.asarray([result.ssr for result in results], dtype=np.float64),
        sst_trace=np.asarray([result.sst for result in results], dtype=np.float64),
        _se_trace=np.vstack([result.se for result in results]),
        n_retained=len(results),
        retained_reps=np.arange(len(results), dtype=np.int_),
        n_rep=len(results),
        n=first.n,
        k=first.k,
        _raw_status=np.asarray(
            [int(result.status) for result in results], dtype=np.int_
        ),
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
    np.testing.assert_allclose(se_from_cholesky.py_func(L, y, y_hat), expected)


def test_ols_package_exports_public_entry_points() -> None:
    assert ExportedOLSResult is OLSResult
    assert ExportedRegressionStatus is RegressionStatus
    assert exported_ols is ols


def test_regression_package_exports_mc_result() -> None:
    assert ExportedMCRegressionResult is MCRegressionResult


def test_ols_core_uses_cholesky_solver_and_default_variable_names() -> None:
    x = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, 1.9, 3.2, 3.9, 5.1], dtype=np.float64)

    out = ols(x, y)

    X = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    expected_coef = np.linalg.solve(X.T @ X, X.T @ y)
    assert out.variables == ["Intercept", "x0"]
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

    out = ols(x, y, variables=["const", "x", "two_x"], intercept=False)

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
        X=x,
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
        X=x,
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
        ols(x, y0, variables=["const", "trend"], intercept=False),
        ols(x, y1, variables=["const", "trend"], intercept=False),
    )

    out = _mc_regression_from_ols(results)

    np.testing.assert_allclose(
        out.coef_trace,
        np.vstack([result.coefficients for result in results]),
    )
    np.testing.assert_allclose(
        out.coefficients,
        np.vstack([result.coefficients for result in results]),
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

    trace_frame = out.trace_frame(alpha=0.1)
    assert list(trace_frame.index.names) == ["retained_row", "variable"]
    np.testing.assert_array_equal(trace_frame["rep_idx"].to_numpy(), [0, 0, 1, 1])
    assert list(trace_frame.columns) == [
        "rep_idx",
        "coef",
        "std_err",
        "coef_ci_low",
        "coef_ci_high",
        "t_stat",
        "pval",
        "partial_r2",
    ]
    np.testing.assert_allclose(
        trace_frame.loc[0, ["coef_ci_low", "coef_ci_high"]].to_numpy(),
        results[0].confidence_intervals(alpha=0.1),
    )

    f_test = out.F_test(alpha=0.1)
    assert f_test.test_name == "F-test"
    assert f_test.dist is ReferenceDistribution.F
    assert f_test.df == (np.float64(out.k), np.float64(out.n - out.k - 1))
    np.testing.assert_allclose(f_test.statistic_trace, out.F_stat_trace)
    assert f_test.status_trace == (TestStatus.OK, TestStatus.OK)

    spec = out.to_spec()
    assert spec.meta["variables"] == ["const", "trend"]
    assert MCRegressionResult.from_spec(spec).status_trace == (
        RegressionStatus.OK,
        RegressionStatus.OK,
    )


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
        ols(x, np.array([1.0, 2.1, 2.9, 4.2], dtype=np.float64), intercept=False),
        ols(x, np.array([0.8, 2.4, 3.2, 3.9], dtype=np.float64), intercept=False),
    )

    out = _mc_regression_from_ols(results)

    assert out.status_trace == (
        RegressionStatus.RANK_DEFICIENT,
        RegressionStatus.RANK_DEFICIENT,
    )
    np.testing.assert_allclose(
        out.se_trace,
        np.vstack([result.se for result in results]),
    )


def test_mc_regression_result_uses_declared_native_variables() -> None:
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    first = ols(
        x,
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        variables=["c", "x"],
        intercept=False,
    )
    second = ols(
        x,
        np.array([1.5, 2.5, 3.5], dtype=np.float64),
        variables=["const", "trend"],
        intercept=False,
    )

    out = _mc_regression_from_ols((first, second))
    assert out.variables == ["c", "x"]
    np.testing.assert_allclose(out.coef_trace[1], second.coefficients)


def _mc_regression_trace(n_retained: int = 64, k: int = 2) -> MCRegressionResult:
    rng = np.random.default_rng(7)
    coef = rng.normal([0.5, -0.2], [0.05, 0.03], size=(n_retained, k))
    se = np.abs(rng.normal(0.06, 0.005, size=(n_retained, k)))
    return MCRegressionResult(
        kind="ols",
        variables=["Intercept", "x1"],
        coef_trace=coef,
        ssr_trace=rng.uniform(1.0, 2.0, n_retained),
        sst_trace=rng.uniform(3.0, 4.0, n_retained),
        _se_trace=se,
        n_retained=n_retained,
        retained_reps=np.arange(n_retained, dtype=np.int_),
        n_rep=n_retained,
        n=50,
        k=k,
        _raw_status=np.zeros(n_retained, dtype=np.int_),
    )


def test_mc_regression_summary_is_one_row_per_variable() -> None:
    out = _mc_regression_trace()

    summary = out.summary(alpha=0.1)
    assert list(summary.index) == ["Intercept", "x1"]
    assert summary.index.name == "variable"
    assert list(summary.columns) == [
        "coef",
        "coef_se",
        "t_stat",
        "pval",
        "reject_rate",
    ]

    np.testing.assert_allclose(summary["coef"].to_numpy(), out.coef_trace.mean(axis=0))
    np.testing.assert_allclose(
        summary["coef_se"].to_numpy(),
        out.coef_trace.std(ddof=1, axis=0) / np.sqrt(out.n_retained),
    )
    np.testing.assert_allclose(
        summary["reject_rate"].to_numpy(), (out.pval_trace < 0.1).mean(axis=0)
    )


def test_mc_regression_intervals_bracket_the_summary_estimates() -> None:
    out = _mc_regression_trace()

    summary = out.summary()
    intervals = out.intervals()
    assert list(intervals.index.names) == ["variable", "quantity"]
    assert list(intervals.columns) == ["ci_low", "ci_high"]

    for variable in out.variables:
        for quantity in ("coef", "t_stat", "pval", "reject_rate"):
            low, high = intervals.loc[(variable, quantity)]
            assert low <= summary.loc[variable, quantity] <= high

    bounded = intervals.xs("pval", level="quantity")
    assert (bounded["ci_low"] >= 0.0).all()
    assert (intervals.xs("reject_rate", level="quantity")["ci_high"] <= 1.0).all()


def test_mc_regression_intervals_widen_with_confidence_level() -> None:
    out = _mc_regression_trace()

    narrow = out.intervals(confidence_level=0.90).loc[("Intercept", "coef")]
    wide = out.intervals(confidence_level=0.99).loc[("Intercept", "coef")]
    assert wide.ci_low < narrow.ci_low
    assert wide.ci_high > narrow.ci_high

    student = out.intervals(t_interval=True).loc[("Intercept", "coef")]
    normal = out.intervals(t_interval=False).loc[("Intercept", "coef")]
    assert student.ci_high > normal.ci_high

    assert not np.isclose(
        out.intervals(wilson=False).loc[("Intercept", "reject_rate"), "ci_low"],
        out.intervals(wilson=True).loc[("Intercept", "reject_rate"), "ci_low"],
    )


def test_mc_regression_mc_se_needs_two_replications() -> None:
    out = _mc_regression_trace(n_retained=1)
    assert np.isnan(out.coef_se).all()
    assert np.isnan(out.summary()["coef_se"].to_numpy()).all()
