from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.regression import RegressionResult
from SymbolicDSGE.regression.elastic_net import (
    ElasticNetResult,
    elastic_net,
    elastic_net_gs,
)
from SymbolicDSGE.regression.elastic_net.core import (
    elastic_net_active_dof,
    elastic_net_gram_cd,
    elastic_net_gram_cd_path,
    split_penalty,
)
from SymbolicDSGE.regression.enums import RegressionStatus


def test_elastic_net_returns_result_with_l1_and_l2_diagnostics() -> None:
    x = np.eye(3, dtype=np.float64)
    y = np.array([3.0, -1.0, 0.25], dtype=np.float64)

    out = elastic_net(
        x,
        y,
        alpha=np.float64(0.5),
        l1_ratio=np.float64(0.5),
        intercept=False,
    )

    expected = np.array([9.0 / 7.0, -1.0 / 7.0, 0.0], dtype=np.float64)
    assert isinstance(out, ElasticNetResult)
    assert isinstance(out, RegressionResult)
    assert out.status is RegressionStatus.OK
    assert out.variables == ["x0", "x1", "x2"]
    np.testing.assert_allclose(out.coefficients, expected)
    assert out.n_active == 2
    assert out.selected_variables == ["x0", "x1"]
    assert out.l1_norm == pytest.approx(10.0 / 7.0)
    assert out.l2_norm_sq == pytest.approx(np.dot(expected, expected))
    assert out.l1_penalty == pytest.approx(0.5 * 0.5 * (10.0 / 7.0))
    assert out.l2_penalty == pytest.approx(0.5 * 0.5 * 0.5 * np.dot(expected, expected))
    assert out.penalty == pytest.approx(out.l1_penalty + out.l2_penalty)
    assert out.effective_dof == pytest.approx(8.0 / 7.0)


def test_elastic_net_keeps_intercept_unpenalized() -> None:
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    out = elastic_net(
        x,
        y,
        alpha=np.float64(100.0),
        l1_ratio=np.float64(0.5),
        variables=["trend"],
    )

    assert out.variables == ["Intercept", "trend"]
    np.testing.assert_allclose(out.coefficients, np.array([2.0, 0.0]))
    assert out.selected_variables == []
    assert out.l1_penalty == pytest.approx(0.0)
    assert out.l2_penalty == pytest.approx(0.0)
    assert out.effective_dof == pytest.approx(1.0)


def test_elastic_net_grid_search_records_grid_traces() -> None:
    x = np.eye(2, dtype=np.float64)
    y = np.array([3.0, -1.0], dtype=np.float64)

    out = elastic_net_gs(
        x,
        y,
        start=np.float64(0.5),
        stop=np.float64(2.0),
        num=3,
        l1_ratio=np.float64(0.5),
        criterion="bic",
        intercept=False,
    )

    assert out.status is RegressionStatus.OK
    assert out.alpha_grid is not None
    assert out.coefficient_path is not None
    assert out.objective_trace is not None
    assert out.rss_trace is not None
    assert out.effective_dof_trace is not None
    assert out.status_trace is not None
    assert out.alpha_grid.shape == (3,)
    assert out.coefficient_path.shape == (3, 2)
    assert out.objective_trace.shape == (3,)
    assert out.rss_trace.shape == (3,)
    assert out.effective_dof_trace.shape == (3,)
    assert out.status_trace.shape == (3,)
    assert out.alpha in set(out.alpha_grid)
    np.testing.assert_allclose(
        out.coefficients,
        out.coefficient_path[int(np.argmin(out.objective_trace))],
    )


def test_elastic_net_shortcuts_wrap_ridge_and_lasso_results() -> None:
    x = np.eye(3, dtype=np.float64)
    y = np.array([3.0, -1.0, 0.25], dtype=np.float64)

    ridge_like = elastic_net(
        x,
        y,
        alpha=np.float64(0.5),
        l1_ratio=np.float64(0.0),
        intercept=False,
    )
    lasso_like = elastic_net(
        x,
        y,
        alpha=np.float64(0.5),
        l1_ratio=np.float64(1.0),
        intercept=False,
    )
    ridge_grid_like = elastic_net_gs(
        x,
        y,
        start=np.float64(0.1),
        stop=np.float64(1.0),
        num=2,
        l1_ratio=np.float64(0.0),
        intercept=False,
    )
    lasso_grid_like = elastic_net_gs(
        x,
        y,
        start=np.float64(0.5),
        stop=np.float64(1.0),
        num=2,
        l1_ratio=np.float64(1.0),
        criterion="loss",
        intercept=False,
    )

    assert ridge_like.l1_ratio == np.float64(0.0)
    assert ridge_like.l1_penalty == np.float64(0.0)
    assert lasso_like.l1_ratio == np.float64(1.0)
    assert lasso_like.l2_penalty == np.float64(0.0)
    assert ridge_grid_like.alpha_grid is None
    assert lasso_grid_like.alpha_grid is not None
    assert lasso_grid_like.coefficient_path is not None


def test_elastic_net_low_level_helpers_cover_path_order_and_statuses() -> None:
    G = np.eye(2, dtype=np.float64) / 2.0
    g = np.array([1.5, -1.0], dtype=np.float64)
    alpha_l1, alpha_l2 = split_penalty.py_func(np.float64(0.5), np.float64(0.25))

    beta, status = elastic_net_gram_cd.py_func(
        G,
        g,
        alpha_l1,
        alpha_l2,
        np.zeros(2, dtype=np.float64),
    )
    assert status == int(RegressionStatus.OK)
    np.testing.assert_allclose(
        beta,
        np.array([11.0 / 7.0, -1.0], dtype=np.float64),
    )

    beta_nc, status_nc = elastic_net_gram_cd.py_func(
        G,
        g,
        alpha_l1,
        alpha_l2,
        np.zeros(2, dtype=np.float64),
        max_iter=1,
        tol=np.float64(1e-14),
    )
    assert status_nc == int(RegressionStatus.NON_CONVERGENT)
    assert np.isfinite(beta_nc).all()

    alpha_grid = np.array([0.25, 1.0], dtype=np.float64)
    coef_path, status_path = elastic_net_gram_cd_path.py_func(
        G,
        g,
        alpha_grid,
        np.float64(0.5),
    )
    assert coef_path.shape == (2, 2)
    np.testing.assert_array_equal(status_path, np.zeros(2, dtype=np.int64))

    assert elastic_net_active_dof.py_func(
        G,
        np.zeros(2, dtype=np.float64),
        np.float64(0.25),
        False,
    ) == np.float64(0.0)


def test_elastic_net_validates_inputs_and_result_grid_shapes() -> None:
    x = np.eye(2, dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="alpha"):
        elastic_net(x, y, alpha=np.float64(-1.0), l1_ratio=np.float64(0.5))
    with pytest.raises(ValueError, match="l1_ratio"):
        elastic_net(x, y, alpha=np.float64(1.0), l1_ratio=np.float64(1.5))
    with pytest.raises(ValueError, match="max_iter"):
        elastic_net(
            x,
            y,
            alpha=np.float64(1.0),
            l1_ratio=np.float64(0.5),
            max_iter=0,
        )
    with pytest.raises(ValueError, match="tol"):
        elastic_net(
            x,
            y,
            alpha=np.float64(1.0),
            l1_ratio=np.float64(0.5),
            tol=np.float64(0.0),
        )
    with pytest.raises(ValueError, match="start and stop"):
        elastic_net_gs(x, y, -1.0, 1.0, 2, l1_ratio=np.float64(0.5))
    with pytest.raises(ValueError, match="num"):
        elastic_net_gs(x, y, 0.1, 1.0, 0, l1_ratio=np.float64(0.5))
    with pytest.raises(ValueError, match="criterion"):
        elastic_net_gs(
            x,
            y,
            0.1,
            1.0,
            2,
            l1_ratio=np.float64(0.5),
            criterion="bad",  # type: ignore[arg-type]
        )

    base = dict(
        variables=["x0", "x1"],
        coefficients=np.array([1.0, 0.0], dtype=np.float64),
        y=y,
        X=x,
        status=RegressionStatus.OK,
        alpha=np.float64(0.1),
        l1_ratio=np.float64(0.5),
        effective_dof=np.float64(1.0),
        intercept=False,
    )

    with pytest.raises(ValueError, match="l1_ratio"):
        ElasticNetResult(**{**base, "l1_ratio": np.float64(-0.1)})
    with pytest.raises(ValueError, match="alpha_grid is required"):
        ElasticNetResult(
            **base,
            objective_trace=np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="coefficient_path"):
        ElasticNetResult(
            **base,
            alpha_grid=np.array([0.1, 1.0], dtype=np.float64),
            coefficient_path=np.ones((3, 2), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="objective_trace"):
        ElasticNetResult(
            **base,
            alpha_grid=np.array([0.1, 1.0], dtype=np.float64),
            objective_trace=np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="alpha_grid"):
        ElasticNetResult(
            **base,
            alpha_grid=np.ones((1, 1), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="coefficient_path"):
        ElasticNetResult(
            **base,
            alpha_grid=np.array([0.1], dtype=np.float64),
            coefficient_path=np.ones(2, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="status_trace"):
        ElasticNetResult(
            **base,
            alpha_grid=np.array([0.1], dtype=np.float64),
            status_trace=np.ones((1, 1), dtype=np.int64),
        )
