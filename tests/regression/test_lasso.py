from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.regression import RegressionResult
from SymbolicDSGE.regression.lasso import (
    LassoResult,
    lars_lasso_gram,
    lasso,
    lasso_gs,
    lasso_path_eval,
)


def test_lasso_result_validates_and_exposes_l1_diagnostics() -> None:
    x = np.eye(3, dtype=np.float64)
    y = np.array([3.0, -1.0, 0.25], dtype=np.float64)
    out = lasso(x, y, alpha=np.float64(0.5), intercept=False)

    assert isinstance(out, LassoResult)
    assert isinstance(out, RegressionResult)
    assert out.variables == ["x0", "x1", "x2"]
    np.testing.assert_allclose(
        out.coefficients,
        np.array([1.5, 0.0, 0.0], dtype=np.float64),
    )
    assert out.n_active == 1
    assert out.selected_variables == ["x0"]
    assert out.l1_norm == pytest.approx(1.5)
    assert out.l1_penalty == pytest.approx(0.75)
    assert out.effective_dof == pytest.approx(1.0)


def test_lasso_keeps_intercept_unpenalized() -> None:
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    out = lasso(x, y, alpha=np.float64(100.0), variables=["trend"])

    assert out.variables == ["Intercept", "trend"]
    np.testing.assert_allclose(
        out.X,
        np.column_stack([np.ones(x.shape[0], dtype=np.float64), x]),
    )
    np.testing.assert_allclose(out.coefficients, np.array([2.0, 0.0]))
    assert out.selected_variables == []
    assert out.l1_penalty == pytest.approx(0.0)
    assert out.effective_dof == pytest.approx(1.0)


def test_lars_lasso_gram_matches_hardcoded_sklearn_path() -> None:
    G = np.eye(2, dtype=np.float64) / 2.0
    c = np.array([1.5, -1.0], dtype=np.float64)

    lam_path, beta_path, status = lars_lasso_gram(G, c)

    assert status == 0
    np.testing.assert_allclose(
        lam_path,
        np.array([1.5, 1.0, 0.0], dtype=np.float64),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        beta_path,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [3.0, -2.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        lasso_path_eval(
            lam_path,
            beta_path,
            np.array([1.5, 0.5, 0.0], dtype=np.float64),
        ),
        np.array(
            [
                [0.0, 0.0],
                [2.0, -1.0],
                [3.0, -2.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-12,
    )


def test_lars_lasso_gram_matches_hardcoded_dense_path() -> None:
    x = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.5, 1.0, -1.0],
            [2.0, -0.5, 0.0],
            [-1.0, 1.5, 1.0],
            [0.0, -2.0, 0.5],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, -0.5, 2.0, -1.0, 0.75], dtype=np.float64)
    G = (x.T @ x) / x.shape[0]
    c = (x.T @ y) / x.shape[0]

    lam_path, beta_path, status = lars_lasso_gram(G, c)

    assert status == 0
    np.testing.assert_allclose(
        lam_path,
        np.array([1.15, 0.7823529412, 0.2895559211, 0.0], dtype=np.float64),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        beta_path,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2941176471, 0.0, 0.0],
                [0.6101973684, -0.2442434211, 0.0],
                [0.7826291632, -0.3774861716, 0.2071907732],
            ],
            dtype=np.float64,
        ),
        atol=1e-10,
    )


def test_lasso_grid_search_uses_lars_path_grid() -> None:
    x = np.eye(2, dtype=np.float64)
    y = np.array([3.0, -2.0], dtype=np.float64)

    out = lasso_gs(
        x,
        y,
        start=np.float64(0.5),
        stop=np.float64(4.0),
        num=4,
        intercept=False,
    )

    assert out.alpha == np.float64(0.5)
    np.testing.assert_allclose(out.coefficients, np.array([2.0, -1.0]))
    assert out.alpha_grid is not None
    assert out.coefficient_path is not None
    assert out.objective_trace is not None
    assert out.knot_lambdas is not None
    assert out.knot_coefficients is not None
    assert out.alpha_grid.shape == (4,)
    assert out.coefficient_path.shape == (4, 2)
    assert out.objective_trace.shape == (4,)
    np.testing.assert_allclose(
        out.knot_lambdas,
        np.array([1.5, 1.0, 0.0], dtype=np.float64),
        atol=1e-12,
    )
    np.testing.assert_allclose(out.coefficient_path[0], out.coefficients)


def test_lasso_result_validates_grid_shapes() -> None:
    x = np.eye(2, dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="coefficient_path"):
        LassoResult(
            variables=["x0", "x1"],
            coefficients=np.array([1.0, 0.0], dtype=np.float64),
            y=y,
            X=x,
            status=0,
            alpha=np.float64(0.1),
            effective_dof=np.float64(1.0),
            alpha_grid=np.array([0.1, 1.0], dtype=np.float64),
            coefficient_path=np.ones((3, 2), dtype=np.float64),
        )
