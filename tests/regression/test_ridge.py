from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.regression import RegressionResult
from SymbolicDSGE.regression.utils import l2_loss
from SymbolicDSGE.regression.ridge import (
    RidgeObjective,
    RidgeResult,
    l2_grid_search,
    ridge,
    ridge_gs,
)


def test_ridge_returns_result_with_shared_regression_diagnostics() -> None:
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
    alpha = np.float64(0.5)

    out = ridge(x, y, alpha=alpha, variables=["trend"])

    X = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    G = (X.T @ X) / X.shape[0]
    g = (X.T @ y) / X.shape[0]
    penalty = np.diag([0.0, alpha])
    expected_coef = np.linalg.solve(G + penalty, g)
    expected_y_hat = X @ expected_coef

    assert isinstance(out, RidgeResult)
    assert isinstance(out, RegressionResult)
    assert out.variables == ["Intercept", "trend"]
    assert out.alpha == alpha
    np.testing.assert_allclose(out.coefficients, expected_coef)
    np.testing.assert_allclose(out.y_hat, expected_y_hat)
    np.testing.assert_allclose(out.residuals, y - expected_y_hat)
    assert out.ssr == pytest.approx(((y - expected_y_hat) ** 2).sum())
    assert out.mse == pytest.approx(out.ssr / out.n)
    assert out.rmse == pytest.approx(np.sqrt(out.mse))
    assert out.effective_dof == pytest.approx(np.trace(np.linalg.solve(G + penalty, G)))
    assert out.l2_penalty == pytest.approx(
        0.5 * alpha * np.dot(expected_coef[1:], expected_coef[1:])
    )


def test_l2_grid_search_selects_alpha_with_lowest_objective() -> None:
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
    start = np.float64(0.01)
    stop = np.float64(10.0)
    num = 6

    alpha, coef, obj, status = l2_grid_search(x, y, start, stop, num, l2_loss, False)

    alphas = np.exp(np.linspace(np.log(start), np.log(stop), num=num))
    G = (x.T @ x) / x.shape[0]
    g = (x.T @ y) / x.shape[0]
    expected_values = []
    expected_coefs = []
    for candidate in alphas:
        candidate_coef = np.linalg.solve(
            G + candidate * np.eye(x.shape[1]),
            g,
        )
        expected_coefs.append(candidate_coef)
        expected_values.append(((y - x @ candidate_coef) ** 2).sum())
    expected_idx = int(np.argmin(expected_values))

    assert status == 0
    assert alpha == np.float64(alphas[expected_idx])
    assert obj == pytest.approx(expected_values[expected_idx])
    np.testing.assert_allclose(coef, expected_coefs[expected_idx])


def test_ridge_grid_search_keeps_intercept_unpenalized() -> None:
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
    start = np.float64(0.01)
    stop = np.float64(1.0)
    num = 5

    out = ridge_gs(
        x,
        y,
        start=start,
        stop=stop,
        num=num,
        criterion="loss",
        variables=["trend"],
    )

    X = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    G = (X.T @ X) / X.shape[0]
    g = (X.T @ y) / X.shape[0]
    alphas = np.exp(np.linspace(np.log(start), np.log(stop), num=num))
    expected_values = []
    expected_coefs = []
    expected_dofs = []
    for candidate in alphas:
        penalty = np.diag([0.0, candidate])
        penalized = G + penalty
        candidate_coef = np.linalg.solve(penalized, g)
        expected_coefs.append(candidate_coef)
        expected_values.append(((y - X @ candidate_coef) ** 2).sum())
        expected_dofs.append(np.trace(np.linalg.solve(penalized, G)))
    expected_idx = int(np.argmin(expected_values))
    expected_coef = expected_coefs[expected_idx]

    assert out.variables == ["Intercept", "trend"]
    assert out.objective is RidgeObjective.LOSS
    assert out.alpha == np.float64(alphas[expected_idx])
    assert out.objective_value == pytest.approx(expected_values[expected_idx])
    assert out.effective_dof == pytest.approx(expected_dofs[expected_idx])
    np.testing.assert_allclose(out.coefficients, expected_coef)
    assert out.l2_penalty == pytest.approx(
        0.5 * out.alpha * np.dot(expected_coef[1:], expected_coef[1:])
    )
