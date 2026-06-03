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
from SymbolicDSGE.regression.lasso.core import (
    NON_CONVERGENT,
    OK,
    lasso_gram_cd,
    smooth_threshold,
    solve_small,
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

    base = dict(
        variables=["x0", "x1"],
        coefficients=np.array([1.0, 0.0], dtype=np.float64),
        y=y,
        X=x,
        status=0,
        alpha=np.float64(0.1),
        effective_dof=np.float64(1.0),
        intercept=False,
    )
    with pytest.raises(ValueError, match="objective_trace"):
        LassoResult(
            **base,
            alpha_grid=np.array([0.1, 1.0], dtype=np.float64),
            objective_trace=np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="alpha_grid is required"):
        LassoResult(
            **base,
            objective_trace=np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="alpha_grid"):
        LassoResult(**base, alpha_grid=np.ones((1, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="coefficient_path"):
        LassoResult(
            **base,
            alpha_grid=np.array([0.1], dtype=np.float64),
            coefficient_path=np.ones(2, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="knot_coefficients is required"):
        LassoResult(
            **base,
            knot_lambdas=np.array([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="knot_lambdas is required"):
        LassoResult(
            **base,
            knot_coefficients=np.ones((1, 2), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="knot_coefficients"):
        LassoResult(
            **base,
            knot_lambdas=np.array([1.0], dtype=np.float64),
            knot_coefficients=np.ones((2, 2), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="knot_lambdas"):
        LassoResult(**base, knot_lambdas=np.ones((1, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="knot_coefficients"):
        LassoResult(
            **base,
            knot_lambdas=np.array([1.0], dtype=np.float64),
            knot_coefficients=np.ones(2, dtype=np.float64),
        )


def test_lasso_low_level_coordinate_descent_branches() -> None:
    assert smooth_threshold.py_func(np.float64(2.0), np.float64(0.5)) == np.float64(1.5)
    assert smooth_threshold.py_func(np.float64(-2.0), np.float64(0.5)) == np.float64(
        -1.5
    )
    assert smooth_threshold.py_func(np.float64(0.25), np.float64(0.5)) == np.float64(
        0.0
    )

    coef, status = lasso_gram_cd.py_func(
        np.array([[0.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.float64(0.1),
    )
    assert status == OK
    np.testing.assert_allclose(coef, np.zeros(1, dtype=np.float64))

    coef, status = lasso_gram_cd.py_func(
        np.eye(1, dtype=np.float64),
        np.array([10.0], dtype=np.float64),
        np.float64(0.0),
        max_iter=1,
        tol=np.float64(1e-14),
    )
    assert status == NON_CONVERGENT
    np.testing.assert_allclose(coef, np.array([10.0], dtype=np.float64))


def test_lasso_low_level_lars_and_path_edge_cases() -> None:
    A = np.array([[0.0, 2.0], [1.0, 1.0]], dtype=np.float64)
    b = np.array([4.0, 3.0], dtype=np.float64)
    np.testing.assert_allclose(solve_small.py_func(A, b), np.linalg.solve(A, b))

    lam_path, beta_path, status = lars_lasso_gram.py_func(
        np.eye(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
    )
    assert status == OK
    np.testing.assert_allclose(lam_path, np.zeros(1, dtype=np.float64))
    np.testing.assert_allclose(beta_path, np.zeros((1, 2), dtype=np.float64))

    lam_path, beta_path, status = lars_lasso_gram.py_func(
        -np.eye(1, dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    )
    assert status == NON_CONVERGENT
    np.testing.assert_allclose(lam_path, np.array([1.0], dtype=np.float64))
    np.testing.assert_allclose(beta_path, np.zeros((1, 1), dtype=np.float64))

    out = lasso_path_eval.py_func(
        np.array([2.0, 1.0, 0.0], dtype=np.float64),
        np.array([[0.0], [1.0], [3.0]], dtype=np.float64),
        np.array([3.0, -1.0], dtype=np.float64),
    )
    np.testing.assert_allclose(out, np.array([[0.0], [3.0]], dtype=np.float64))


def test_lasso_validates_public_wrapper_inputs() -> None:
    x = np.eye(2, dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="alpha"):
        lasso(x, y, alpha=np.float64(-0.1))
    with pytest.raises(ValueError, match="start and stop"):
        lasso_gs(x, y, start=0.0, stop=1.0, num=2)
    with pytest.raises(ValueError, match="num"):
        lasso_gs(x, y, start=0.1, stop=1.0, num=0)
