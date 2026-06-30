from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.regression.enums import RegressionStatus
from SymbolicDSGE.regression.result import RegressionResult
from SymbolicDSGE.regression.ridge.core import (
    l2_grid_search,
    ridge,
    ridge_gs,
)
from SymbolicDSGE.regression.solvers import (
    chol_solve,
    chol_solve_L2,
    lstsq_solve,
    use_scalar_path,
    xtx_xty,
)
from SymbolicDSGE.regression.utils import (
    aic,
    bic,
    get_criterion,
    l2_loss,
    log_grid,
    process_args,
)


def test_shared_regression_objective_and_grid_helpers() -> None:
    np.testing.assert_allclose(
        log_grid.py_func(np.float64(0.1), np.float64(10.0), 1),
        [0.1],
    )
    np.testing.assert_allclose(
        log_grid.py_func(np.float64(0.1), np.float64(10.0), 2),
        [0.1, 10.0],
    )
    np.testing.assert_allclose(
        log_grid.py_func(np.float64(0.1), np.float64(10.0), 3),
        np.array([0.1, 1.0, 10.0], dtype=np.float64),
    )

    assert aic.py_func(np.float64(0.0), 5, np.float64(2.0)) == -np.inf
    assert bic.py_func(np.float64(0.0), 5, np.float64(2.0)) == -np.inf
    assert l2_loss.py_func(np.float64(3.0), 5, np.float64(2.0)) == np.float64(3.0)
    assert get_criterion("aic") is aic
    assert get_criterion("bic") is bic
    assert get_criterion("loss") is l2_loss
    with pytest.raises(ValueError, match="criterion"):
        get_criterion("bad")  # type: ignore[arg-type]


def test_process_args_normalizes_defaults_and_validates_shapes() -> None:
    x = np.arange(6.0, dtype=np.float64).reshape(3, 2)
    y = np.arange(3.0, dtype=np.float64)

    out_x, out_y, variables = process_args(x, y, None)

    assert out_x.flags.c_contiguous
    assert out_y.flags.c_contiguous
    assert variables == ["x0", "x1"]

    with pytest.raises(ValueError, match="2D"):
        process_args(np.arange(3.0), y, None)
    with pytest.raises(ValueError, match="1D"):
        process_args(x, y.reshape(3, 1), None)
    with pytest.raises(ValueError, match="row counts"):
        process_args(x, np.arange(2.0), None)
    with pytest.raises(ValueError, match="variables"):
        process_args(x, y, ["x0"])


def test_regression_result_validates_core_container_shapes() -> None:
    base = dict(
        variables=["x0", "x1"],
        coefficients=np.array([1.0, 0.0], dtype=np.float64),
        y=np.array([1.0, 2.0], dtype=np.float64),
        X=np.eye(2, dtype=np.float64),
        status=RegressionStatus.OK,
    )
    out = RegressionResult(**base)
    assert out.x is out.X
    assert out.to_dict()["variables"] == ["x0", "x1"]

    with pytest.raises(ValueError, match="response must be a 1D"):
        RegressionResult(**{**base, "y": np.ones((2, 1), dtype=np.float64)})
    with pytest.raises(ValueError, match="design matrix must be a 2D"):
        RegressionResult(**{**base, "X": np.ones(2, dtype=np.float64)})
    with pytest.raises(ValueError, match="coefficients must be a 1D"):
        RegressionResult(**{**base, "coefficients": np.ones((1, 2), dtype=np.float64)})
    with pytest.raises(ValueError, match="row counts"):
        RegressionResult(**{**base, "y": np.ones(3, dtype=np.float64)})
    with pytest.raises(ValueError, match="coefficient count"):
        RegressionResult(**{**base, "coefficients": np.ones(1, dtype=np.float64)})
    with pytest.raises(ValueError, match="variables"):
        RegressionResult(**{**base, "variables": ["x0"]})


def test_solver_helpers_cover_cholesky_lstsq_and_rank_deficient_paths() -> None:
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    G, g = xtx_xty.py_func(x, y)
    np.testing.assert_allclose(G, x.T @ x)
    np.testing.assert_allclose(g, x.T @ y)

    # p above SCALAR_PATH_MAX_P -> the BLAS branch of xtx_xty.
    wide_x = np.ones((1, 300), dtype=np.float64)
    wide_y = np.ones(1, dtype=np.float64)
    G_wide, g_wide = xtx_xty.py_func(wide_x, wide_y)
    np.testing.assert_allclose(G_wide, wide_x.T @ wide_x)
    np.testing.assert_allclose(g_wide, wide_x.T @ wide_y)

    coef, L, status = chol_solve.py_func(x, y)
    np.testing.assert_allclose(coef, np.array([1.0, 1.0], dtype=np.float64))
    assert L.shape == (2, 2)
    assert status == int(RegressionStatus.OK)

    coef_l2, L_l2, dof, status_l2 = chol_solve_L2.py_func(
        x,
        y,
        np.float64(0.5),
        True,
    )
    assert status_l2 == int(RegressionStatus.OK)
    assert L_l2.shape == (2, 2)
    assert np.isfinite(dof)
    assert np.isfinite(coef_l2).all()

    singular = np.ones((3, 2), dtype=np.float64)
    coef_bad, L_bad, dof_bad, status_bad = chol_solve_L2.py_func(
        singular,
        y,
        np.float64(0.0),
        False,
    )
    assert status_bad == int(RegressionStatus.RANK_DEFICIENT)
    assert np.isnan(coef_bad).all()
    assert L_bad.shape == (0, 0)
    assert np.isnan(dof_bad)

    coef_ls, L_ls, status_ls = lstsq_solve.py_func(singular, y)
    assert status_ls == int(RegressionStatus.RANK_DEFICIENT)
    assert L_ls.shape == (0, 0)
    assert coef_ls.shape == (2,)


def test_ridge_grid_search_and_validation_branches() -> None:
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert use_scalar_path.py_func(10, 10)
    assert not use_scalar_path.py_func(200_000, 10)  # n*p = 2e6 > 1e6
    assert not use_scalar_path.py_func(1, 2_000)  # p = 2000 > 256

    alpha, coef, obj, status = l2_grid_search.py_func(
        np.ones((3, 2), dtype=np.float64),
        y,
        np.float64(0.0),
        np.float64(0.0),
        1,
        l2_loss,
        False,
    )
    assert alpha == np.float64(0.0)
    assert status == int(RegressionStatus.RANK_DEFICIENT)
    assert np.isinf(obj)
    assert np.isnan(coef).all()

    with pytest.raises(ValueError, match="alpha"):
        ridge(x, y, alpha=np.float64(-0.1))
    with pytest.raises(ValueError, match="start and stop"):
        ridge_gs(x, y, start=0.0, stop=1.0, num=2)
    with pytest.raises(ValueError, match="num"):
        ridge_gs(x, y, start=0.1, stop=1.0, num=0)
    with pytest.raises(ValueError, match="criterion"):
        ridge_gs(x, y, start=0.1, stop=1.0, num=2, criterion="bad")  # type: ignore[arg-type]
