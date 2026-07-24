"""Coverage for estimator module-level helpers and small edge branches."""

from __future__ import annotations

import numpy as np
import pytest
from numpy import float64

from SymbolicDSGE.estimation import backend
from SymbolicDSGE.estimation.estimator import (
    Estimator,
    _method_from_result,
    _method_kwargs_from_result,
)
from SymbolicDSGE.estimation.results import MCMCResult, MLEResult


def _opt_result(**cfg) -> MLEResult:
    return MLEResult(
        x=np.zeros(1, dtype=float64),
        theta={"a": float64(1.0)},
        success=True,
        message="",
        fun=float64(0.0),
        nfev=1,
        nit=1,
        optimizer_config=dict(cfg),
        loglik=float64(0.0),
    )


def _mcmc_result() -> MCMCResult:
    return MCMCResult(
        param_names=["a"],
        samples=np.zeros((2, 1), dtype=float64),
        logpost_trace=np.zeros(2, dtype=float64),
        accept_rate=float64(0.5),
        n_draws=2,
        burn_in=1,
        thin=1,
        sampler_config={"adapt": True},
    )


def test_matrix_name_for_reserved_key_rejects_unknown():
    assert Estimator._matrix_name_for_reserved_key("R_corr") == "R"
    assert Estimator._matrix_name_for_reserved_key("Q_corr") == "Q"
    with pytest.raises(KeyError, match="Unknown reserved matrix key"):
        Estimator._matrix_name_for_reserved_key("not_a_key")


def test_method_from_result_branches():
    assert _method_from_result(None) is None
    assert _method_from_result(_mcmc_result()) == "mcmc"
    assert _method_from_result(_opt_result()) == "mle"
    with pytest.raises(TypeError, match="Unsupported estimation result type"):
        _method_from_result(object())  # type: ignore[arg-type]


def test_method_kwargs_from_result_branches():
    mc = _method_kwargs_from_result(_mcmc_result())
    assert mc["n_draws"] == 2 and mc["burn_in"] == 1 and mc["adapt"] is True
    opt = _method_kwargs_from_result(
        _opt_result(method="L-BFGS-B", options={"maxiter": 5})
    )
    assert opt == {"method": "L-BFGS-B", "options": {"maxiter": 5}}
