"""End-to-end oracle for ``Estimator.mle`` / ``map`` on a real model.

Locks the current behavior so the native-driver rewrite is immediately
parity-testable. Two guards per case:

* ``fun`` is pinned to a golden value, so the optimizer basin can't drift.
* the packed fields (``loglik`` / ``logpost`` / ``logprior`` / ``theta``) are
  cross-checked against the Python evaluators at the result ``x``, so a
  theta-naming or value error surfaces regardless of which optimizer produced
  ``x``. Those evaluators are the same reference the native path must reproduce.

The golden ``fun`` values were recorded from the scipy path; the native driver
must land within tolerance. ``x`` is not pinned tightly (native FD lands in the
same basin at a slightly different point).
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.estimation import Estimator
from SymbolicDSGE.estimation.results import MLEResult, MAPResult

# The ``post82`` fixture lives in tests/estimation/conftest.py.

_TH0 = np.array([2.0, 0.8], dtype=np.float64)
_BNDS = [(1.0, 5.0), (0.0, 0.99)]


def _mle_estimator(post82, mode: str) -> Estimator:
    return Estimator(
        solver=post82["solver"],
        compiled=post82["compiled"],
        y=post82["y"],
        observables=post82["obs"],
        filter_mode=mode,
        estimated_params=["psi_pi", "rho_r"],
        ss_seed=post82["steady"],
    )


def _assert_theta_named(est: Estimator, res) -> None:
    expected = est.theta_to_params(res.x)
    assert res.theta.keys() == expected.keys()
    for name, value in expected.items():
        assert float(res.theta[name]) == pytest.approx(float(value), abs=1e-9)


def _assert_mle_packing(est: Estimator, res: MLEResult) -> None:
    assert isinstance(res, MLEResult)
    assert res.loglik == pytest.approx(float(est.loglik(res.x)), abs=1e-6)
    assert res.fun == pytest.approx(-res.loglik, abs=1e-9)
    _assert_theta_named(est, res)


@pytest.mark.parametrize(
    "mode,golden_fun",
    [
        ("linear", 257.2567583860),
        ("extended", 257.2567583860),
        ("unscented", 257.9468468429),
    ],
)
def test_mle_lbfgsb_oracle(post82, mode, golden_fun):
    est = _mle_estimator(post82, mode)
    res = est.mle(theta0=_TH0, bounds=_BNDS)
    assert res.fun == pytest.approx(golden_fun, abs=1e-3)
    _assert_mle_packing(est, res)


def test_mle_nelder_mead_oracle(post82):
    est = _mle_estimator(post82, "linear")
    res = est.mle(theta0=_TH0, bounds=_BNDS, method="Nelder-Mead")
    assert res.fun == pytest.approx(257.2567583857, abs=1e-3)
    _assert_mle_packing(est, res)


def test_map_lbfgsb_oracle(post82):
    prior = Estimator.make_prior(
        distribution="normal",
        parameters={"mean": 2.0, "std": 0.5},
        transform="identity",
    )
    est = Estimator(
        solver=post82["solver"],
        compiled=post82["compiled"],
        y=post82["y"],
        observables=post82["obs"],
        filter_mode="linear",
        estimated_params=["psi_pi"],
        priors={"psi_pi": prior},
        ss_seed=post82["steady"],
    )
    res = est.map(theta0=np.array([2.0], dtype=np.float64), bounds=[(1.0, 5.0)])
    assert isinstance(res, MAPResult)
    assert res.fun == pytest.approx(257.5019149357, abs=1e-3)

    lp = float(est.loglik(res.x))
    lpr = float(est.logprior(res.x))
    assert res.logprior == pytest.approx(lpr, abs=1e-6)
    assert res.logpost == pytest.approx(lp + lpr, abs=1e-6)
    assert res.fun == pytest.approx(-res.logpost, abs=1e-9)
    _assert_theta_named(est, res)
