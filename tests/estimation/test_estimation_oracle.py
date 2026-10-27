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

They score a panel this module simulates, so they are re-recorded whenever the
model underneath moves: once when natural time dating changed the state space
the panel comes out of, and again when the filters took Dynare's initialization,
which redates what ``P0`` is the covariance of.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.estimation import Estimator, make_prior
from SymbolicDSGE.estimation.results import MLEResult, MAPResult

# The POST82 model is the session-wide bundle; ``y_post82`` lives in
# tests/estimation/conftest.py.

_TH0 = {"psi_pi": 2.0, "rho_r": 0.8}
_BNDS = {"psi_pi": (1.0, 5.0), "rho_r": (0.0, 0.99)}


def _mle_estimator(solved, y, mode: str) -> Estimator:
    P0 = None
    compiled = solved.compiled
    config = solved.config
    if mode == "unscented":
        P0 = np.eye(len(compiled.var_names), dtype=np.float64)
    return Estimator(
        config=config,
        compiled=compiled,
        y=y,
        observables=compiled.observable_names,
        filter_mode=mode,
        estimated_params=["psi_pi", "rho_r"],
        ss_seed=solved.policy.steady_state,
        P0=P0,
    )


def _assert_theta_named(est: Estimator, res) -> None:
    assert list(res.theta) == list(est.param_names)
    assert all(np.isfinite(float(res.theta[name])) for name in est.param_names)


def _assert_mle_packing(est: Estimator, res: MLEResult) -> None:
    assert isinstance(res, MLEResult)
    assert res.loglik == pytest.approx(float(est.loglik(res.theta)), abs=1e-6)
    assert res.fun == pytest.approx(-res.loglik, abs=1e-9)
    _assert_theta_named(est, res)


@pytest.mark.parametrize(
    "mode,golden_fun",
    [
        ("linear", 203.5359358702),
        ("extended", 203.5359358702),
        ("unscented", 205.1999554164),
    ],
)
def test_mle_lbfgsb_oracle(solved_post82, y_post82, mode, golden_fun):
    est = _mle_estimator(solved_post82, y_post82, mode)
    res = est.mle(theta0=_TH0, bounds=_BNDS)
    assert res.fun == pytest.approx(golden_fun, abs=1e-3)
    _assert_mle_packing(est, res)


def test_mle_nelder_mead_oracle(solved_post82, y_post82):
    est = _mle_estimator(solved_post82, y_post82, "linear")
    res = est.mle(theta0=_TH0, bounds=_BNDS, method="Nelder-Mead")
    assert res.fun == pytest.approx(203.5359358684, abs=1e-3)
    _assert_mle_packing(est, res)


def test_map_lbfgsb_oracle(solved_post82, y_post82):
    prior = make_prior(
        distribution="normal",
        parameters={"mean": 2.0, "std": 0.5},
        transform="identity",
    )
    config = solved_post82.config
    compiled = solved_post82.compiled
    est = Estimator(
        config=config,
        compiled=compiled,
        y=y_post82,
        observables=compiled.observable_names,
        filter_mode="linear",
        estimated_params={"psi_pi": prior},
        ss_seed=solved_post82.policy.steady_state,
    )
    res = est.map(theta0={"psi_pi": 2.0}, bounds={"psi_pi": (1.0, 5.0)})
    assert isinstance(res, MAPResult)
    assert res.fun == pytest.approx(204.3426449357, abs=1e-3)

    lp = float(est.loglik(res.theta))
    lpr = float(est.logprior(res.theta))
    assert res.logprior == pytest.approx(lpr, abs=1e-6)
    assert res.logpost == pytest.approx(lp + lpr, abs=1e-6)
    assert res.fun == pytest.approx(-res.logpost, abs=1e-9)
    _assert_theta_named(est, res)
