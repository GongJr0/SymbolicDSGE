"""Exact parity against Dynare's first-order solution of POST82.

Every assertion here is ``ours == dynare`` on raw arrays. Nothing is re-dated,
rescaled, reordered, or sliced on either side before comparison. If a test needs
a transform to pass, that transform is the finding, not the fix.

``tests/fixtures/models/POST82.yaml`` and
``tests/fixtures/models/post82_first_order.mod`` are the same model written
twice, both at Dynare's dating, so the two solutions are directly comparable
object by object. Goldens are transcribed in
``tests/_oracles/dynare_post82_first_order``.

The internals are compared, not only the outputs, so a state space that is
individually wrong but happens to cancel in a simulation cannot pass:

* ``var_names`` and the exogenous block order against Dynare's declaration order
* ``A[:, states]`` against ``ghx`` and ``B`` against ``ghu``
* the exogenous impact block and the control loadings separately, which is what
  localises the failure
* simulated paths, IRFs, measurement matrices and the Kalman loglik

Known red, tracked by https://github.com/GongJr0/SymbolicDSGE/issues/390:
``sdsge_assemble_state_space`` writes the exogenous block of ``B`` as the
identity. In this model ``e_r`` enters a Taylor rule carrying ``Pi(t)`` and
``x(t)``, so the impact of ``e_r`` on ``r`` is a fixed point, ``0.3336``, and
``e_g`` moves ``r`` on impact by ``0.3470`` through the same rule. Everything
downstream of ``B`` fails with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.estimation import backend
from SymbolicDSGE.kalman.filter import KalmanFilter

from _oracles import dynare_post82_first_order as dyn

YAML = "tests/fixtures/models/POST82.yaml"

# Dynare reports the solution to 1e-16; these compare against transcribed
# doubles, so the only slack needed is accumulated floating point.
TOL = 1e-10


@pytest.fixture(scope="module")
def solved():
    model, kalman = ModelParser(YAML).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    return compiled, solver.solve(compiled, order=1)


@pytest.fixture(scope="module")
def matrices(solved):
    compiled, sol = solved
    return np.real(sol.A), np.real(sol.B), np.real(sol.policy.f)


def _x0(compiled):
    x0 = np.zeros(len(compiled.var_names))
    x0[: compiled.n_exog] = dyn.X0_STATES
    return x0


def _as_array(compiled, out):
    return np.column_stack([out[name] for name in compiled.var_names])


# --- ordering ---------------------------------------------------------------
# Nothing below means anything if the columns are not the same columns.


def test_variable_order_matches_dynare_declaration_order(solved):
    compiled, _ = solved
    assert tuple(compiled.var_names) == dyn.DECL_COLUMNS


def test_exogenous_block_order_matches_dynare(solved):
    compiled, _ = solved
    targets = tuple(
        str(compiled.config.shock_map[sym]) for sym in compiled.config.shock_map
    )
    assert targets == tuple(name[2:] for name in dyn.EXO_COLUMNS)
    assert tuple(compiled.var_names[: compiled.n_exog]) == targets


# --- the solution itself ----------------------------------------------------


def test_transition_matches_ghx(solved, matrices):
    compiled, _ = solved
    A, _, _ = matrices
    n = compiled.n_exog
    assert np.abs(A[:, :n] - dyn.GHX_DECL).max() < TOL
    assert np.abs(A[:, n:]).max() < TOL


def test_impact_matrix_matches_ghu(matrices):
    """The headline parity check. Red: issue #390."""
    _, B, _ = matrices
    assert np.abs(B - dyn.GHU_DECL).max() < TOL


def test_exogenous_impact_block_matches_ghu(solved, matrices):
    """The defect on its own. ``B``'s exogenous block is written as the identity
    rather than solved, so it misses every within-period feedback."""
    compiled, _ = solved
    _, B, _ = matrices
    n = compiled.n_exog
    assert np.abs(B[:n] - dyn.GHU_DECL[:n]).max() < TOL


def test_control_loadings_reproduce_ghu_given_the_true_impact_block(solved, matrices):
    """Green, and it is what localises the failure. Feeding Dynare's own
    exogenous impact block through our control loadings reproduces ``ghu``
    exactly, so ``f`` is correct and the whole error is in the block above it."""
    compiled, _ = solved
    _, _, f = matrices
    n = compiled.n_exog
    exog = dyn.GHU_DECL[:n]
    assert np.abs(np.vstack([exog, f @ exog]) - dyn.GHU_DECL).max() < TOL


def test_shock_covariance_matches_dynare(solved):
    compiled, _ = solved
    params = backend.extract_base_params(compiled)
    assert np.abs(backend.build_Q(compiled, params) - dyn.SIGMA_E).max() < TOL


# --- simulation -------------------------------------------------------------


def test_deterministic_path_matches_dynare(solved):
    """Green. Driven by ``A`` alone, so it also confirms the row convention:
    row 0 is the first simulated period, not the initial condition."""
    compiled, sol = solved
    out = sol.sim(T=len(dyn.DET), x0=_x0(compiled))
    assert np.abs(_as_array(compiled, out) - dyn.DET).max() < TOL


def test_stochastic_path_matches_dynare(solved):
    """Red: the same innovations through a wrong ``B``."""
    compiled, sol = solved
    shocks = {
        str(compiled.config.shock_map[sym]): dyn.SHOCK_BLOCK[:, j]
        for j, sym in enumerate(compiled.config.shock_map)
    }
    out = sol.sim(T=len(dyn.STOCH), shocks=shocks, x0=_x0(compiled))
    assert np.abs(_as_array(compiled, out) - dyn.STOCH).max() < TOL


@pytest.mark.parametrize("shock", dyn.EXO_COLUMNS)
def test_unit_innovation_irf_matches_dynare(solved, shock):
    """Red. The goldens are unit-innovation responses, so ``scale`` undoes the
    standard deviation ``irf`` applies by default. That is the experiment being
    matched to the golden's experiment, not the output being adjusted."""
    compiled, sol = solved
    j = dyn.EXO_COLUMNS.index(shock)
    target = str(compiled.config.shock_map[list(compiled.config.shock_map)[j]])
    out = sol.irf(
        shocks=[target],
        T=len(dyn.IRF[shock]),
        scale=1.0 / float(np.sqrt(dyn.SIGMA_E[j, j])),
    )
    assert np.abs(_as_array(compiled, out) - dyn.IRF[shock]).max() < TOL


# --- measurement and filter -------------------------------------------------


def test_measurement_matrices_match_dynare(solved):
    compiled, _ = solved
    params = backend.extract_base_params(compiled)
    C, d = compiled.build_affine_measurement_matrices(
        params, list(compiled.observable_names), np.zeros(len(compiled.var_names))
    )
    assert np.abs(C - dyn.KF_Z).max() < TOL
    assert np.abs(d - dyn.KF_D).max() < TOL


@pytest.mark.parametrize(
    "P0, expected",
    [
        (dyn.P0_UNCOND, dyn.LOGLIK_UNCOND_CONST),
        (dyn.P0_FIXED, dyn.LOGLIK_FIXED_CONST),
    ],
    ids=["uncond", "fixed"],
)
def test_kalman_loglik_matches_dynare(solved, matrices, P0, expected):
    """Red through ``B``: the filter's state noise is ``B Q B'``.

    A pure dating slip would move this by far more than the tolerance, so a
    failure here that is *only* the size of the ``B`` error is itself evidence
    the predict-then-update pairing is right.
    """
    compiled, _ = solved
    A, B, _ = matrices
    params = backend.extract_base_params(compiled)
    Q = backend.build_Q(compiled, params)
    C, d = compiled.build_affine_measurement_matrices(
        params, list(compiled.observable_names), np.zeros(len(compiled.var_names))
    )
    out = KalmanFilter.run(
        A,
        B,
        C,
        d,
        Q,
        dyn.KF_H,
        dyn.KF_DATA,
        x0=np.zeros(len(compiled.var_names)),
        P0=P0,
    )
    assert abs(float(out.loglik) - expected) < 1e-8
