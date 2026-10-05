# type: ignore
"""Solve failures: the class a kernel status raises and what its message carries."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from SymbolicDSGE._ckernels.core._solve_errors import SolveStatus
from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solver_backend import (
    _FAILURES,
    DecisionRuleError,
    SteadyStateError,
)

MODELS = Path(__file__).resolve().parents[1] / "fixtures" / "models"

# `_classify` walks `compiled.var_names`, the canonical layout, which leads with
# the lagged variables. Gali (2015) lags y, i, nu, a, and z, and leads pi and y_gap.
_GALI_STATES = ("y", "i", "nu", "a", "z")
_GALI_JUMPS = ("pi", "y_gap")


def _no_fixed_point_yaml() -> str:
    """A drift: the residual is a nonzero constant and its Jacobian is zero."""
    return textwrap.dedent(
        """
        name: "NO_FIXED_POINT"
        variables:
          x: {}
        shocks:
          - e
        equations:
          model:
            drift: "x(t) = x(t-1) + 1 + e"
          constraint: {}
          observables: {}
        calibration:
          parameters:
            sig: 0.01
          shocks:
            std:
              e: sig
            corr: {}
        """
    )


def _overflowing_residual_yaml() -> str:
    """A seed that overflows the first residual evaluation."""
    return textwrap.dedent(
        """
        name: "OVERFLOWING_RESIDUAL"
        variables:
          x: {ss_seed: x_seed}
        shocks:
          - e
        equations:
          model:
            growth: "x(t) = exp(x(t-1)) + e"
          constraint: {}
          observables: {}
        calibration:
          parameters:
            x_seed: 1000.0
            sig: 0.01
          shocks:
            std:
              e: sig
            corr: {}
        """
    )


def _compiled(path: Path):
    model, kalman = ModelParser(path).get_all()
    solver = DSGESolver(model, kalman)
    return solver, solver.compile()


def _written(tmp_path: Path, name: str, spec: str):
    path = tmp_path / f"{name}.yaml"
    path.write_text(spec, encoding="utf-8")
    return _compiled(path)


def _params(compiled, **overrides) -> dict[str, float]:
    params = {
        str(name): float(value)
        for name, value in compiled.config.calibration.parameters.items()
    }
    params.update(overrides)
    return params


@pytest.fixture(scope="module")
def gali():
    return _compiled(MODELS / "gali_2015.yaml")


@pytest.fixture(scope="module")
def rbc():
    return _compiled(MODELS / "rbc_second_order.yaml")


def test_no_stable_solution_carries_the_partition_and_the_eigenvalues(gali):
    solver, compiled = gali
    with pytest.raises(DecisionRuleError) as exc:
        solver.solve(compiled, parameters=_params(compiled, rho_a=1.8), order=1)

    err = exc.value
    assert err.status is SolveStatus.NO_STABLE_SOLUTION
    assert err.states == _GALI_STATES
    assert err.jumps == _GALI_JUMPS
    assert err.mixed == ()
    assert len(err.states) == compiled.n_state

    # The verdict is fewer stable roots than states, which the reported
    # eigenvalues have to agree with.
    assert (abs(err.eig) < 1.0).sum() < compiled.n_state

    message = str(err)
    assert "Code: NO_STABLE_SOLUTION" in message
    assert f"State space: 5 state(s) {_GALI_STATES}, 2 jump(s) {_GALI_JUMPS}" in message
    assert "Generalized eigenvalues:" in message
    assert "occurring at both dates" not in message


def test_a_variable_at_both_dates_is_counted_in_each_block(rbc):
    solver, compiled = rbc
    with pytest.raises(DecisionRuleError) as exc:
        solver.solve(compiled, parameters=_params(compiled, rho=1.8), order=2)

    err = exc.value
    assert err.status is SolveStatus.NO_STABLE_SOLUTION
    assert err.states == ("k", "z")
    assert err.jumps == ("z", "c")
    assert err.mixed == ("z",)

    message = str(err)
    assert "State space: 2 state(s) ('k', 'z'), 2 jump(s) ('z', 'c')" in message
    assert "('z',) occurring at both dates and counted in each" in message


def test_singular_steady_state_jacobian_reports_no_partition(tmp_path):
    solver, compiled = _written(tmp_path, "no_fixed_point", _no_fixed_point_yaml())
    with pytest.raises(SteadyStateError) as exc:
        solver.solve(compiled, parameters=_params(compiled), order=1)

    err = exc.value
    assert err.status is SolveStatus.NEWTON_SINGULAR
    message = str(err)
    assert "Code: NEWTON_SINGULAR (-901)" in message
    assert "State space:" not in message
    assert "Generalized eigenvalues:" not in message


def test_non_finite_steady_state_residual_does_not_converge(tmp_path):
    solver, compiled = _written(
        tmp_path, "overflowing_residual", _overflowing_residual_yaml()
    )
    with pytest.raises(SteadyStateError) as exc:
        solver.solve(compiled, parameters=_params(compiled), order=1)

    assert exc.value.status is SolveStatus.NEWTON_NO_CONVERGE
    assert "State space:" not in str(exc.value)


def test_every_status_has_a_failure_row():
    assert set(_FAILURES) == set(SolveStatus)
