# type: ignore
"""Blanchard-Kahn reporting: the silent path and the warn path."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solver_backend import BKStatus

MODELS = Path(__file__).resolve().parents[1] / "fixtures" / "models"


def _compiled(name: str):
    model, kalman = ModelParser(MODELS / name).get_all()
    solver = DSGESolver(model, kalman)
    return solver, solver.compile()


@pytest.fixture(scope="module")
def gali():
    return _compiled("gali_2015.yaml")


def _params(compiled, **overrides) -> dict[str, float]:
    params = {
        str(name): float(value)
        for name, value in compiled.config.calibration.parameters.items()
    }
    params.update(overrides)
    return params


@pytest.mark.parametrize("order", [1, 2])
def test_determinate_solve_is_silent(gali, order):
    solver, compiled = gali
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        solved = solver.solve(compiled, parameters=_params(compiled), order=order)
    assert solved.policy.stab is BKStatus.DETERMINATE
    assert solved.is_determinate is True
    assert "unique and stable" in solved.policy.stab.message


# Gali (2015) is determinate as calibrated. A passive Taylor rule leaves a
# forward-looking variable unpinned, one violation off one parameter.
@pytest.mark.parametrize("order", [1, 2])
def test_indeterminacy_warns_and_returns(gali, order):
    solver, compiled = gali
    with pytest.warns(UserWarning, match="INDETERMINATE") as record:
        solved = solver.solve(
            compiled, parameters=_params(compiled, phi_pi=0.5), order=order
        )
    assert BKStatus.INDETERMINATE.message in str(record[0].message)
    assert solved.policy.stab is BKStatus.INDETERMINATE
    assert solved.is_determinate is False
