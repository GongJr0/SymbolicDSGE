# type: ignore
"""Blanchard-Kahn reporting: the raise branch, the warn branch, and the messages."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solver_backend import BKStatus

MODELS = Path(__file__).resolve().parents[1] / "fixtures" / "models"

# Gali (2015) is determinate as calibrated. A passive Taylor rule leaves a
# forward-looking variable unpinned; an explosive technology process puts an
# unstable root on a state. One violation each way, off one parameter.
_VIOLATIONS = [
    pytest.param({"phi_pi": 0.5}, BKStatus.INDETERMINATE, id="indeterminate"),
    pytest.param({"rho_a": 1.8}, BKStatus.NO_STABLE_SOLUTION, id="no-stable-solution"),
]


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
@pytest.mark.parametrize("overrides, expected", _VIOLATIONS)
def test_violation_raises_by_default(gali, order, overrides, expected):
    solver, compiled = gali
    with pytest.raises(ValueError, match=expected.name) as exc:
        solver.solve(compiled, parameters=_params(compiled, **overrides), order=order)
    assert expected.message in str(exc.value)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("overrides, expected", _VIOLATIONS)
def test_violation_warns_and_returns_when_raising_is_off(
    gali, order, overrides, expected
):
    solver, compiled = gali
    with pytest.warns(UserWarning, match=expected.name) as record:
        solved = solver.solve(
            compiled,
            parameters=_params(compiled, **overrides),
            order=order,
            raise_on_bk_violation=False,
        )
    assert expected.message in str(record[0].message)
    assert solved.policy.stab is expected
    assert solved.policy.is_determinate is False


@pytest.mark.parametrize("order", [1, 2])
def test_determinate_solve_is_silent(gali, order):
    solver, compiled = gali
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        solved = solver.solve(compiled, parameters=_params(compiled), order=order)
    assert solved.policy.stab is BKStatus.DETERMINATE
    assert solved.policy.is_determinate is True


def test_every_status_has_its_own_message():
    messages = {status.message for status in BKStatus}
    assert len(messages) == len(BKStatus)
    assert all(messages)


def test_unset_sentinel_constructs_from_the_c_value():
    assert BKStatus(2) is BKStatus.UNSET
