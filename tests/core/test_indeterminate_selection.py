# type: ignore
"""Which subspace a solve selects when the stable roots outnumber the states.

``indeterminate.yaml`` is block triangular: ``g`` drives the forward-looking
block and nothing feeds back into it, so the eigenvectors belonging to the two
Phillips/Euler roots carry a zero ``g`` component. With one state the selection
is a line, which leaves every selection but the one holding the ``g`` root with
an exactly singular ``z11``. The rule below is therefore the only one the solve
can return, whether it arrives at that ordering out of the QZ or reaches it by
exchange. At the passive calibration the stable roots are ``g`` and one of the
two, so the search here is the baseline against a single swap.

The companion to this is ``tests/ckernels/test_klein_reorder.py``, which drives
the search on synthetic triples. What is checked here is the number the search
exists to produce.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solver_backend import BKStatus

MODELS = Path(__file__).resolve().parents[1] / "fixtures" / "models"

#: The persistence the fixture writes as a literal in ``demand_s``.
RHO_G = 0.8


@pytest.fixture(scope="module")
def indeterminate():
    model, kalman = ModelParser(MODELS / "indeterminate.yaml").get_all()
    solver = DSGESolver(model, kalman)
    return solver, solver.compile()


def _params(compiled, **overrides) -> dict[str, float]:
    params = {
        str(name): float(value)
        for name, value in compiled.config.calibration.parameters.items()
    }
    params.update(overrides)
    return params


def _msv_loadings(par: dict[str, float]) -> dict[str, float]:
    """The loadings of ``pi``, ``x``, and ``r`` on ``g``.

    One state means the selected subspace is a line, so the rule is the
    minimum-state-variable solution and the Phillips curve and the Euler
    equation pin it between them without reference to which roots are stable.
    The determinate and the indeterminate calibration share this closed form;
    only the count of stable roots differs.
    """
    lhs = np.array(
        [
            [1.0 - RHO_G * par["beta"], -par["kappa"]],
            [
                par["tau_inv"] * (par["psi_pi"] - RHO_G),
                1.0 - RHO_G + par["tau_inv"] * par["psi_x"],
            ],
        ]
    )
    pi_g, x_g = np.linalg.solve(lhs, np.array([0.0, 1.0]))
    return {
        "pi": pi_g,
        "x": x_g,
        "r": par["psi_pi"] * pi_g + par["psi_x"] * x_g,
    }


@pytest.mark.parametrize(
    ("psi_pi", "stab", "n_stable"),
    [(0.8, BKStatus.INDETERMINATE, 2), (1.5, BKStatus.DETERMINATE, 1)],
    ids=["passive-rule", "active-rule"],
)
def test_the_selection_spans_the_state_on_either_side_of_the_taylor_principle(
    indeterminate, psi_pi, stab, n_stable
):
    solver, compiled = indeterminate
    par = _params(compiled, psi_pi=psi_pi)

    warns = (
        pytest.warns(UserWarning, match="INDETERMINATE")
        if stab is BKStatus.INDETERMINATE
        else nullcontext()
    )
    with warns:
        solved = solver.solve(compiled, parameters=par, order=1)

    assert solved.policy.stab is stab
    assert compiled.n_state == 1
    assert compiled.var_names[0] == "g"

    # sdim against nspred in the only form a caller can see. The search runs on
    # the first of these and the plain factorization on the second.
    assert (np.abs(solved.policy.eig) < 1.0).sum() == n_stable

    np.testing.assert_allclose(solved.policy.p, [[RHO_G]], atol=1e-12)

    rows = {
        name: solved.policy.f[compiled.idx[name] - compiled.n_state]
        for name in ("pi", "x", "r")
    }
    # ``f`` reads the state entering the period, which is ``g`` at ``t-1``, so a
    # loading on ``g_t`` reaches it through one more period of persistence.
    for name, loading in _msv_loadings(par).items():
        np.testing.assert_allclose(rows[name], [RHO_G * loading], rtol=1e-9, atol=1e-12)


def test_the_indeterminate_rule_is_the_determinate_one_continued(indeterminate):
    """A passive rule changes which roots are stable, not which equations the
    rule has to satisfy, so the two calibrations differ only through ``psi_pi``.
    """
    solver, compiled = indeterminate
    passive = _params(compiled, psi_pi=0.8)
    active = _params(compiled, psi_pi=1.5)

    with pytest.warns(UserWarning, match="INDETERMINATE"):
        solved_passive = solver.solve(compiled, parameters=passive, order=1)
    solved_active = solver.solve(compiled, parameters=active, order=1)

    np.testing.assert_allclose(solved_passive.policy.p, solved_active.policy.p)
    assert not np.allclose(solved_passive.policy.f, solved_active.policy.f)

    for solved, par in ((solved_passive, passive), (solved_active, active)):
        x_g = solved.policy.f[compiled.idx["x"] - compiled.n_state, 0]
        pi_g = solved.policy.f[compiled.idx["pi"] - compiled.n_state, 0]
        # The Phillips curve at the rule, which holds whatever psi_pi does.
        np.testing.assert_allclose(
            pi_g * (1.0 - RHO_G * par["beta"]), par["kappa"] * x_g, rtol=1e-9
        )
