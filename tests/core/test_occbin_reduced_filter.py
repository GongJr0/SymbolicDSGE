"""Filtering and estimation on a constrained model reach its reference block.

A ``PiecewiseSolution`` holds the reference regime as an ordinary
``FirstOrderSolution``. The filter modes that consume a first-order state space,
``linear`` and ``extended``, read it through the solution's ``A``/``B``, so
filtering a model with occasionally binding constraints returns the reference
regime's answer and the constraint reaches nothing.

That is the whole claim here, and it is checked as an identity rather than a
tolerance: the same data through :class:`FirstOrderSolvedModel` built on
``policy.ref`` has to give back the same arrays bit for bit, because it is the
same matrices through the same kernel. A tolerance would pass on a path that
merely resembles the reference one.

``unscented`` consumes second-order tensors, which a piecewise solve does not
produce, and stays refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from SymbolicDSGE.core import DSGESolver, ModelParser
from SymbolicDSGE.core.solved_model import FirstOrderSolvedModel
from SymbolicDSGE.estimation import Estimator

_MODEL = "tests/fixtures/models/nk_zlb_1_constraint.yaml"

#: The discount-factor shock the parity fixture is built around, at the size
#: that drives the rate into the bound. Fixed here so the observations every
#: test filters are the same ones.
_T = 40
_SHOCK_DATE = 5
_SHOCK_SIZE = 0.025

#: Estimating one parameter is enough to build the theta vector the likelihood
#: is evaluated at; nothing here optimizes.
_ESTIMATED = ["RHO"]


@pytest.fixture(scope="module")
def solved():
    model, kalman = ModelParser(_MODEL).get_all()
    solver = DSGESolver(model, kalman)
    return solver.solve(solver.compile())


@pytest.fixture(scope="module")
def observations(solved):
    """Observables off the model's own piecewise path, so the run binds."""
    shock = np.zeros(_T, dtype=np.float64)
    shock[_SHOCK_DATE] = _SHOCK_SIZE
    sim = solved.sim(_T, shocks={"epsi": shock}, observables=True)
    return np.asarray(sim.y, dtype=np.float64)


@pytest.fixture(scope="module")
def reference(solved):
    """The reference regime as the first-order model it already is."""
    return FirstOrderSolvedModel(solved.compiled, solved.policy.ref)


@pytest.mark.parametrize("filter_mode", ["linear", "extended"])
def test_filtering_a_constrained_model_is_filtering_its_reference_regime(
    solved, reference, observations, filter_mode
):
    ours = solved.kalman(observations, filter_mode=filter_mode)
    theirs = reference.kalman(observations, filter_mode=filter_mode)

    assert float(ours.loglik) == float(theirs.loglik)
    np.testing.assert_array_equal(ours.x_filt, theirs.x_filt)
    np.testing.assert_array_equal(ours.P_filt, theirs.P_filt)
    np.testing.assert_array_equal(ours.y_pred, theirs.y_pred)


def test_the_constraint_binds_on_the_data_being_filtered(solved, observations):
    # Without this the equality above would hold for the uninteresting reason
    # that the run never left the reference regime to begin with.
    shock = np.zeros(_T, dtype=np.float64)
    shock[_SHOCK_DATE] = _SHOCK_SIZE

    assert (solved.sim(_T, shocks={"epsi": shock}).regimes[:, 0] == 1).any()


def test_unscented_filtering_is_refused(solved, observations):
    with pytest.raises(ValueError, match="requires a second order solution"):
        solved.kalman(observations, filter_mode="unscented")


def test_the_estimation_likelihood_runs_on_a_constrained_model(solved, observations):
    # Estimation solves per draw rather than taking a solved model, so it
    # reaches the reference block by its own route. At the calibrated theta it
    # has to land on the one-shot filter's number exactly.
    model, kalman = ModelParser(_MODEL).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()

    est = Estimator(
        solver=solver,
        compiled=compiled,
        y=observations,
        filter_mode="linear",
        estimated_params=_ESTIMATED,
    )

    ours = float(est.loglik(est.theta0()))
    one_shot = float(solved.kalman(observations, filter_mode="linear").loglik)

    assert ours == one_shot
