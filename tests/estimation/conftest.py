"""Shared real-model fixtures for the estimation suite.

The native estimation path (``Estimator.mle`` / ``map``, ``solver.estimate``)
drives the compiled kernels end to end, so it can no longer be exercised on a
``SimpleNamespace`` stub: construction resolves ``ss_seed`` through the solver
and the driver needs real cfunc addresses and a real Klein solve. Tests that
run an optimization therefore share one compiled POST82 model, built once for
the package.

Sharing it means the one mutable thing it carries has to be put back after
every test: ``estimate_and_solve`` writes the estimated point into the model's
calibration, so a golden computed off that calibration would otherwise depend
on what ran before it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sympy import Symbol

from SymbolicDSGE import ModelParser, DSGESolver
from SymbolicDSGE.core.config import SymbolGetterDict
from SymbolicDSGE.estimation import Estimator


@pytest.fixture(scope="session")
def post82(post82_test_model_path):
    """A compiled POST82 model plus a short simulated observation panel.

    Returns the solver, compiled model, observation frame, steady-state seed and
    observable names. The simulation seed is fixed so the golden oracle values
    stay reproducible.
    """
    model, kalman = ModelParser(post82_test_model_path).get_all()
    solver = DSGESolver(model, kalman)
    compiled = solver.compile()
    steady = np.zeros((compiled.n_declared,), dtype=np.float64)
    solved = solver.solve(compiled=compiled, ss_seed=steady)

    calib = compiled.config.calibration
    sig = {
        s: float(calib.parameters[calib.shock_std[Symbol(s)]])
        for s in ("e_g", "e_z", "e_r")
    }
    T = 48
    rng = np.random.default_rng(20260724)
    sim = solved.sim(
        T=T,
        shocks={
            "e_g": rng.normal(0.0, sig["e_g"], size=T),
            "e_z": rng.normal(0.0, sig["e_z"], size=T),
            "e_r": rng.normal(0.0, sig["e_r"], size=T),
        },
        x0=np.zeros((compiled.n_var,), dtype=np.float64),
        observables=True,
    )
    y = pd.DataFrame(
        {
            "OutGap": sim.observables["OutGap"],
            "Infl": sim.observables["Infl"],
            "Rate": sim.observables["Rate"],
        }
    )
    return {
        "solver": solver,
        "compiled": compiled,
        "y": y,
        "steady": steady,
        "obs": ["OutGap", "Infl", "Rate"],
    }


@pytest.fixture(autouse=True)
def post82_calibration_is_restored(request):
    """Put the shared model's calibration back after each test.

    Requested lazily: a test that never asks for ``post82`` does not pay for
    building it.
    """
    if "post82" not in request.fixturenames:
        yield
        return

    calibration = request.getfixturevalue("post82")["compiled"].config.calibration
    entered = SymbolGetterDict(calibration.parameters)
    try:
        yield
    finally:
        calibration.parameters = entered


@pytest.fixture
def post82_estimator(post82):
    """Factory building an :class:`Estimator` on the shared POST82 model.

    Defaults to the two-parameter linear MLE setup the oracle uses; callers
    override ``mode`` / ``estimated_params`` as needed. A mapping passed as
    ``estimated_params`` carries the priors.
    """

    def _make(mode="linear", estimated_params=("psi_pi", "rho_r")):
        return Estimator(
            compiled=post82["compiled"],
            y=post82["y"],
            observables=post82["obs"],
            filter_mode=mode,
            estimated_params=estimated_params,
            ss_seed=post82["steady"],
        )

    return _make
