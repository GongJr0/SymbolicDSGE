"""Shared real-model fixtures for the estimation suite.

The native estimation path (``Estimator.mle`` / ``map``, ``solver.estimate``)
drives the compiled kernels end to end, so it cannot be exercised on a
``SimpleNamespace`` stub: construction resolves ``ss_seed`` through the solver
and the driver needs real cfunc addresses and a real Klein solve. The model is
the session-wide ``post82_bundle``; what this package adds is the observation
panel the goldens are values off, and a guarded handle for the one call that
writes to the model it was given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SymbolicDSGE.core.config import SymbolGetterDict
from SymbolicDSGE.estimation import Estimator


@pytest.fixture(scope="session")
def y_post82(compiled_post82, solved_post82):
    """A simulated panel over POST82's observables, in their compiled order.

    The seed and ``T`` are fixed because the oracle goldens are values off this
    panel, not properties of it.
    """
    calib = compiled_post82.config.calibration.parameters
    T = 48
    rng = np.random.default_rng(20260724)
    sim = solved_post82.sim(
        T=T,
        shocks={
            "e_g": rng.normal(0.0, calib["sig_g"], size=T),
            "e_z": rng.normal(0.0, calib["sig_z"], size=T),
            "e_r": rng.normal(0.0, calib["sig_r"], size=T),
        },
        x0=np.zeros((compiled_post82.n_var,), dtype=np.float64),
        observables=True,
    )
    return pd.DataFrame(
        {name: sim.observables[name] for name in compiled_post82.observable_names}
    )


@pytest.fixture
def post82_restore(post82_bundle):
    """The session bundle, with its calibration put back afterwards.

    ``estimate_and_solve`` syncs the estimated point into the model's
    calibration, and the bundle is shared with the whole suite, so a test that
    runs one reaches the model through here rather than through
    ``compiled_post82``.
    """
    calibration = post82_bundle["compiled"].config.calibration
    entered = SymbolGetterDict(calibration.parameters)
    try:
        yield post82_bundle
    finally:
        calibration.parameters = entered


@pytest.fixture
def post82_estimator(compiled_post82, solved_post82, y_post82):
    """Factory building an :class:`Estimator` on the shared POST82 model.

    Defaults to the two-parameter linear MLE setup the oracle uses; callers
    override ``mode`` / ``estimated_params`` as needed. A mapping passed as
    ``estimated_params`` carries the priors.
    """

    def _make(mode="linear", estimated_params=("psi_pi", "rho_r")):
        return Estimator(
            compiled=compiled_post82,
            y=y_post82,
            observables=compiled_post82.observable_names,
            filter_mode=mode,
            estimated_params=estimated_params,
            ss_seed=solved_post82.policy.steady_state,
        )

    return _make
