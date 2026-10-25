"""Shared real-model fixtures for the estimation suite.

The native estimation path (``Estimator.mle`` / ``map``, ``solver.estimate``)
drives the compiled kernels end to end, so it cannot be exercised on a
``SimpleNamespace`` stub: construction resolves ``ss_seed`` through the solver
and the driver needs real cfunc addresses and a real Klein solve. The model and
the observation panel are the session-wide ``post82_bundle`` and ``y_post82``;
what this package adds is a guarded handle for the one call that writes to the
model it was given.
"""

from __future__ import annotations

import pytest

from SymbolicDSGE.core.config import SymbolGetterDict


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
