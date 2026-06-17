"""Monte-Carlo operations, organized by group.

Each subpackage exposes its ``*_step`` factories; reach them through the group
rather than a flat namespace, e.g.::

    from SymbolicDSGE.monte_carlo.operations.tests import wald_test_step
    from SymbolicDSGE.monte_carlo.operations.regressions import regression_step
    from SymbolicDSGE.monte_carlo.operations.transforms import standardize_step
    from SymbolicDSGE.monte_carlo.operations.core import simulation_step

The underlying ``run_*`` op implementations are internal and live in each
group's ``ops`` module (e.g. ``operations.core.ops.simulate_dgp``).
"""

from . import core, postproc, regressions, tests, transforms

__all__ = ["core", "tests", "transforms", "regressions", "postproc"]
