from __future__ import annotations
from typing import Any, Literal

from ...mc_constructs import MCStep, OpType

from .ops import (
    simulate,
    raw_data_datagen,
    run_reference_filter,
)


def simulation_step(
    name: str = "datagen",
    **kwargs: Any,
) -> MCStep:
    """Simulate one replication's data by driving the DGP model with shocks.

    Signature: ``simulation_step(name="datagen", *, target="dgp", T, shocks=None,
    seed_increment="auto", shock_scale=1.0, x0=None, observables=True)``.

    Draws fresh shocks per replication (keyed by
    ``rep_idx``) unless ``shocks`` are supplied.

    Example:
        >>> simulation_step(T=200)

    See ``operations.core`` for the shared data-source / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        func=simulate,
        kwargs=kwargs,
        step_type="simulation",
    )


def raw_data_step(name: str = "datagen", **kwargs: Any) -> MCStep:
    """Feed pre-generated arrays as each replication's data (no simulation).

    Signature: ``raw_data_step(name="datagen", *, states=None, observables=None,
    n_exog=-1, raw=None, observable_names=())``.

    Provide ``states``, ``observables``, or both; a leading replication axis is
    indexed by ``rep_idx`` (a 2-D array is reused across replications).

    Example:
        >>> raw_data_step(observables=obs, observable_names=("y", "x"))

    See ``operations.core`` for the shared data-source / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        func=raw_data_datagen,
        kwargs=kwargs,
        step_type="raw_data",
    )


def reference_filter_step(name: str = "filter", **kwargs: Any) -> MCStep:
    """Kalman-filter each replication's observables with the reference model.

    Signature: ``reference_filter_step(name="filter", *, filter_mode="linear",
    observables=None, x0=None, R=None, estimate_R_diag=False, R_scale=1.0,
    return_shocks=False, ...)``.

    Requires generated observables (run a DATAGEN step first); the resulting
    Raw filter output is read downstream with ``source=name, field=field``.

    Example:
        >>> reference_filter_step()

    See ``operations.core`` for the shared data-source / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.FILTER,
        func=run_reference_filter,
        kwargs=kwargs,
        step_type="filter",
    )
