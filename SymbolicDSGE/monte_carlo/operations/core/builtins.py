from __future__ import annotations
from typing import Any

from ...mc_constructs import MCStep, OpType
from .._docs import with_base_doc

from .ops import simulate_dgp, raw_data_datagen, run_reference_filter

_BASE_DOC = """
Monte Carlo per-replication data sources and the reference Kalman filter.

- These seed a pipeline: a DATAGEN step (``simulation`` or ``raw_data``) must
  run first to populate ``context.data`` for that replication.
- Output location: datagen fills ``context.data`` (states / observables / raw
  series); the ``filter`` step stores a ``FilterResult`` that downstream steps
  read via ``source="filter"``.
"""


@with_base_doc(_BASE_DOC)
def simulation_step(name: str = "datagen", **kwargs: Any) -> MCStep:
    """Simulate one replication's data by driving the DGP model with shocks.

    Signature: ``simulation_step(name="datagen", *, T, shocks=None,
    seed_increment="auto", shock_scale=1.0, x0=None, observables=True)``.

    Requires a DGP ``SolvedModel``; draws fresh shocks per replication (keyed by
    ``rep_idx``) unless ``shocks`` are supplied.

    Example:
        >>> simulation_step(T=200)
    """
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        func=simulate_dgp,
        kwargs=kwargs,
        step_type="simulation",
    )


@with_base_doc(_BASE_DOC)
def raw_data_step(name: str = "datagen", **kwargs: Any) -> MCStep:
    """Feed pre-generated arrays as each replication's data (no simulation).

    Signature: ``raw_data_step(name="datagen", *, states=None, observables=None,
    n_exog=-1, raw=None, observable_names=())``.

    Provide ``states``, ``observables``, or both; a leading replication axis is
    indexed by ``rep_idx`` (a 2-D array is reused across replications).

    Example:
        >>> raw_data_step(observables=obs, observable_names=("y", "x"))
    """
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        func=raw_data_datagen,
        kwargs=kwargs,
        step_type="raw_data",
    )


@with_base_doc(_BASE_DOC)
def reference_filter_step(name: str = "filter", **kwargs: Any) -> MCStep:
    """Kalman-filter each replication's observables with the reference model.

    Signature: ``reference_filter_step(name="filter", *, filter_mode="linear",
    observables=None, x0=None, R=None, estimate_R_diag=False, R_scale=1.0,
    return_shocks=False, ...)``.

    Requires generated observables (run a DATAGEN step first); the resulting
    ``FilterResult`` is read downstream via ``source="filter"``.

    Example:
        >>> reference_filter_step()
    """
    return MCStep(
        name=name,
        op_type=OpType.FILTER,
        func=run_reference_filter,
        kwargs=kwargs,
        step_type="filter",
    )
