from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, Literal
from numpy import float64
from numpy.typing import NDArray

from ....core.shock_generators import Shock
from ...mc_constructs import MCStep, OpType
from .ops import (
    simulate,
    raw_model_data_datagen,
    run_reference_filter,
    add_payload,
)

NDF = NDArray[float64]


def simulation_step(
    name: str = "datagen",
    target: str = "dgp",
    *,
    T: int,
    shocks: Mapping[str, Shock | Callable[[float | NDF], NDF] | NDF] | None = None,
    seed_increment: int | Literal["auto"] = "auto",
    shock_scale: float = 1.0,
    x0: list[float] | NDF | None = None,
    observables: bool = True,
) -> MCStep:
    """Simulate one replication's data by driving the DGP model with shocks.

    Draws fresh shocks per replication (keyed by
    ``rep_idx``) unless ``shocks`` are supplied.

    Example:
        >>> simulation_step(T=200)

    See ``operations.core`` for the shared data-source / output contract.
    """
    step_kwargs = dict(
        target=target,
        T=T,
        shocks=shocks,
        seed_increment=seed_increment,
        shock_scale=shock_scale,
        x0=x0,
        observables=observables,
    )

    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        func=simulate,
        kwargs=step_kwargs,
        step_type="simulation",
    )


def raw_model_data_step(
    name: str = "datagen",
    *,
    states: NDF | Sequence[float] | Sequence[Sequence[float]] | None = None,
    observables: NDF | Sequence[float] | Sequence[Sequence[float]] | None = None,
    raw: Mapping[str, NDF] | None = None,
    observable_names: Sequence[str] = (),
) -> MCStep:
    """Feed pre-generated arrays as each replication's data (no simulation).

    Provide ``states``, ``observables``, or both; a leading replication axis is
    indexed by ``rep_idx`` (a 2-D array is reused across replications).

    Example:
        >>> raw_model_data_step(observables=obs, observable_names=("y", "x"))

    See ``operations.core`` for the shared data-source / output contract.
    """
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        func=raw_model_data_datagen,
        kwargs={
            "states": states,
            "observables": observables,
            "raw": raw,
            "observable_names": observable_names,
        },
        step_type="raw_model_data",
    )


def reference_filter_step(
    name: str = "filter",
    *,
    filter_mode: Literal["linear", "extended", "unscented"] = "linear",
    observables: list[str] | None = None,
    x0: list[float] | NDF | None = None,
    R: NDF | None = None,
    p0_mode: Literal["diag", "eye"] | None = None,
    p0_scale: float | float64 | None = None,
    jitter: float | float64 | None = None,
    symmetrize: bool | None = None,
    return_shocks: bool = False,
) -> MCStep:
    """Kalman-filter each replication's observables with the reference model.

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
        kwargs={
            "filter_mode": filter_mode,
            "observables": observables,
            "x0": x0,
            "R": R,
            "p0_mode": p0_mode,
            "p0_scale": p0_scale,
            "jitter": jitter,
            "symmetrize": symmetrize,
            "return_shocks": return_shocks,
        },
        step_type="filter",
    )


def add_payload_step(
    name: str,
    payload: NDF | Sequence[float] | Sequence[Sequence[float]],
) -> MCStep:
    """Add a replication-specific payload to the context.

    The ``payload`` is stored in ``context.payload[name]`` for that replication.
    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=add_payload,
        kwargs={"value": payload},
        step_type="payload",
    )
