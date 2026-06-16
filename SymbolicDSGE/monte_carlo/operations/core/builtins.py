from __future__ import annotations
from typing import Any, Callable

from ...mc_constructs import MCStep, OpType

from .ops import simulate_dgp, raw_data_datagen, run_reference_filter


def simulation_step(name: str = "datagen", **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        func=simulate_dgp,
        kwargs=kwargs,
        step_type="simulation",
    )


def raw_data_step(name: str = "datagen", **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        func=raw_data_datagen,
        kwargs=kwargs,
        step_type="raw_data",
    )


def reference_filter_step(name: str = "filter", **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.FILTER,
        func=run_reference_filter,
        kwargs=kwargs,
        step_type="filter",
    )
