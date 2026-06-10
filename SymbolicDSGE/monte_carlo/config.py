from __future__ import annotations

from typing import Any, Callable

from .mc_constructs import MCStep, OpType
from .operations import (
    raw_data_datagen as _raw_data_datagen,
    run_ljung_box_test as _run_ljung_box_test,
    run_reference_filter as _run_reference_filter,
    run_regression as _run_regression,
    run_wald_test as _run_wald_test,
    run_jarque_bera_test as _run_jarque_bera_test,
    run_breusch_pagan_test as _run_breusch_pagan_test,
    run_breusch_godfrey_test as _run_breusch_godfrey_test,
    run_cusum_test as _run_cusum_test,
    simulate_dgp as _simulate_dgp,
)


def simulation_step(name: str = "datagen", **kwargs: Any) -> MCStep:
    return MCStep(name=name, op_type=OpType.DATAGEN, func=_simulate_dgp, kwargs=kwargs)


def raw_data_step(name: str = "datagen", **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.DATAGEN, func=_raw_data_datagen, kwargs=kwargs
    )


def transform_step(
    name: str,
    func: Callable[..., Any],
    *,
    store_key: str | None = None,
    **kwargs: Any,
) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=func,
        kwargs=kwargs,
        store_key=store_key,
    )


def reference_filter_step(name: str = "filter", **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.FILTER, func=_run_reference_filter, kwargs=kwargs
    )


def wald_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(name=name, op_type=OpType.TEST, func=_run_wald_test, kwargs=kwargs)


def ljung_box_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.TEST, func=_run_ljung_box_test, kwargs=kwargs
    )


def jarque_bera_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.TEST, func=_run_jarque_bera_test, kwargs=kwargs
    )


def breusch_pagan_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.TEST, func=_run_breusch_pagan_test, kwargs=kwargs
    )


def breusch_godfrey_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.TEST, func=_run_breusch_godfrey_test, kwargs=kwargs
    )


def cusum_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(name=name, op_type=OpType.TEST, func=_run_cusum_test, kwargs=kwargs)


def regression_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.REGRESSION, func=_run_regression, kwargs=kwargs
    )
