from __future__ import annotations

from typing import Any
from ...mc_constructs import MCStep, OpType

from .ops import (
    run_wald_test,
    run_ljung_box_test,
    run_jarque_bera_test,
    run_breusch_pagan_test,
    run_breusch_godfrey_test,
    run_cusum_test,
    run_cusumsq_test,
    run_chow_test,
)


def wald_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_wald_test,
        kwargs=kwargs,
        step_type="wald",
    )


def ljung_box_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_ljung_box_test,
        kwargs=kwargs,
        step_type="ljung_box",
    )


def jarque_bera_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_jarque_bera_test,
        kwargs=kwargs,
        step_type="jarque_bera",
    )


def breusch_pagan_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_breusch_pagan_test,
        kwargs=kwargs,
        step_type="breusch_pagan",
    )


def breusch_godfrey_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_breusch_godfrey_test,
        kwargs=kwargs,
        step_type="breusch_godfrey",
    )


def cusum_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_cusum_test,
        kwargs=kwargs,
        step_type="cusum",
    )


def cusumsq_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_cusumsq_test,
        kwargs=kwargs,
        step_type="cusumsq",
    )


def chow_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        func=run_chow_test,
        kwargs=kwargs,
        step_type="chow",
    )
