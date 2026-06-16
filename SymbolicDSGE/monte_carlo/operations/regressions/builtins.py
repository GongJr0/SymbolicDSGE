from __future__ import annotations

from typing import Any
from ...mc_constructs import MCStep, OpType
from .ops import run_regression


def regression_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.REGRESSION,
        func=run_regression,
        kwargs=kwargs,
        step_type="regression",
    )
