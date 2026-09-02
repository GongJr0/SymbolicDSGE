"""Native diagnostic-test step lowering."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..._ckernels.monte_carlo import _offsets
from ..._ckernels.monte_carlo._runner import (
    DEFAULT_BREUSCH_GODFREY_LAGS,
    NativeStep,
    breusch_godfrey_step,
    breusch_pagan_step,
    chow_step,
    cusum_step,
    cusumsq_step,
    jarque_bera_step,
    ljung_box_step,
    wald_step,
)
from ..allocation import BufferPlan, _selected_source_shape
from ..defaults import DEFAULT_WALD_KIND_NAME
from ..mc_constructs import MCStep
from .utils import (
    NDF,
    FloatInputBinding,
    _supplied,
    _source_binding,
)


def lower_test_step(
    step: MCStep,
    source_indices: tuple[int, ...],
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    """Compile source transfers and context constants for one diagnostic."""
    n, first_columns = _selected_source_shape(
        plan, steps, source_indices[0], step.source_args[0]
    )
    kind = step.step_type
    if kind == "wald":
        return _lower_wald_step(step, source_indices[0], steps, plan, n, first_columns)
    if kind in {"ljung_box", "jarque_bera"}:
        native_step = (
            ljung_box_step(step.name, n, **_supplied(step.kwargs, "lags"))
            if kind == "ljung_box"
            else jarque_bera_step(step.name, n)
        )
        return native_step, (
            _source_binding(
                source_indices[0],
                steps,
                plan,
                step.source_args[0],
                target_offset=0,
                target_row_stride=1,
            ),
        )
    _, x_columns = _selected_source_shape(
        plan, steps, source_indices[1], step.source_args[1]
    )
    lags = (
        int(step.kwargs.get("lags", DEFAULT_BREUSCH_GODFREY_LAGS))
        if kind == "breusch_godfrey"
        else 0
    )
    offsets = _offsets.diagnostic_offsets(kind or "", n, x_columns, lags).foffset
    if kind == "breusch_pagan":
        native_step = breusch_pagan_step(
            step.name, n, x_columns, **_supplied(step.kwargs, "robust")
        )
    elif kind == "breusch_godfrey":
        native_step = breusch_godfrey_step(
            step.name, n, x_columns, **_supplied(step.kwargs, "lags")
        )
    elif kind == "cusum":
        native_step = cusum_step(step.name, n, x_columns)
    elif kind == "cusumsq":
        native_step = cusumsq_step(step.name, n, x_columns)
    else:
        native_step = chow_step(
            step.name, n, x_columns, **_supplied(step.kwargs, "t_break")
        )
    return native_step, (
        _source_binding(
            source_indices[0],
            steps,
            plan,
            step.source_args[0],
            target_offset=0,
            target_row_stride=1,
        ),
        _source_binding(
            source_indices[1],
            steps,
            plan,
            step.source_args[1],
            target_offset=offsets[1],
            target_row_stride=x_columns,
        ),
    )


def _lower_wald_step(
    step: MCStep,
    source_idx: int,
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
    n: int,
    q: int,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    # The target's expected shape follows the kind, so this one resolves here.
    kind_code = _wald_kind(step.kwargs)
    target = _wald_target(step.kwargs["target"], q, kind_code)
    return wald_step(
        step.name,
        target,
        n,
        q,
        kind=kind_code,
        **_wald_kernel(step.kwargs),
        **_wald_bandwidth(step.kwargs),
    ), (
        _source_binding(
            source_idx,
            steps,
            plan,
            step.source_args[0],
            target_offset=0,
            target_row_stride=q,
        ),
    )


def _wald_kind(kwargs: Mapping[str, Any]) -> int:
    """A Wald statistic's kind as its native code."""
    kind = kwargs.get("kind", DEFAULT_WALD_KIND_NAME)
    try:
        return {"mean": 0, "covariance": 1, "second_moment": 2}[kind]
    except KeyError as exc:
        raise ValueError("Unsupported native Wald configuration.") from exc


def _wald_kernel(kwargs: Mapping[str, Any]) -> dict[str, int]:
    """A Wald HAC kernel as its native id, or nothing when unset."""
    kernel = kwargs.get("kernel")
    if kernel is None:
        return {}
    try:
        return {"kernel_id": {"bartlett": 0, "parzen": 1, "qs": 2}[kernel]}
    except KeyError as exc:
        raise ValueError("Unsupported native Wald configuration.") from exc


def _wald_bandwidth(kwargs: Mapping[str, Any]) -> dict[str, int]:
    """A Wald bandwidth as its native mode and manual pair, or nothing when unset."""
    bandwidth_modes = {"andrews": 2, "wooldridge": 1, "auto": 3}
    bandwidth = kwargs.get("bandwidth")
    if bandwidth is None:
        return {}
    if isinstance(bandwidth, bool):
        raise ValueError("Wald bandwidth must be an integer, mode, or None.")
    if isinstance(bandwidth, int):
        return {"bandwidth_mode": 0, "manual_bandwidth": bandwidth}
    try:
        return {"bandwidth_mode": bandwidth_modes[bandwidth], "manual_bandwidth": 0}
    except KeyError as exc:
        raise ValueError("Unsupported native Wald bandwidth mode.") from exc


def _wald_target(value: object, q: int, kind: int) -> NDF:
    target = np.asarray(value, dtype=np.float64)
    if kind == 0:
        if target.ndim != 1 or target.shape[0] != q:
            raise ValueError("Wald mean target must have one value per source column.")
    elif target.ndim != 2 or target.shape != (q, q):
        raise ValueError("Wald matrix target must match the source column count.")
    return target
