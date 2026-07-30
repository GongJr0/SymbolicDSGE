"""Native diagnostic-test step lowering."""

from __future__ import annotations

import numpy as np

from ..._ckernels.monte_carlo._runner import (
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
from ..allocation import BufferPlan
from ..mc_constructs import MCStep
from .core import (
    NDF,
    FloatInputBinding,
    _array_f64,
    _selected_shape,
    _source_binding,
)


def lower_test_step(
    step: MCStep,
    source_indices: tuple[int, ...],
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    """Compile source transfers and context constants for one diagnostic."""
    n, first_columns = _selected_shape(
        source_indices[0], step.source_args[0], steps, plan
    )
    kind = step.step_type
    if kind == "wald":
        return _lower_wald_step(step, source_indices[0], steps, plan, n, first_columns)
    if kind in {"ljung_box", "jarque_bera"}:
        if first_columns != 1:
            raise ValueError(f"Native {kind} lowering requires one column.")
        native_step = (
            ljung_box_step(step.name, n, int(step.kwargs["lags"]))
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
    supported = {"breusch_pagan", "breusch_godfrey", "cusum", "cusumsq", "chow"}
    if kind not in supported:
        raise ValueError(f"Unsupported native diagnostic kind: {kind!r}.")
    if first_columns != 1:
        raise ValueError(f"Native {kind} lowering requires a one-column response.")
    x_rows, x_columns = _selected_shape(
        source_indices[1], step.source_args[1], steps, plan
    )
    if x_rows != n:
        raise ValueError("Native diagnostic sources must have matching row counts.")
    if kind == "breusch_pagan":
        native_step = breusch_pagan_step(
            step.name, n, x_columns, bool(step.kwargs["robust"])
        )
    elif kind == "breusch_godfrey":
        native_step = breusch_godfrey_step(
            step.name, n, x_columns, int(step.kwargs["lags"])
        )
    elif kind == "cusum":
        native_step = cusum_step(step.name, n, x_columns)
    elif kind == "cusumsq":
        native_step = cusumsq_step(step.name, n, x_columns)
    else:
        native_step = chow_step(step.name, n, x_columns, int(step.kwargs["t_break"]))
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
            target_offset=n,
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
    kind_codes = {"mean": 0, "covariance": 1, "second_moment": 2}
    kernel_codes = {"bartlett": 0, "parzen": 1, "qs": 2}
    bandwidth_modes = {None: 3, "andrews": 2, "wooldridge": 1, "auto": 3}
    try:
        kind_code = kind_codes[step.kwargs["kind"]]
        kernel_code = kernel_codes[step.kwargs["kernel"]]
    except KeyError as exc:
        raise ValueError("Unsupported native Wald configuration.") from exc
    bandwidth = step.kwargs["bandwidth"]
    if isinstance(bandwidth, bool):
        raise ValueError("Wald bandwidth must be an integer, mode, or None.")
    if isinstance(bandwidth, int):
        bandwidth_mode = 0
        manual_bandwidth = bandwidth
    else:
        try:
            bandwidth_mode = bandwidth_modes[bandwidth]
        except KeyError as exc:
            raise ValueError("Unsupported native Wald bandwidth mode.") from exc
        manual_bandwidth = 0
    target = _wald_target(step.kwargs["target"], q, kind_code)
    return wald_step(
        step.name,
        target,
        n,
        q,
        manual_bandwidth,
        kernel_code,
        bandwidth_mode,
        kind_code,
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


def _wald_target(value: object, q: int, kind: int) -> NDF:
    target = _array_f64(np.asarray(value, dtype=np.float64))
    if kind == 0:
        if target.ndim != 1 or target.shape[0] != q:
            raise ValueError("Wald mean target must have one value per source column.")
    elif target.ndim != 2 or target.shape != (q, q):
        raise ValueError("Wald matrix target must match the source column count.")
    return target
