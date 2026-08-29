"""Dependency-neutral native-lowering types and input-binding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

import numpy as np
from numpy.typing import NDArray

from ..._diag_tests.distributions import (
    DistributionParameter,
    PvalMethod,
    ReferenceDistribution,
)
from ...core.solved_model import SolvedModel
from ..allocation import BufferPlan, FieldLayout
from ..mc_constructs import MCStep, SourceArgs

_FILL_COLUMNS = np.zeros(1, dtype=np.int64)
_STATIC_SOURCE_STEP = -2

NDF = NDArray[np.float64]
NDI = NDArray[np.int64]


@dataclass(frozen=True)
class FloatInputBinding:
    """One resolved transfer into a consumer's flat float input lane."""

    source_step_idx: int
    source_offset: int
    source_row_stride: int
    row_start: int
    n_rows: int
    columns: NDI
    target_offset: int
    target_row_stride: int
    fill_value: float = 0.0
    static_values: NDF | None = None
    static_rep_stride: int = 0
    static_batched: bool = False


@dataclass(frozen=True)
class TestResultSpec:
    """Resolved result metadata declared by a native diagnostic step."""

    name: str
    dist: ReferenceDistribution
    df: DistributionParameter | tuple[DistributionParameter, ...]
    pval_method: PvalMethod
    alpha: np.float64 | float = 0.05


@dataclass(frozen=True)
class RegressionResultSpec:
    """Resolved semantic metadata for one native regression result."""

    name: str
    kind: str
    variables: tuple[str, ...]
    n: int
    k: int


def _supplied(kwargs: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {name: kwargs[name] for name in names if kwargs.get(name) is not None}


def _model_params(model: SolvedModel) -> NDF:
    parameters = model.config.calibration.parameters
    return _flat_f64(
        np.asarray([parameters[param] for param in model.compiled.calib_params])
    )


def _flat_f64(values: NDF) -> NDF:
    return values.reshape(-1)


def _selected_shape(
    source_idx: int,
    source: SourceArgs,
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
) -> tuple[int, int]:
    source_step = steps[source_idx]
    layout = plan[source_step.name].out_fields[source.field]
    if len(layout.shape) != 2:
        raise ValueError(
            f"Native source {source_step.name!r}.{source.field!r} must be 2D."
        )
    n_rows, n_columns = layout.shape
    return n_rows - source.row_start, _selected_columns(source, n_columns).shape[0]


def _source_binding(
    source_step_idx: int,
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
    source: SourceArgs,
    *,
    target_offset: int,
    target_row_stride: int,
) -> FloatInputBinding:
    producer = steps[source_step_idx]
    layout = plan[producer.name].out_fields[source.field]
    if layout.dtype != np.dtype(np.float64) or len(layout.shape) != 2:
        raise ValueError(
            f"Native source {producer.name!r}.{source.field!r} must be a 2D float field."
        )
    n_rows, n_columns = layout.shape
    columns = _selected_columns(source, n_columns)
    n_selected_rows = n_rows - source.row_start
    if n_selected_rows < 0:
        raise ValueError("Native source row selection starts past the input.")
    return FloatInputBinding(
        source_step_idx=source_step_idx,
        source_offset=layout.offset,
        source_row_stride=n_columns,
        row_start=source.row_start,
        n_rows=n_selected_rows,
        columns=columns,
        target_offset=target_offset,
        target_row_stride=target_row_stride,
    )


def _fill_binding(
    n_rows: int,
    target_offset: int,
    target_row_stride: int,
    value: float,
) -> FloatInputBinding:
    return FloatInputBinding(
        source_step_idx=-1,
        source_offset=0,
        source_row_stride=0,
        row_start=0,
        n_rows=n_rows,
        columns=_FILL_COLUMNS,
        target_offset=target_offset,
        target_row_stride=target_row_stride,
        fill_value=value,
    )


def _static_binding(
    values: NDF,
    target_offset: int,
    *,
    batched: bool = False,
) -> FloatInputBinding:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if batched:
        if array.ndim < 2 or array.shape[0] == 0:
            raise ValueError("Batched native input requires a non-empty leading axis.")
        row_size = int(np.prod(array.shape[1:], dtype=np.intp))
    else:
        row_size = array.size
    if row_size == 0:
        raise ValueError("Native static input bindings cannot be empty.")
    flattened = np.ascontiguousarray(array.reshape(-1), dtype=np.float64)
    return FloatInputBinding(
        source_step_idx=_STATIC_SOURCE_STEP,
        source_offset=0,
        source_row_stride=row_size,
        row_start=0,
        n_rows=1,
        columns=np.arange(row_size, dtype=np.int64),
        target_offset=target_offset,
        target_row_stride=row_size,
        static_values=flattened,
        static_rep_stride=row_size if batched else 0,
        static_batched=batched,
    )


def _selected_columns(source: SourceArgs, n_columns: int) -> NDI:
    columns = source.column_selector
    if isinstance(columns, slice):
        resolved = np.arange(n_columns, dtype=np.int64)[columns]
    else:
        resolved = np.asarray(columns, dtype=np.int64)
    return np.ascontiguousarray(resolved, dtype=np.int64)


def _check_raw_model_data_layout(
    step: MCStep,
    fields: Mapping[str, FieldLayout],
) -> None:
    expected_offset = 0
    for field in ("states", "observables"):
        value = step.kwargs[field]
        if value is None:
            continue
        layout = fields[field]
        if layout.offset != expected_offset:
            raise ValueError(
                "Raw model data fields must occupy contiguous float output."
            )
        expected_offset += layout.flat_count
