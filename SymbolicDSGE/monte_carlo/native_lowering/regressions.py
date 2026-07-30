"""Native regression step lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._ckernels.monte_carlo._runner import (
    NativeStep,
    elastic_net_gs_step,
    elastic_net_step,
    lasso_gs_step,
    lasso_step,
    ols_step,
    ridge_gs_step,
    ridge_step,
)
from ..allocation import BufferPlan
from ..mc_constructs import MCStep
from .core import (
    FloatInputBinding,
    _fill_binding,
    _selected_shape,
    _source_binding,
)

if TYPE_CHECKING:
    from .core import RegressionResultSpec


def lower_regression_step(
    step: MCStep,
    source_indices: tuple[int, ...],
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
) -> tuple[NativeStep, tuple[FloatInputBinding, ...]]:
    """Compile design and response transfers for one native regression."""
    n, p, intercept, _ = _resolve_regression_shape(step, source_indices, steps, plan)
    x_columns = p - int(intercept)
    bindings: list[FloatInputBinding] = []
    if intercept:
        bindings.append(_fill_binding(n, 0, p, 1.0))
    bindings.extend(
        (
            _source_binding(
                source_indices[1],
                steps,
                plan,
                step.source_args[1],
                target_offset=int(intercept),
                target_row_stride=p,
            ),
            _source_binding(
                source_indices[0],
                steps,
                plan,
                step.source_args[0],
                target_offset=n * p,
                target_row_stride=1,
            ),
        )
    )
    return _native_step(step, n, p, intercept), tuple(bindings)


def regression_result_spec(
    step: MCStep,
    source_indices: tuple[int, ...],
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
) -> "RegressionResultSpec":
    """Resolve result metadata from the same source layout as the native ABI."""
    from .core import RegressionResultSpec

    n, p, _, variables = _resolve_regression_shape(step, source_indices, steps, plan)
    return RegressionResultSpec(
        name=step.name,
        kind=str(step.kwargs["kind"]),
        variables=variables,
        n=n,
        k=p,
    )


def _resolve_regression_shape(
    step: MCStep,
    source_indices: tuple[int, ...],
    steps: tuple[MCStep, ...],
    plan: BufferPlan,
) -> tuple[int, int, bool, tuple[str, ...]]:
    n, x_columns = _selected_shape(source_indices[1], step.source_args[1], steps, plan)
    y_rows, y_columns = _selected_shape(
        source_indices[0], step.source_args[0], steps, plan
    )
    if y_rows != n or y_columns != 1:
        raise ValueError("Native regression lowering requires a one-column response.")
    intercept = bool(step.kwargs["intercept"])
    p = x_columns + int(intercept)
    raw_variables = step.kwargs["variables"]
    if raw_variables is None:
        variables = tuple(f"x{index}" for index in range(x_columns))
    else:
        variables = tuple(raw_variables)
        if len(variables) != x_columns:
            raise ValueError(
                "Regression variable names must match the number of design columns."
            )
    if intercept:
        variables = ("Intercept", *variables)
    return n, p, intercept, variables


def _native_step(step: MCStep, n: int, p: int, intercept: bool) -> NativeStep:
    kind = step.kwargs["kind"]
    if kind == "ols":
        return ols_step(step.name, n, p, intercept)
    if kind == "ridge":
        return ridge_step(step.name, n, p, float(step.kwargs["alpha"]), intercept)
    if kind == "ridge_gs":
        return ridge_gs_step(
            step.name,
            n,
            p,
            float(step.kwargs["start"]),
            float(step.kwargs["stop"]),
            int(step.kwargs["num"]),
            _criterion_code(step.kwargs.get("criterion", "aic")),
            intercept,
        )
    if kind == "lasso":
        return lasso_step(
            step.name,
            n,
            p,
            float(step.kwargs["alpha"]),
            int(step.kwargs.get("max_iter", 1000)),
            float(step.kwargs.get("tol", 1e-10)),
            intercept,
        )
    if kind == "lasso_gs":
        return lasso_gs_step(
            step.name,
            n,
            p,
            float(step.kwargs["start"]),
            float(step.kwargs["stop"]),
            int(step.kwargs["num"]),
            int(step.kwargs.get("max_iter", 1000)),
            float(step.kwargs.get("tol", 1e-10)),
            intercept,
        )
    if kind == "elastic_net":
        return elastic_net_step(
            step.name,
            n,
            p,
            float(step.kwargs["alpha"]),
            float(step.kwargs["l1_ratio"]),
            int(step.kwargs.get("max_iter", 1000)),
            float(step.kwargs.get("tol", 1e-10)),
            intercept,
        )
    if kind == "elastic_net_gs":
        return elastic_net_gs_step(
            step.name,
            n,
            p,
            float(step.kwargs["start"]),
            float(step.kwargs["stop"]),
            int(step.kwargs["num"]),
            float(step.kwargs["l1_ratio"]),
            _criterion_code(step.kwargs.get("criterion", "loss")),
            int(step.kwargs.get("max_iter", 1000)),
            float(step.kwargs.get("tol", 1e-10)),
            intercept,
        )
    raise ValueError(f"Unsupported native regression kind: {kind!r}.")


def _criterion_code(value: object) -> int:
    codes = {"aic": 1, "bic": 2, "loss": 3}
    try:
        return codes[str(value)]
    except KeyError as exc:
        raise ValueError(
            "Regression criterion must be 'aic', 'bic', or 'loss'."
        ) from exc
