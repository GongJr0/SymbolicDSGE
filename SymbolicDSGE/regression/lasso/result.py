from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..result import RegressionResult

NDF = NDArray[float64]


@dataclass(frozen=True)
class LassoResult(RegressionResult):
    alpha: float64
    effective_dof: float64
    intercept: bool = False
    alpha_grid: NDF | None = field(default=None, repr=False)
    coefficient_path: NDF | None = field(default=None, repr=False)
    objective_trace: NDF | None = field(default=None, repr=False)
    knot_lambdas: NDF | None = field(default=None, repr=False)
    knot_coefficients: NDF | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        object.__setattr__(self, "alpha", float64(self.alpha))
        object.__setattr__(self, "effective_dof", float64(self.effective_dof))
        object.__setattr__(self, "intercept", bool(self.intercept))

        alpha_grid = _as_optional_vector(self.alpha_grid, "alpha_grid")
        objective_trace = _as_optional_vector(
            self.objective_trace,
            "objective_trace",
        )
        knot_lambdas = _as_optional_vector(self.knot_lambdas, "knot_lambdas")
        coefficient_path = _as_optional_matrix(
            self.coefficient_path,
            "coefficient_path",
        )
        knot_coefficients = _as_optional_matrix(
            self.knot_coefficients,
            "knot_coefficients",
        )

        if alpha_grid is not None:
            if (
                objective_trace is not None
                and objective_trace.shape != alpha_grid.shape
            ):
                raise ValueError("objective_trace must match alpha_grid shape.")
            if coefficient_path is not None and coefficient_path.shape != (
                alpha_grid.shape[0],
                self.k,
            ):
                raise ValueError(
                    "coefficient_path must have shape (n_alpha, n_coefficients)."
                )
        elif objective_trace is not None or coefficient_path is not None:
            raise ValueError(
                "alpha_grid is required when objective_trace or coefficient_path "
                "is supplied."
            )

        if knot_lambdas is not None:
            if knot_coefficients is None:
                raise ValueError(
                    "knot_coefficients is required when knot_lambdas is supplied."
                )
            if knot_coefficients.shape != (knot_lambdas.shape[0], self.k):
                raise ValueError(
                    "knot_coefficients must have shape " "(n_knots, n_coefficients)."
                )
        elif knot_coefficients is not None:
            raise ValueError(
                "knot_lambdas is required when knot_coefficients is supplied."
            )

        object.__setattr__(self, "alpha_grid", alpha_grid)
        object.__setattr__(self, "coefficient_path", coefficient_path)
        object.__setattr__(self, "objective_trace", objective_trace)
        object.__setattr__(self, "knot_lambdas", knot_lambdas)
        object.__setattr__(self, "knot_coefficients", knot_coefficients)

    @cached_property
    def penalized_coefficients(self) -> NDF:
        if self.intercept:
            return self.coefficients[1:]
        return self.coefficients

    @cached_property
    def active_mask(self) -> NDArray[np.bool_]:
        return np.asarray(self.penalized_coefficients != 0.0, dtype=bool)

    @cached_property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @cached_property
    def selected_variables(self) -> list[str]:
        variables = self.variables[1:] if self.intercept else self.variables
        return [name for name, active in zip(variables, self.active_mask) if active]

    @cached_property
    def l1_norm(self) -> float64:
        return float64(np.abs(self.penalized_coefficients).sum())

    @cached_property
    def l1_penalty(self) -> float64:
        return float64(self.alpha * self.l1_norm)


def _as_optional_vector(value: NDF | None, name: str) -> NDF | None:
    if value is None:
        return None
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return arr


def _as_optional_matrix(value: NDF | None, name: str) -> NDF | None:
    if value is None:
        return None
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return arr
