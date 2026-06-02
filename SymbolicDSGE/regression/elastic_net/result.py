from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..result import RegressionResult

NDF = NDArray[float64]
NDI = NDArray[np.int64]


@dataclass(frozen=True)
class ElasticNetResult(RegressionResult):
    alpha: float64
    l1_ratio: float64
    effective_dof: float64
    intercept: bool = True
    alpha_grid: NDF | None = field(default=None, repr=False)
    coefficient_path: NDF | None = field(default=None, repr=False)
    objective_trace: NDF | None = field(default=None, repr=False)
    rss_trace: NDF | None = field(default=None, repr=False)
    effective_dof_trace: NDF | None = field(default=None, repr=False)
    status_trace: NDI | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        object.__setattr__(self, "alpha", float64(self.alpha))
        object.__setattr__(self, "l1_ratio", float64(self.l1_ratio))
        object.__setattr__(self, "effective_dof", float64(self.effective_dof))
        object.__setattr__(self, "intercept", bool(self.intercept))

        if self.l1_ratio < 0 or self.l1_ratio > 1:
            raise ValueError("l1_ratio must be between 0 and 1.")

        alpha_grid = _as_optional_vector(self.alpha_grid, "alpha_grid")
        coefficient_path = _as_optional_matrix(
            self.coefficient_path,
            "coefficient_path",
        )
        objective_trace = _as_optional_vector(
            self.objective_trace,
            "objective_trace",
        )
        rss_trace = _as_optional_vector(self.rss_trace, "rss_trace")
        effective_dof_trace = _as_optional_vector(
            self.effective_dof_trace,
            "effective_dof_trace",
        )
        status_trace = _as_optional_int_vector(self.status_trace, "status_trace")

        if alpha_grid is not None:
            for name, trace in (
                ("objective_trace", objective_trace),
                ("rss_trace", rss_trace),
                ("effective_dof_trace", effective_dof_trace),
                ("status_trace", status_trace),
            ):
                if trace is not None and trace.shape != alpha_grid.shape:
                    raise ValueError(f"{name} must match alpha_grid shape.")

            if coefficient_path is not None and coefficient_path.shape != (
                alpha_grid.shape[0],
                self.k,
            ):
                raise ValueError(
                    "coefficient_path must have shape (n_alpha, n_coefficients)."
                )
        elif (
            coefficient_path is not None
            or objective_trace is not None
            or rss_trace is not None
            or effective_dof_trace is not None
            or status_trace is not None
        ):
            raise ValueError("alpha_grid is required when grid traces are supplied.")

        object.__setattr__(self, "alpha_grid", alpha_grid)
        object.__setattr__(self, "coefficient_path", coefficient_path)
        object.__setattr__(self, "objective_trace", objective_trace)
        object.__setattr__(self, "rss_trace", rss_trace)
        object.__setattr__(self, "effective_dof_trace", effective_dof_trace)
        object.__setattr__(self, "status_trace", status_trace)

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
    def l2_norm_sq(self) -> float64:
        return float64(np.dot(self.penalized_coefficients, self.penalized_coefficients))

    @cached_property
    def l1_penalty(self) -> float64:
        return float64(self.alpha * self.l1_ratio * self.l1_norm)

    @cached_property
    def l2_penalty(self) -> float64:
        return float64(
            0.5 * self.alpha * (float64(1.0) - self.l1_ratio) * self.l2_norm_sq
        )

    @cached_property
    def penalty(self) -> float64:
        return float64(self.l1_penalty + self.l2_penalty)


def _as_optional_vector(value: NDF | None, name: str) -> NDF | None:
    if value is None:
        return None
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return arr


def _as_optional_int_vector(value: NDI | None, name: str) -> NDI | None:
    if value is None:
        return None
    arr = np.ascontiguousarray(value, dtype=np.int64)
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
