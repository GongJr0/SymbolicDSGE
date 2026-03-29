from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Callable, Mapping, Sequence, cast
import re
import warnings

import numba
import numpy as np
import sympy as sp
from numba import njit
from scipy.stats import chi2
from sympy.core.function import AppliedUndef
from sympy import Eq, Expr, Function, Symbol
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from ..core.shock_generators import Shock
from ..core.solved_model import SolvedModel

_GLOBAL_TRANSFORMATIONS = standard_transformations + (convert_xor,)
_FOC_CACHE: dict[
    tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[tuple[str, str], ...],
    ],
    tuple[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], tuple[str, ...]],
] = {}


@dataclass(frozen=True)
class _FocLocalDef:
    kind: str
    symbol: sp.Basic
    expr: Expr


@dataclass(frozen=True)
class DenHaanMarcetResult:
    statistic: float
    df: int
    p_value: float
    critical_value: float
    rejects_null: bool
    mean_moments: np.ndarray
    covariance: np.ndarray
    moments: np.ndarray
    residuals: np.ndarray
    raw_residuals: np.ndarray
    instruments: np.ndarray
    states: np.ndarray
    shock_matrix: np.ndarray | None
    variables: list[str]
    equation_idx: np.ndarray
    instrument_idx: np.ndarray
    include_constant: bool
    burn_in: int
    foc_expressions: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DenHaanMarcetMonteCarloResult:
    rejection_rate: float
    alpha: float
    df: int
    critical_value: float
    statistics: np.ndarray
    p_values: np.ndarray
    rejections: np.ndarray
    raw_residuals: np.ndarray
    variables: list[str]
    equation_idx: np.ndarray
    foc_expressions: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MeasurementMomentResult:
    statistic: float
    df: int
    p_value: float
    critical_value: float
    rejects_null: bool
    mean_moments: np.ndarray
    covariance: np.ndarray
    moments: np.ndarray
    observed: np.ndarray
    predicted_measurements: np.ndarray
    measurement_errors: np.ndarray
    instruments: np.ndarray
    states: np.ndarray
    shock_matrix: np.ndarray | None
    variables: list[str]
    observables: list[str]
    instrument_idx: np.ndarray
    include_constant: bool
    lagged_instruments: bool
    burn_in: int
    n_estimated_params: int


@dataclass(frozen=True)
class _PreparedMeasurementMoments:
    observed: np.ndarray
    predicted_measurements: np.ndarray
    measurement_errors: np.ndarray
    instruments: np.ndarray
    states: np.ndarray
    shock_matrix: np.ndarray | None
    observables: tuple[str, ...]
    instrument_idx: np.ndarray


@njit(cache=True)
def _simulate_linear_states(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    shock_mat: np.ndarray,
) -> np.ndarray:
    T = shock_mat.shape[0]
    n = A.shape[0]
    X = np.zeros((T + 1, n), dtype=np.float64)
    X[0] = x0

    for t in range(T):
        X[t + 1] = A @ X[t] + B @ shock_mat[t]

    return X


@njit(cache=True)
def _build_forward_moments(
    current_states: np.ndarray,
    forward_states: np.ndarray,
    params: np.ndarray,
    objective_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    equation_idx: np.ndarray,
    instrument_idx: np.ndarray,
    include_constant: bool,
    burn_in: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_steps = current_states.shape[0]
    n_obs = n_steps - burn_in
    n_eq = equation_idx.shape[0]
    n_inst = instrument_idx.shape[0] + (1 if include_constant else 0)

    moments = np.empty((n_obs, n_eq * n_inst), dtype=np.float64)
    residuals = np.empty((n_obs, n_eq), dtype=np.float64)
    instruments = np.empty((n_obs, n_inst), dtype=np.float64)

    cur = np.empty((current_states.shape[1],), dtype=np.complex128)
    fwd = np.empty((forward_states.shape[1],), dtype=np.complex128)

    for row, t in enumerate(range(burn_in, n_steps)):
        cur[:] = current_states[t]
        fwd[:] = forward_states[t]
        residual_vec = objective_fn(fwd, cur, params)

        col = 0
        if include_constant:
            instruments[row, 0] = 1.0
            col = 1

        for j in range(instrument_idx.shape[0]):
            instruments[row, col + j] = current_states[t, instrument_idx[j]]

        out_col = 0
        for i in range(n_eq):
            resid = residual_vec[equation_idx[i]].real
            residuals[row, i] = resid

            for j in range(n_inst):
                moments[row, out_col] = instruments[row, j] * resid
                out_col += 1

    return moments, residuals, instruments


@njit(cache=True)
def _build_lagged_foc_moments(
    current_states: np.ndarray,
    forward_states: np.ndarray,
    params: np.ndarray,
    foc_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    instrument_idx: np.ndarray,
    include_constant: bool,
    burn_in: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_steps = current_states.shape[0]
    n_obs = n_steps - burn_in
    n_inst = instrument_idx.shape[0] + (1 if include_constant else 0)

    first_resid = foc_fn(forward_states[burn_in], current_states[burn_in], params)
    n_eq = first_resid.shape[0]

    moments = np.empty((n_obs, n_eq * n_inst), dtype=np.float64)
    residuals = np.empty((n_obs, n_eq), dtype=np.float64)
    instruments = np.empty((n_obs, n_inst), dtype=np.float64)

    for row, t in enumerate(range(burn_in, n_steps)):
        residual_vec = foc_fn(forward_states[t], current_states[t], params)

        col = 0
        if include_constant:
            instruments[row, 0] = 1.0
            col = 1

        for j in range(instrument_idx.shape[0]):
            instruments[row, col + j] = current_states[t, instrument_idx[j]]

        out_col = 0
        for i in range(n_eq):
            resid = residual_vec[i]
            residuals[row, i] = resid

            for j in range(n_inst):
                moments[row, out_col] = instruments[row, j] * resid
                out_col += 1

    return moments, residuals, instruments


class DenHaanMarcet:
    def __init__(
        self,
        solved: SolvedModel,
        focs: Sequence[str] | None = None,
        foc_locals: Mapping[str, str] | None = None,
    ) -> None:
        self.solved = solved
        self._objective = solved.compiled.construct_objective_vector_func()
        self._focs = tuple(focs) if focs is not None else None
        self._foc_locals = dict(foc_locals) if foc_locals is not None else None
        self._t = Symbol("t", integer=True)
        self._A_float = np.ascontiguousarray(self.solved.A, dtype=np.float64)

    def one_sample(
        self,
        T: int,
        shocks: (
            Mapping[str, Callable[[float | np.ndarray], np.ndarray] | np.ndarray] | None
        ) = None,
        *,
        focs: Sequence[str] | None = None,
        foc_locals: Mapping[str, str] | None = None,
        shock_scale: float = 1.0,
        x0: np.ndarray | None = None,
        equation_idx: Sequence[int] | None = None,
        instrument_idx: Sequence[int | str] | None = None,
        include_constant: bool = True,
        burn_in: int = 0,
        alpha: float = 0.05,
        use_conditional_expectation: bool = True,
    ) -> DenHaanMarcetResult:
        if T <= 0:
            raise ValueError("T must be positive.")
        state0 = self._prepare_initial_state(x0)
        shock_mat = self._prepare_shock_matrix(T, shocks, shock_scale)
        states = _simulate_linear_states(
            self._A_float,
            np.ascontiguousarray(self.solved.B, dtype=np.float64),
            state0,
            shock_mat,
        )
        return self._evaluate_state_path(
            states,
            shock_matrix=shock_mat,
            focs=focs,
            foc_locals=foc_locals,
            equation_idx=equation_idx,
            instrument_idx=instrument_idx,
            include_constant=include_constant,
            burn_in=burn_in,
            alpha=alpha,
            use_conditional_expectation=use_conditional_expectation,
        )

    def from_state_path(
        self,
        states: np.ndarray,
        *,
        focs: Sequence[str] | None = None,
        foc_locals: Mapping[str, str] | None = None,
        equation_idx: Sequence[int] | None = None,
        instrument_idx: Sequence[int | str] | None = None,
        include_constant: bool = True,
        burn_in: int = 0,
        alpha: float = 0.05,
        use_conditional_expectation: bool = True,
    ) -> DenHaanMarcetResult:
        state_path = self._prepare_state_matrix(states)
        return self._evaluate_state_path(
            state_path,
            shock_matrix=None,
            focs=focs,
            foc_locals=foc_locals,
            equation_idx=equation_idx,
            instrument_idx=instrument_idx,
            include_constant=include_constant,
            burn_in=burn_in,
            alpha=alpha,
            use_conditional_expectation=use_conditional_expectation,
        )

    def monte_carlo(
        self,
        T: int,
        shocks: Mapping[str, Shock],
        *,
        focs: Sequence[str] | None = None,
        foc_locals: Mapping[str, str] | None = None,
        n_rep: int,
        shock_scale: float = 1.0,
        x0: np.ndarray | None = None,
        equation_idx: Sequence[int] | None = None,
        instrument_idx: Sequence[int | str] | None = None,
        include_constant: bool = True,
        burn_in: int = 0,
        alpha: float = 0.05,
        use_conditional_expectation: bool = True,
    ) -> DenHaanMarcetMonteCarloResult:
        if n_rep <= 0:
            raise ValueError("n_rep must be positive.")

        statistics = np.empty((n_rep,), dtype=np.float64)
        p_values = np.empty((n_rep,), dtype=np.float64)
        rejections = np.empty((n_rep,), dtype=bool)
        raw_residuals: np.ndarray | None = None
        variables: list[str] | None = None
        equation_idx_out: np.ndarray | None = None
        foc_expressions: tuple[str, ...] | None = None
        df = None
        critical_value = None

        for rep in range(n_rep):
            rep_shocks = self._clone_mc_shocks(shocks, T, rep)
            result = self.one_sample(
                T,
                rep_shocks,
                focs=focs,
                foc_locals=foc_locals,
                shock_scale=shock_scale,
                x0=x0,
                equation_idx=equation_idx,
                instrument_idx=instrument_idx,
                include_constant=include_constant,
                burn_in=burn_in,
                alpha=alpha,
                use_conditional_expectation=use_conditional_expectation,
            )
            statistics[rep] = result.statistic
            p_values[rep] = result.p_value
            rejections[rep] = result.rejects_null
            if df is None:
                df = result.df
                critical_value = result.critical_value
                variables = list(result.variables)
                equation_idx_out = result.equation_idx.copy()
                foc_expressions = result.foc_expressions
                raw_residuals = np.empty(
                    (n_rep, *result.raw_residuals.shape), dtype=np.float64
                )
            if raw_residuals is not None:
                raw_residuals[rep] = result.raw_residuals

        return DenHaanMarcetMonteCarloResult(
            rejection_rate=float(np.mean(rejections)),
            alpha=alpha,
            df=int(df if df is not None else 0),
            critical_value=float(
                critical_value if critical_value is not None else np.nan
            ),
            statistics=statistics,
            p_values=p_values,
            rejections=rejections,
            raw_residuals=(
                raw_residuals
                if raw_residuals is not None
                else np.empty((0, 0, 0), dtype=np.float64)
            ),
            variables=variables if variables is not None else [],
            equation_idx=(
                equation_idx_out
                if equation_idx_out is not None
                else np.empty((0,), dtype=np.int64)
            ),
            foc_expressions=foc_expressions,
        )

    def measurement_moment_test(
        self,
        y: Mapping[str, Sequence[float] | np.ndarray] | np.ndarray,
        observable: str | Sequence[str],
        *,
        shocks: (
            Mapping[str, Callable[[float | np.ndarray], np.ndarray] | np.ndarray] | None
        ) = None,
        shock_scale: float = 1.0,
        x0: np.ndarray | None = None,
        instrument_idx: Sequence[int | str] | None = None,
        include_constant: bool = True,
        lagged_instruments: bool = False,
        burn_in: int = 0,
        alpha: float = 0.05,
        n_estimated_params: int,
        measurement_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> MeasurementMomentResult | list[MeasurementMomentResult]:
        """Run a GMM-style measurement residual orthogonality test.

        This is not a DHM Euler orthogonality test. By default it tests
        E[z_t * e_t] = 0; with `lagged_instruments=True` it tests
        E[z_{t-1} * e_t] = 0.

        The chi-square degrees-of-freedom adjustment `m - p` assumes the same
        free parameters were estimated from the same moment conditions.
        Otherwise, treating `n_estimated_params` as a df correction is heuristic.
        """
        state0 = self._prepare_initial_state(x0)
        prepared = self._prepare_measurement_moments_from_simulation(
            y,
            (observable,) if isinstance(observable, str) else tuple(observable),
            shocks=shocks,
            shock_scale=shock_scale,
            state0=state0,
            instrument_idx=instrument_idx,
            include_constant=include_constant,
            lagged_instruments=lagged_instruments,
            burn_in=burn_in,
            alpha=alpha,
            measurement_fn=measurement_fn,
        )
        n_est, df = self._resolve_adjusted_df(
            prepared.instruments.shape[1],
            n_estimated_params,
            "Measurement moment test",
        )
        results: list[MeasurementMomentResult] = []

        for col, obs_name in enumerate(prepared.observables):
            observed = np.ascontiguousarray(prepared.observed[:, col], dtype=np.float64)
            predicted = np.ascontiguousarray(
                prepared.predicted_measurements[:, col], dtype=np.float64
            )
            errors = np.ascontiguousarray(
                prepared.measurement_errors[:, col], dtype=np.float64
            )
            moments = np.ascontiguousarray(
                prepared.instruments * errors.reshape(-1, 1),
                dtype=np.float64,
            )
            (
                statistic,
                _df,
                p_value,
                critical_value,
                rejects_null,
                mean_moments,
                covariance,
            ) = self._moment_summary(
                moments,
                alpha,
                df_override=df,
                test_name="Measurement moment test",
            )
            results.append(
                MeasurementMomentResult(
                    statistic=statistic,
                    df=_df,
                    p_value=p_value,
                    critical_value=critical_value,
                    rejects_null=rejects_null,
                    mean_moments=mean_moments,
                    covariance=covariance,
                    moments=moments,
                    observed=observed,
                    predicted_measurements=predicted,
                    measurement_errors=errors,
                    instruments=prepared.instruments,
                    states=prepared.states,
                    shock_matrix=prepared.shock_matrix,
                    variables=list(self.solved.compiled.var_names),
                    observables=[obs_name],
                    instrument_idx=prepared.instrument_idx,
                    include_constant=include_constant,
                    lagged_instruments=lagged_instruments,
                    burn_in=burn_in,
                    n_estimated_params=n_est,
                )
            )

        if isinstance(observable, str):
            return results[0]
        return results

    def measurement_moment_test_from_state_path(
        self,
        states: np.ndarray,
        y: Mapping[str, Sequence[float] | np.ndarray] | np.ndarray,
        observable: str | Sequence[str],
        *,
        instrument_idx: Sequence[int | str] | None = None,
        include_constant: bool = True,
        lagged_instruments: bool = False,
        burn_in: int = 0,
        alpha: float = 0.05,
        n_estimated_params: int,
        measurement_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> MeasurementMomentResult | list[MeasurementMomentResult]:
        """Run the measurement residual orthogonality test on provided states.

        `states` may be either an already aligned `(T, n)` state matrix or a
        simulation-style `(T+1, n)` path including an initial condition row.
        """
        prepared = self._prepare_measurement_moments_from_state_path(
            states,
            y,
            (observable,) if isinstance(observable, str) else tuple(observable),
            instrument_idx=instrument_idx,
            include_constant=include_constant,
            lagged_instruments=lagged_instruments,
            burn_in=burn_in,
            alpha=alpha,
            measurement_fn=measurement_fn,
            shock_matrix=None,
        )
        n_est, df = self._resolve_adjusted_df(
            prepared.instruments.shape[1],
            n_estimated_params,
            "Measurement moment test",
        )
        results: list[MeasurementMomentResult] = []

        for col, obs_name in enumerate(prepared.observables):
            observed = np.ascontiguousarray(prepared.observed[:, col], dtype=np.float64)
            predicted = np.ascontiguousarray(
                prepared.predicted_measurements[:, col], dtype=np.float64
            )
            errors = np.ascontiguousarray(
                prepared.measurement_errors[:, col], dtype=np.float64
            )
            moments = np.ascontiguousarray(
                prepared.instruments * errors.reshape(-1, 1),
                dtype=np.float64,
            )
            (
                statistic,
                _df,
                p_value,
                critical_value,
                rejects_null,
                mean_moments,
                covariance,
            ) = self._moment_summary(
                moments,
                alpha,
                df_override=df,
                test_name="Measurement moment test",
            )
            results.append(
                MeasurementMomentResult(
                    statistic=statistic,
                    df=_df,
                    p_value=p_value,
                    critical_value=critical_value,
                    rejects_null=rejects_null,
                    mean_moments=mean_moments,
                    covariance=covariance,
                    moments=moments,
                    observed=observed,
                    predicted_measurements=predicted,
                    measurement_errors=errors,
                    instruments=prepared.instruments,
                    states=prepared.states,
                    shock_matrix=None,
                    variables=list(self.solved.compiled.var_names),
                    observables=[obs_name],
                    instrument_idx=prepared.instrument_idx,
                    include_constant=include_constant,
                    lagged_instruments=lagged_instruments,
                    burn_in=burn_in,
                    n_estimated_params=n_est,
                )
            )

        if isinstance(observable, str):
            return results[0]
        return results

    def joint_measurement_moment_test(
        self,
        y: Mapping[str, Sequence[float] | np.ndarray] | np.ndarray,
        observables: Sequence[str] | None = None,
        *,
        shocks: (
            Mapping[str, Callable[[float | np.ndarray], np.ndarray] | np.ndarray] | None
        ) = None,
        shock_scale: float = 1.0,
        x0: np.ndarray | None = None,
        instrument_idx: Sequence[int | str] | None = None,
        include_constant: bool = True,
        lagged_instruments: bool = False,
        burn_in: int = 0,
        alpha: float = 0.05,
        n_estimated_params: int,
        measurement_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> MeasurementMomentResult:
        """Run the stacked joint GMM-style measurement residual test."""
        state0 = self._prepare_initial_state(x0)
        prepared = self._prepare_measurement_moments_from_simulation(
            y,
            observables,
            shocks=shocks,
            shock_scale=shock_scale,
            state0=state0,
            instrument_idx=instrument_idx,
            include_constant=include_constant,
            lagged_instruments=lagged_instruments,
            burn_in=burn_in,
            alpha=alpha,
            measurement_fn=measurement_fn,
        )
        n_est, df = self._resolve_adjusted_df(
            prepared.instruments.shape[1] * len(prepared.observables),
            n_estimated_params,
            "Joint measurement moment test",
        )
        moments = self._stack_measurement_moments(
            prepared.instruments,
            prepared.measurement_errors,
        )
        (
            statistic,
            _df,
            p_value,
            critical_value,
            rejects_null,
            mean_moments,
            covariance,
        ) = self._moment_summary(
            moments,
            alpha,
            df_override=df,
            test_name="Joint measurement moment test",
        )
        return MeasurementMomentResult(
            statistic=statistic,
            df=_df,
            p_value=p_value,
            critical_value=critical_value,
            rejects_null=rejects_null,
            mean_moments=mean_moments,
            covariance=covariance,
            moments=moments,
            observed=prepared.observed,
            predicted_measurements=prepared.predicted_measurements,
            measurement_errors=prepared.measurement_errors,
            instruments=prepared.instruments,
            states=prepared.states,
            shock_matrix=prepared.shock_matrix,
            variables=list(self.solved.compiled.var_names),
            observables=list(prepared.observables),
            instrument_idx=prepared.instrument_idx,
            include_constant=include_constant,
            lagged_instruments=lagged_instruments,
            burn_in=burn_in,
            n_estimated_params=n_est,
        )

    def joint_measurement_moment_test_from_state_path(
        self,
        states: np.ndarray,
        y: Mapping[str, Sequence[float] | np.ndarray] | np.ndarray,
        observables: Sequence[str] | None = None,
        *,
        instrument_idx: Sequence[int | str] | None = None,
        include_constant: bool = True,
        lagged_instruments: bool = False,
        burn_in: int = 0,
        alpha: float = 0.05,
        n_estimated_params: int,
        measurement_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> MeasurementMomentResult:
        """Run the stacked joint measurement residual test on provided states."""
        prepared = self._prepare_measurement_moments_from_state_path(
            states,
            y,
            observables,
            instrument_idx=instrument_idx,
            include_constant=include_constant,
            lagged_instruments=lagged_instruments,
            burn_in=burn_in,
            alpha=alpha,
            measurement_fn=measurement_fn,
            shock_matrix=None,
        )
        n_est, df = self._resolve_adjusted_df(
            prepared.instruments.shape[1] * len(prepared.observables),
            n_estimated_params,
            "Joint measurement moment test",
        )
        moments = self._stack_measurement_moments(
            prepared.instruments,
            prepared.measurement_errors,
        )
        (
            statistic,
            _df,
            p_value,
            critical_value,
            rejects_null,
            mean_moments,
            covariance,
        ) = self._moment_summary(
            moments,
            alpha,
            df_override=df,
            test_name="Joint measurement moment test",
        )
        return MeasurementMomentResult(
            statistic=statistic,
            df=_df,
            p_value=p_value,
            critical_value=critical_value,
            rejects_null=rejects_null,
            mean_moments=mean_moments,
            covariance=covariance,
            moments=moments,
            observed=prepared.observed,
            predicted_measurements=prepared.predicted_measurements,
            measurement_errors=prepared.measurement_errors,
            instruments=prepared.instruments,
            states=prepared.states,
            shock_matrix=None,
            variables=list(self.solved.compiled.var_names),
            observables=list(prepared.observables),
            instrument_idx=prepared.instrument_idx,
            include_constant=include_constant,
            lagged_instruments=lagged_instruments,
            burn_in=burn_in,
            n_estimated_params=n_est,
        )

    def _resolve_focs(self, focs: Sequence[str] | None) -> tuple[str, ...] | None:
        if focs is not None:
            return tuple(focs)
        return self._focs

    def _resolve_foc_locals(
        self,
        foc_locals: Mapping[str, str] | None,
    ) -> dict[str, str] | None:
        if self._foc_locals is None and foc_locals is None:
            return None

        resolved: dict[str, str] = {}
        if self._foc_locals is not None:
            resolved.update(self._foc_locals)
        if foc_locals is not None:
            resolved.update(foc_locals)
        return resolved

    def _prepare_state_matrix(self, states: np.ndarray) -> np.ndarray:
        state_path = np.asarray(states, dtype=np.float64)
        if state_path.ndim != 2:
            raise ValueError("states must be a 2D array of consecutive state vectors.")
        n_var = len(self.solved.compiled.var_names)
        if state_path.shape[0] < 2:
            raise ValueError("states must contain at least two consecutive periods.")
        if state_path.shape[1] != n_var:
            raise ValueError(
                f"states must have {n_var} columns in compiled variable order; "
                f"got {state_path.shape[1]}."
            )
        return np.ascontiguousarray(state_path, dtype=np.float64)

    def _prepare_initial_state(self, x0: np.ndarray | None) -> np.ndarray:
        n = self.solved.A.shape[0]
        if x0 is None:
            state0 = np.zeros((n,), dtype=np.float64)
        else:
            state0 = np.array(x0, dtype=np.float64, copy=True).reshape(-1)
            if state0.shape[0] != n:
                raise ValueError(
                    f"Initial state length {state0.shape[0]} does not match model dimension {n}."
                )

        n_state = self.solved.compiled.n_state
        state0[n_state:] = state0[:n_state] @ np.real_if_close(self.solved.policy.f.T)
        return np.ascontiguousarray(state0, dtype=np.float64)

    def _prepare_shock_matrix(
        self,
        T: int,
        shocks: (
            Mapping[str, Callable[[float | np.ndarray], np.ndarray] | np.ndarray] | None
        ),
        shock_scale: float,
    ) -> np.ndarray:
        shock_mat = np.zeros((T, self.solved.B.shape[1]), dtype=np.float64)
        if shocks is None:
            return shock_mat

        for idx, shock_vals in self.solved._shock_unpack(shocks):
            shock_vals = np.asarray(shock_vals, dtype=np.float64).reshape(-1)
            if shock_vals.shape[0] != T:
                raise ValueError(
                    f"Shock array for variable index {idx} must have length {T}."
                )
            shock_mat[:, idx] = float(shock_scale) * shock_vals

        return np.ascontiguousarray(shock_mat, dtype=np.float64)

    def _build_state_pairs(
        self,
        states: np.ndarray,
        use_conditional_expectation: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        current_states = np.ascontiguousarray(states[:-1], dtype=np.float64)
        if use_conditional_expectation:
            forward_states = np.ascontiguousarray(
                current_states @ self._A_float.T,
                dtype=np.float64,
            )
        else:
            forward_states = np.ascontiguousarray(states[1:], dtype=np.float64)
        return current_states, forward_states

    def _evaluate_state_path(
        self,
        states: np.ndarray,
        *,
        shock_matrix: np.ndarray | None,
        focs: Sequence[str] | None,
        foc_locals: Mapping[str, str] | None,
        equation_idx: Sequence[int] | None,
        instrument_idx: Sequence[int | str] | None,
        include_constant: bool,
        burn_in: int,
        alpha: float,
        use_conditional_expectation: bool,
    ) -> DenHaanMarcetResult:
        current_states, forward_states = self._build_state_pairs(
            states, use_conditional_expectation
        )
        n_steps = current_states.shape[0]
        self._validate_moment_controls(n_steps, burn_in, alpha)

        inst_idx = self._resolve_instrument_idx(instrument_idx, include_constant)
        foc_source = self._resolve_focs(focs)
        foc_locals_source = self._resolve_foc_locals(foc_locals)
        if foc_source is not None:
            if equation_idx is not None:
                raise ValueError(
                    "equation_idx is only available when using compiled-model equations without custom FOCs."
                )
            foc_fn, normalized_focs = self._compile_foc_bundle(
                foc_source, foc_locals_source
            )
            moments, residuals, instruments = _build_lagged_foc_moments(
                current_states,
                forward_states,
                self._param_vector_float(),
                foc_fn,
                inst_idx,
                include_constant,
                burn_in,
            )
            eq_idx = np.arange(len(normalized_focs), dtype=np.int64)
            foc_expressions = normalized_focs
        else:
            eq_idx = self._resolve_equation_idx(equation_idx)
            moments, residuals, instruments = _build_forward_moments(
                current_states,
                forward_states,
                self._param_vector_complex(),
                self._objective,
                eq_idx,
                inst_idx,
                include_constant,
                burn_in,
            )
            foc_expressions = None

        (
            statistic,
            df,
            p_value,
            critical_value,
            rejects_null,
            mean_moments,
            covariance,
        ) = self._moment_summary(moments, alpha)

        return DenHaanMarcetResult(
            statistic=statistic,
            df=df,
            p_value=p_value,
            critical_value=critical_value,
            rejects_null=rejects_null,
            mean_moments=mean_moments,
            covariance=covariance,
            moments=moments,
            residuals=residuals,
            raw_residuals=residuals,
            instruments=instruments,
            states=self._aligned_testing_states(states),
            shock_matrix=shock_matrix,
            variables=list(self.solved.compiled.var_names),
            equation_idx=eq_idx,
            instrument_idx=inst_idx,
            include_constant=include_constant,
            burn_in=burn_in,
            foc_expressions=foc_expressions,
        )

    def _prepare_measurement_moments_from_simulation(
        self,
        y: Mapping[str, Sequence[float] | np.ndarray] | np.ndarray,
        observables: Sequence[str] | None,
        *,
        shocks: (
            Mapping[str, Callable[[float | np.ndarray], np.ndarray] | np.ndarray] | None
        ),
        shock_scale: float,
        state0: np.ndarray,
        instrument_idx: Sequence[int | str] | None,
        include_constant: bool,
        lagged_instruments: bool,
        burn_in: int,
        alpha: float,
        measurement_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    ) -> _PreparedMeasurementMoments:
        requested = self._requested_observable_names(observables)
        resolved = self._resolve_observable_names(observables)
        observed = self._prepare_measurement_observed(y, requested, resolved)
        if observed.shape[0] <= 0:
            raise ValueError("y must contain at least one observation.")

        shock_mat = self._prepare_shock_matrix(observed.shape[0], shocks, shock_scale)
        states = _simulate_linear_states(
            self._A_float,
            np.ascontiguousarray(self.solved.B, dtype=np.float64),
            state0,
            shock_mat,
        )
        return self._prepare_measurement_moments_from_state_path(
            states,
            observed,
            resolved,
            instrument_idx=instrument_idx,
            include_constant=include_constant,
            lagged_instruments=lagged_instruments,
            burn_in=burn_in,
            alpha=alpha,
            measurement_fn=measurement_fn,
            shock_matrix=shock_mat,
        )

    def _prepare_measurement_moments_from_state_path(
        self,
        states: np.ndarray,
        y: Mapping[str, Sequence[float] | np.ndarray] | np.ndarray,
        observables: Sequence[str] | None,
        *,
        instrument_idx: Sequence[int | str] | None,
        include_constant: bool,
        lagged_instruments: bool,
        burn_in: int,
        alpha: float,
        measurement_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
        shock_matrix: np.ndarray | None,
    ) -> _PreparedMeasurementMoments:
        requested = self._requested_observable_names(observables)
        resolved = self._resolve_observable_names(observables)
        observed = self._prepare_measurement_observed(y, requested, resolved)
        if observed.shape[0] <= 0:
            raise ValueError("y must contain at least one observation.")

        self._validate_measurement_controls(
            observed.shape[0],
            burn_in,
            alpha,
            lagged_instruments,
        )
        inst_idx = self._resolve_instrument_idx(instrument_idx, include_constant)
        aligned_states = self._prepare_measurement_state_matrix(
            states, observed.shape[0]
        )
        predicted = self._evaluate_measurement_array(
            aligned_states,
            resolved,
            measurement_fn,
        )
        observed_eff, predicted_eff, instruments = self._slice_measurement_samples(
            aligned_states,
            observed,
            predicted,
            inst_idx,
            include_constant,
            burn_in,
            lagged_instruments,
        )
        measurement_errors = np.ascontiguousarray(
            observed_eff - predicted_eff, dtype=np.float64
        )
        return _PreparedMeasurementMoments(
            observed=observed_eff,
            predicted_measurements=predicted_eff,
            measurement_errors=measurement_errors,
            instruments=instruments,
            states=aligned_states,
            shock_matrix=shock_matrix,
            observables=resolved,
            instrument_idx=inst_idx,
        )

    def _resolve_equation_idx(self, equation_idx: Sequence[int] | None) -> np.ndarray:
        n_eq = len(self.solved.compiled.objective_eqs)
        if equation_idx is None:
            out = np.arange(n_eq, dtype=np.int64)
        else:
            out = np.asarray(equation_idx, dtype=np.int64).reshape(-1)

        if out.size == 0:
            raise ValueError("At least one equation index must be selected.")
        if np.any(out < 0) or np.any(out >= n_eq):
            raise IndexError(f"Equation indices must lie in [0, {n_eq}).")
        if np.unique(out).size != out.size:
            raise ValueError("Equation indices must be unique.")
        return np.ascontiguousarray(out, dtype=np.int64)

    def _resolve_instrument_idx(
        self,
        instrument_idx: Sequence[int | str] | None,
        include_constant: bool,
    ) -> np.ndarray:
        n_var = len(self.solved.compiled.var_names)
        if instrument_idx is None:
            out = np.arange(n_var, dtype=np.int64)
        else:
            resolved: list[int] = []
            for entry in instrument_idx:
                if isinstance(entry, str):
                    if entry not in self.solved.compiled.idx:
                        raise KeyError(f"Unknown instrument variable '{entry}'.")
                    resolved.append(self.solved.compiled.idx[entry])
                else:
                    resolved.append(int(entry))
            out = np.asarray(resolved, dtype=np.int64).reshape(-1)

        if out.size == 0 and not include_constant:
            raise ValueError(
                "At least one instrument index is required when include_constant=False."
            )
        if np.any(out < 0) or np.any(out >= n_var):
            raise IndexError(f"Instrument indices must lie in [0, {n_var}).")
        if np.unique(out).size != out.size:
            raise ValueError("Instrument indices must be unique.")
        return np.ascontiguousarray(out, dtype=np.int64)

    def _requested_observable_names(
        self,
        observables: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if observables is None:
            return tuple(self.solved.compiled.observable_names)

        requested = tuple(observables)
        if not requested:
            raise ValueError("At least one observable must be selected.")
        return requested

    def _resolve_observable_names(
        self,
        observables: Sequence[str] | None,
    ) -> tuple[str, ...]:
        requested = self._requested_observable_names(observables)
        if len(set(requested)) != len(requested):
            raise ValueError("Observable list contains duplicates.")

        obs_idx = {
            name: idx for idx, name in enumerate(self.solved.compiled.observable_names)
        }
        missing = [name for name in requested if name not in obs_idx]
        if missing:
            raise KeyError(f"Unknown observables not in compiled model: {missing}")

        return tuple(sorted(requested, key=lambda name: obs_idx[name]))

    def _prepare_measurement_observed(
        self,
        y: Mapping[str, Sequence[float] | np.ndarray] | np.ndarray,
        requested_observables: tuple[str, ...],
        resolved_observables: tuple[str, ...],
    ) -> np.ndarray:
        if isinstance(y, Mapping):
            missing = [name for name in resolved_observables if name not in y]
            if missing:
                raise KeyError(f"Observed data is missing observables: {missing}")

            cols = [
                np.asarray(y[name], dtype=np.float64).reshape(-1)
                for name in resolved_observables
            ]
            lengths = {col.shape[0] for col in cols}
            if len(lengths) != 1:
                raise ValueError(
                    "All observed measurement series must share the same length."
                )
            return np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)

        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim == 1:
            if len(requested_observables) != 1:
                raise ValueError(
                    "y must be a 2D array or mapping when testing multiple observables."
                )
            return np.ascontiguousarray(arr.reshape(-1, 1), dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("y must be a 1D/2D array or mapping keyed by observable.")
        if arr.shape[1] != len(requested_observables):
            raise ValueError(
                f"y has {arr.shape[1]} columns, but {len(requested_observables)} "
                "observable series were requested."
            )

        requested_idx = {name: idx for idx, name in enumerate(requested_observables)}
        perm = [requested_idx[name] for name in resolved_observables]
        return np.ascontiguousarray(arr[:, perm], dtype=np.float64)

    def _evaluate_measurement_array(
        self,
        states: np.ndarray,
        observables: Sequence[str],
        measurement_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    ) -> np.ndarray:
        fn = measurement_fn
        if fn is None:
            fn = self.solved.compiled.construct_measurement_array_func(
                list(observables)
            )
        params = self._param_vector_float()
        out = np.empty((states.shape[0], len(observables)), dtype=np.float64)

        for row in range(states.shape[0]):
            measurement = np.asarray(fn(states[row], params), dtype=np.float64).reshape(
                -1
            )
            if measurement.shape[0] != len(observables):
                raise ValueError(
                    "measurement_fn must return one value per requested observable "
                    f"({len(observables)}), got {measurement.shape[0]}."
                )
            out[row] = measurement

        return np.ascontiguousarray(out, dtype=np.float64)

    def _prepare_measurement_state_matrix(
        self,
        states: np.ndarray,
        n_obs: int,
    ) -> np.ndarray:
        state_path = np.asarray(states, dtype=np.float64)
        if state_path.ndim != 2:
            raise ValueError("states must be a 2D array of state vectors.")

        n_var = len(self.solved.compiled.var_names)
        if state_path.shape[1] != n_var:
            raise ValueError(
                f"states must have {n_var} columns in compiled variable order; "
                f"got {state_path.shape[1]}."
            )
        if state_path.shape[0] == n_obs + 1:
            # For simulated paths, dropping the initial condition aligns
            # states[t] with shock_matrix[t].
            return self._aligned_testing_states(state_path)
        if state_path.shape[0] == n_obs:
            return np.ascontiguousarray(state_path, dtype=np.float64)

        raise ValueError(
            "states must contain either T aligned rows or T+1 rows including "
            "an initial condition."
        )

    def _slice_measurement_samples(
        self,
        states: np.ndarray,
        observed: np.ndarray,
        predicted: np.ndarray,
        instrument_idx: np.ndarray,
        include_constant: bool,
        burn_in: int,
        lagged_instruments: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if lagged_instruments:
            instrument_states = np.ascontiguousarray(
                states[burn_in:-1], dtype=np.float64
            )
            observed_eff = np.ascontiguousarray(
                observed[burn_in + 1 :], dtype=np.float64
            )
            predicted_eff = np.ascontiguousarray(
                predicted[burn_in + 1 :], dtype=np.float64
            )
        else:
            instrument_states = np.ascontiguousarray(states[burn_in:], dtype=np.float64)
            observed_eff = np.ascontiguousarray(observed[burn_in:], dtype=np.float64)
            predicted_eff = np.ascontiguousarray(predicted[burn_in:], dtype=np.float64)

        instruments = self._build_instrument_matrix(
            instrument_states,
            instrument_idx,
            include_constant,
        )
        return observed_eff, predicted_eff, instruments

    def _build_instrument_matrix(
        self,
        states: np.ndarray,
        instrument_idx: np.ndarray,
        include_constant: bool,
    ) -> np.ndarray:
        effective_states = np.ascontiguousarray(states, dtype=np.float64)
        n_obs = effective_states.shape[0]
        n_inst = instrument_idx.shape[0] + (1 if include_constant else 0)
        instruments = np.empty((n_obs, n_inst), dtype=np.float64)
        col = 0

        if include_constant:
            instruments[:, 0] = 1.0
            col = 1

        if instrument_idx.size:
            instruments[:, col:] = effective_states[:, instrument_idx]

        return np.ascontiguousarray(instruments, dtype=np.float64)

    def _stack_measurement_moments(
        self,
        instruments: np.ndarray,
        measurement_errors: np.ndarray,
    ) -> np.ndarray:
        n_obs = instruments.shape[0]
        n_inst = instruments.shape[1]
        n_meas = measurement_errors.shape[1]
        moments = np.empty((n_obs, n_inst * n_meas), dtype=np.float64)

        col = 0
        for meas_idx in range(n_meas):
            block = instruments * measurement_errors[:, meas_idx : meas_idx + 1]
            moments[:, col : col + n_inst] = block
            col += n_inst

        return np.ascontiguousarray(moments, dtype=np.float64)

    def _aligned_testing_states(self, states: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(states[1:], dtype=np.float64)

    def _validate_moment_controls(
        self,
        n_steps: int,
        burn_in: int,
        alpha: float,
    ) -> None:
        if burn_in < 0 or burn_in >= n_steps:
            raise ValueError(
                f"burn_in must satisfy 0 <= burn_in < {n_steps}; got burn_in={burn_in}."
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")

    def _validate_measurement_controls(
        self,
        n_steps: int,
        burn_in: int,
        alpha: float,
        lagged_instruments: bool,
    ) -> None:
        max_burn_in = n_steps - (2 if lagged_instruments else 1)
        if burn_in < 0 or burn_in > max_burn_in:
            qualifier = " with lagged instruments" if lagged_instruments else ""
            raise ValueError(
                f"burn_in must satisfy 0 <= burn_in <= {max_burn_in} for {n_steps} "
                f"observations{qualifier}; got burn_in={burn_in}."
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1.")

    def _resolve_adjusted_df(
        self,
        n_moments: int,
        n_estimated_params: int,
        test_name: str,
    ) -> tuple[int, int]:
        n_est = int(n_estimated_params)
        if n_est < 0:
            raise ValueError("n_estimated_params must be non-negative.")
        if n_moments <= n_est:
            raise ValueError(
                f"{test_name} requires more moment conditions ({n_moments}) than "
                f"estimated parameters ({n_estimated_params})."
            )
        return n_est, n_moments - n_est

    def _param_vector_complex(self) -> np.ndarray:
        params = np.array(
            [
                self.solved.compiled.config.calibration.parameters[p]
                for p in self.solved.compiled.calib_params
            ],
            dtype=np.complex128,
        )
        return np.ascontiguousarray(params, dtype=np.complex128)

    def _param_vector_float(self) -> np.ndarray:
        params = np.array(
            [
                float(self.solved.compiled.config.calibration.parameters[p])
                for p in self.solved.compiled.calib_params
            ],
            dtype=np.float64,
        )
        return np.ascontiguousarray(params, dtype=np.float64)

    def _moment_summary(
        self,
        moments: np.ndarray,
        alpha: float,
        *,
        df_override: int | None = None,
        test_name: str = "DHM",
    ) -> tuple[float, int, float, float, bool, np.ndarray, np.ndarray]:
        n_obs = moments.shape[0]
        if n_obs <= 1:
            raise ValueError(
                f"{test_name} requires at least two effective observations after burn-in."
            )

        mean_moments = moments.mean(axis=0)
        centered = moments - mean_moments
        covariance = centered.T @ centered / n_obs

        # Ridge regularize covariance with an absolute floor so the stabilizer
        # stays effective even when the empirical covariance is nearly zero.
        cov_trace = np.trace(covariance)
        m = moments.shape[1]
        base = cov_trace / m
        lam = max(1e-8 * base, 1e-10)
        covariance += lam * np.eye(m)

        stat = float(n_obs * mean_moments @ np.linalg.solve(covariance, mean_moments))
        df = moments.shape[1] if df_override is None else int(df_override)
        if df <= 0:
            raise ValueError(
                f"{test_name} requires strictly positive degrees of freedom."
            )
        critical_value = float(chi2.ppf(1.0 - alpha, df))
        p_value = float(chi2.sf(stat, df))
        return (
            stat,
            df,
            p_value,
            critical_value,
            stat > critical_value,
            mean_moments,
            covariance,
        )

    def _clone_mc_shocks(
        self,
        shocks: Mapping[str, Shock],
        T: int,
        rep_idx: int,
    ) -> dict[str, Callable[[float | np.ndarray], np.ndarray]]:
        out: dict[str, Callable[[float | np.ndarray], np.ndarray]] = {}

        for name, shock in shocks.items():
            if not isinstance(shock, Shock):
                raise TypeError(
                    "Monte Carlo DHM requires Shock instances for every shock specification."
                )
            if shock.T != T:
                raise ValueError(
                    f"Shock '{name}' has T={shock.T}, but DHM was called with T={T}."
                )
            if ("," in name) != shock.multivar:
                raise ValueError(
                    f"Shock '{name}' must set multivar={',' in name} to match its specification."
                )
            if shock.shock_arr is not None:
                raise ValueError(
                    "Monte Carlo DHM requires generator-style Shock instances, not shock_arr-backed shocks."
                )

            cloned_seed = None if shock.seed is None else int(shock.seed) + rep_idx
            cloned = Shock(
                T=shock.T,
                dist=shock.dist,
                multivar=shock.multivar,
                seed=cloned_seed,
                dist_args=shock.dist_args,
                dist_kwargs=shock.dist_kwargs.copy(),
            )
            out[name] = cloned.shock_generator()

        return out

    def _compile_foc_bundle(
        self,
        focs: tuple[str, ...],
        foc_locals: Mapping[str, str] | None,
    ) -> tuple[
        Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], tuple[str, ...]
    ]:
        var_key = tuple(self.solved.compiled.var_names)
        param_key = tuple(p.name for p in self.solved.compiled.calib_params)
        locals_key = tuple(foc_locals.items()) if foc_locals is not None else ()
        cache_key = (var_key, param_key, focs, locals_key)
        cached = _FOC_CACHE.get(cache_key)
        if cached is not None:
            return cached

        parsed = self._parse_and_normalize_focs(focs, foc_locals)
        compiled = self._build_foc_vector_func(parsed)
        out = (compiled, tuple(str(expr) for expr in parsed))
        _FOC_CACHE[cache_key] = out
        return out

    def _parse_and_normalize_focs(
        self,
        focs: tuple[str, ...],
        foc_locals: Mapping[str, str] | None,
    ) -> list[Expr]:
        if not focs:
            raise ValueError("At least one FOC expression must be provided.")

        local_dict, var_funcs, param_syms, shock_syms, local_defs = (
            self._foc_parse_context(foc_locals)
        )
        parsed: list[Expr] = []
        for foc in focs:
            expr = self._parse_foc_text(foc, local_dict, tuple(var_funcs.keys()))
            expr = self._expand_foc_locals(expr, local_defs)
            self._validate_expression_symbols(
                expr,
                var_funcs,
                param_syms,
                shock_syms,
                require_time_vars=True,
            )
            parsed.append(self._normalize_foc_timing(expr, var_funcs))
        return parsed

    def _foc_parse_context(
        self,
        foc_locals: Mapping[str, str] | None,
    ) -> tuple[
        dict[str, Any],
        dict[str, Function],
        dict[str, Symbol],
        dict[str, Symbol],
        dict[str, _FocLocalDef],
    ]:
        conf = self.solved.config
        var_funcs = {v.__name__: v for v in conf.variables}
        param_syms = {p.name: p for p in conf.parameters}
        shock_syms = {s.name: s for s in conf.shock_map.keys()}
        local_dict: dict[str, Any] = {
            "t": self._t,
            **var_funcs,
            **param_syms,
            **shock_syms,
        }
        local_defs = self._parse_foc_locals(
            local_dict,
            var_funcs,
            param_syms,
            shock_syms,
            foc_locals,
        )
        return local_dict, var_funcs, param_syms, shock_syms, local_defs

    def _parse_foc_locals(
        self,
        local_dict: dict[str, Any],
        var_funcs: dict[str, Function],
        param_syms: dict[str, Symbol],
        shock_syms: dict[str, Symbol],
        foc_locals: Mapping[str, str] | None,
    ) -> dict[str, _FocLocalDef]:
        if not foc_locals:
            return {}

        defs: dict[str, _FocLocalDef] = {}
        for raw_key, raw_expr in foc_locals.items():
            key = raw_key.strip()
            if not key:
                raise ValueError("FOC local names must be non-empty.")

            func_match = re.fullmatch(r"([A-Za-z_]\w*)\(\s*t\s*\)", key)
            if func_match is not None:
                name = func_match.group(1)
                if name in local_dict or name in defs:
                    raise ValueError(
                        f"FOC local '{name}' collides with an existing model symbol."
                    )
                rhs = self._parse_foc_text(
                    raw_expr, local_dict, tuple(var_funcs.keys())
                )
                rhs = self._expand_foc_locals(rhs, defs)
                self._validate_expression_symbols(
                    rhs,
                    var_funcs,
                    param_syms,
                    shock_syms,
                    require_time_vars=False,
                )
                alias_fun = sp.Function(name)
                defs[name] = _FocLocalDef(kind="function", symbol=alias_fun, expr=rhs)
                local_dict[name] = alias_fun
                continue

            if not re.fullmatch(r"[A-Za-z_]\w*", key):
                raise ValueError(
                    f"Unsupported FOC local name {raw_key!r}. "
                    "Use either 'name' or 'name(t)'."
                )
            if key in local_dict or key in defs:
                raise ValueError(
                    f"FOC local '{key}' collides with an existing model symbol."
                )

            rhs = self._parse_foc_text(raw_expr, local_dict, tuple(var_funcs.keys()))
            rhs = self._expand_foc_locals(rhs, defs)
            self._validate_expression_symbols(
                rhs,
                var_funcs,
                param_syms,
                shock_syms,
                require_time_vars=False,
            )
            alias_sym = Symbol(key)
            defs[key] = _FocLocalDef(kind="symbol", symbol=alias_sym, expr=rhs)
            local_dict[key] = alias_sym

        return defs

    def _parse_foc_text(
        self,
        foc: str,
        local_dict: dict[str, Any],
        var_names: tuple[str, ...],
    ) -> Expr:
        try:
            if "=" in foc:
                parts = [p.strip() for p in foc.split("=", maxsplit=2)]
                if len(parts) != 2:
                    raise ValueError(
                        f"FOC equations must contain exactly one '=': {foc!r}"
                    )
                lhs = parse_expr(
                    parts[0],
                    local_dict=local_dict,
                    evaluate=False,
                    transformations=_GLOBAL_TRANSFORMATIONS,
                )
                rhs = parse_expr(
                    parts[1],
                    local_dict=local_dict,
                    evaluate=False,
                    transformations=_GLOBAL_TRANSFORMATIONS,
                )
                expr = sp.simplify(lhs - rhs)
            else:
                expr = parse_expr(
                    foc,
                    local_dict=local_dict,
                    evaluate=False,
                    transformations=_GLOBAL_TRANSFORMATIONS,
                )
        except Exception as exc:
            bare_vars = [
                name
                for name in var_names
                if re.search(rf"\b{re.escape(name)}\b(?!\s*\()", foc)
            ]
            if bare_vars:
                raise ValueError(
                    "Model variables must be written with an explicit time index, "
                    f"for example '{bare_vars[0]}(t)'."
                ) from exc
            raise ValueError(f"Unable to parse FOC expression {foc!r}: {exc}") from exc

        if isinstance(expr, sp.FunctionClass):
            raise ValueError(
                "Model variables must be written with an explicit time index, "
                f"for example '{expr.__name__}(t)'."
            )
        if isinstance(expr, Eq):
            expr = sp.simplify(expr.lhs - expr.rhs)
        if not isinstance(expr, Expr):
            raise TypeError(f"FOC is not a valid SymPy expression: {foc!r}")
        return expr

    def _expand_foc_locals(
        self,
        expr: Expr,
        local_defs: Mapping[str, _FocLocalDef],
    ) -> Expr:
        if not local_defs:
            return expr

        expanded = expr
        for _ in range(len(local_defs) + 1):
            replacements: dict[sp.Basic, Expr] = {}
            for local_def in local_defs.values():
                if local_def.kind == "symbol":
                    symbol = local_def.symbol
                    if isinstance(symbol, Symbol) and symbol in expanded.free_symbols:
                        replacements[symbol] = local_def.expr
                    continue

                alias_fun = local_def.symbol
                if not isinstance(alias_fun, sp.FunctionClass):
                    continue
                for call in expanded.atoms(sp.Function):
                    if call.func == alias_fun:
                        if len(call.args) != 1:
                            raise ValueError(
                                f"FOC local '{alias_fun.__name__}' must be called with a single time argument."
                            )
                        replacements[call] = sp.simplify(
                            local_def.expr.subs(self._t, call.args[0])
                        )

            if not replacements:
                break
            expanded_next = sp.simplify(expanded.xreplace(replacements))
            if expanded_next == expanded:
                break
            expanded = expanded_next

        return expanded

    def _validate_expression_symbols(
        self,
        expr: Expr,
        var_funcs: dict[str, Function],
        param_syms: dict[str, Symbol],
        shock_syms: dict[str, Symbol],
        *,
        require_time_vars: bool,
    ) -> None:
        calls = list(expr.atoms(AppliedUndef))
        if require_time_vars and not calls:
            raise ValueError(
                "FOC expressions must reference at least one time-indexed model variable."
            )

        offsets: list[int] = []
        for call in calls:
            func_name = call.func.__name__
            if func_name not in var_funcs:
                raise KeyError(
                    f"Unknown time-dependent symbol '{func_name}'. "
                    f"Expected one of {list(var_funcs)}."
                )
            if len(call.args) != 1:
                raise ValueError(
                    f"Variable '{func_name}' must be indexed by a single time argument."
                )

            offset = sp.simplify(call.args[0] - self._t)
            if not offset.is_Integer:
                raise ValueError(
                    f"Variable '{func_name}' must use integer time offsets relative to t."
                )

            kk = int(offset)
            if kk not in {-1, 0, 1}:
                raise ValueError("FOC time shifts must stay within one period of t.")
            offsets.append(kk)

        if calls and offsets and max(offsets) - min(offsets) > 1:
            raise ValueError(
                "FOC time shifts must span at most one period across all variables."
            )

        for sym in expr.free_symbols:
            if sym == self._t:
                continue
            if sym.name in param_syms:
                continue
            if sym.name in shock_syms:
                raise ValueError(
                    "Shock innovations are not supported in DHM FOC strings. "
                    "Use state-process variables or the equation_idx fallback instead."
                )
            raise KeyError(
                f"Unknown symbol '{sym.name}' in FOC. "
                f"Expected a calibrated parameter or a time-indexed model variable."
            )

    def _normalize_foc_timing(
        self,
        expr: Expr,
        var_funcs: dict[str, Function],
    ) -> Expr:
        offsets = [
            int(sp.simplify(call.args[0] - self._t))
            for call in expr.atoms(sp.Function)
            if call.func.__name__ in var_funcs
        ]
        max_off = max(offsets) if offsets else 0
        normalized = expr
        if max_off != 0:
            normalized = sp.simplify(expr.subs(self._t, self._t - max_off))

        final_offsets = [
            int(sp.simplify(call.args[0] - self._t))
            for call in normalized.atoms(sp.Function)
            if call.func.__name__ in var_funcs
        ]
        if any(off not in {-1, 0} for off in final_offsets):
            raise ValueError("Normalized FOCs must only contain variables at t or t-1.")
        return normalized

    def _build_foc_vector_func(
        self,
        foc_exprs: Sequence[Expr],
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        var_order = self.solved.compiled.var_names
        cur_syms = [Symbol(f"cur_{name}") for name in var_order]
        lag_syms = [Symbol(f"lag_{name}") for name in var_order]
        param_syms = list(self.solved.compiled.calib_params)
        var_funcs = {v.__name__: v for v in self.solved.config.variables}

        subs_map: dict[Expr, Expr] = {}
        for name, cur_sym, lag_sym in zip(var_order, cur_syms, lag_syms):
            func = var_funcs[name]
            subs_map[func(self._t)] = cur_sym
            subs_map[func(self._t - 1)] = lag_sym

        compiled_exprs: list[Expr] = []
        allowed_syms = set(param_syms).union(cur_syms).union(lag_syms)
        for expr in foc_exprs:
            compiled = sp.simplify(expr.subs(subs_map))
            if compiled.atoms(AppliedUndef):
                raise ValueError(
                    "Failed to normalize all model variables in the provided FOCs."
                )
            leftovers = compiled.free_symbols.difference(allowed_syms)
            if leftovers:
                leftover_names = ", ".join(sorted(sym.name for sym in leftovers))
                raise ValueError(
                    f"Unresolved symbols remain in compiled FOCs: {leftover_names}."
                )
            compiled_exprs.append(compiled)

        lambda_args = [*cur_syms, *lag_syms, *param_syms]
        scalar_funcs = [
            njit(sp.lambdify(lambda_args, expr, modules="numpy"))
            for expr in compiled_exprs
        ]

        call_args = ", ".join(
            [
                *(f"cur[{i}]" for i in range(len(cur_syms))),
                *(f"lag[{i}]" for i in range(len(lag_syms))),
                *(f"params[{i}]" for i in range(len(param_syms))),
            ]
        )
        body = "\n    ".join(
            f"out[{i}] = func_{i}({call_args})" for i in range(len(scalar_funcs))
        )

        func_str = f"""
def vectorized_focs(cur, lag, params):
    out = np.empty(({len(scalar_funcs)},), dtype=np.float64)
    {body}
    return out
"""

        ns: dict[str, Any] = {"np": np}
        for i, fn in enumerate(scalar_funcs):
            ns[f"func_{i}"] = fn

        exec(dedent(func_str), ns)
        foc_func = njit(ns["vectorized_focs"])
        float_vector = numba.types.Array(numba.float64, 1, "C")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=numba.errors.NumbaExperimentalFeatureWarning
            )
            foc_func.compile((float_vector, float_vector, float_vector))

        return cast(
            Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], foc_func
        )
