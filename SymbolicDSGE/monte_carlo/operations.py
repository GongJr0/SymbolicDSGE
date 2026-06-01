from __future__ import annotations

from typing import Literal, Mapping, Sequence, Any

import numpy as np
from numpy import float64, ndarray

from .._diag_tests.ljung_box import ljung_box
from .._diag_tests.result import TestResult
from .._diag_tests.wald_test import (
    wald_covariance_hac,
    wald_mean_hac,
    wald_second_moment_hac,
)
from ..core.solved_model import SolvedModel
from ..kalman.filter import FilterResult
from ..regression.enums import RegressionKind
from ..regression.ols import ols
from ..regression.result import RegressionResult
from ..regression.ridge import ridge, ridge_gs
from .mc_constructs import MCContext, MCData, NDF, SeedIncrement, ShockMapping
from .operation_utils import (
    InpSources,
    _clone_or_pass_shocks,
    _resolve_context_array,
    _select_raw_rep_array,
)


def simulate_dgp(
    *,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    T: int,
    shocks: ShockMapping | None = None,
    seed_increment: SeedIncrement = "auto",
    shock_scale: float | float64 = 1.0,
    x0: ndarray | None = None,
    observables: bool = True,
) -> MCData:
    del reference
    if dgp is None:
        raise ValueError("simulate_dgp requires a DGP SolvedModel.")
    sim_shocks = _clone_or_pass_shocks(
        shocks,
        T=T,
        rep_idx=rep_idx,
        seed_increment=seed_increment,
    )
    raw = dgp.sim(
        T=T,
        shocks=sim_shocks,
        shock_scale=shock_scale,
        x0=x0,
        observables=observables,
    )
    states = np.ascontiguousarray(raw["_X"], dtype=np.float64)
    obs_names = (
        tuple(getattr(dgp.compiled, "observable_names", ())) if observables else ()
    )
    obs_mat = None
    if obs_names:
        obs_mat = np.ascontiguousarray(
            np.column_stack(
                [
                    np.asarray(raw[name], dtype=np.float64)[1:]
                    for name in obs_names  # pyright: ignore
                ]
            ),
            dtype=np.float64,
        )
    return MCData(
        states=states,
        observables=obs_mat,
        raw=raw,
        n_exog=int(getattr(dgp.compiled, "n_exog", -1)),
        observable_names=obs_names,
    )


def raw_data_datagen(
    *,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    states: NDF | None = None,
    observables: NDF | None = None,
    n_exog: int = -1,
    raw: Mapping[str, NDF] | None = None,
    observable_names: Sequence[str] = (),
) -> MCData:
    del reference, dgp
    if states is None and observables is None:
        raise ValueError("raw_data_datagen requires states, observables, or both.")

    state_mat = None
    if states is not None:
        state_mat = _select_raw_rep_array(
            states,
            rep_idx=rep_idx,
            name="states",
            allow_vector=False,
        )
    obs_mat = None
    if observables is not None:
        obs_mat = _select_raw_rep_array(
            observables,
            rep_idx=rep_idx,
            name="observables",
            allow_vector=True,
        )
    raw_payload: dict[str, NDF] = {}
    if state_mat is not None:
        raw_payload["_X"] = state_mat
    if raw is not None:
        raw_payload.update(
            {
                key: _select_raw_rep_array(
                    value,
                    rep_idx=rep_idx,
                    name=f"raw['{key}']",
                    allow_vector=True,
                )
                for key, value in raw.items()
            }
        )
    return MCData(
        states=state_mat,
        observables=obs_mat,
        n_exog=int(n_exog),
        raw=raw_payload,
        observable_names=tuple(observable_names),
    )


def run_reference_filter(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    filter_mode: Literal["linear", "extended"] = "linear",
    observables: list[str] | None = None,
    x0: NDF | None = None,
    p0_mode: Literal["diag", "eye"] | None = None,
    p0_scale: float | float64 | None = None,
    jitter: float | float64 | None = None,
    symmetrize: bool | None = None,
    return_shocks: bool = False,
    R: NDF | None = None,
    estimate_R_diag: bool = False,
    R_scale: float = 1.0,
) -> FilterResult:
    del dgp, rep_idx
    data = context.require_data()
    if data.observables is None:
        raise ValueError("Reference filter step requires generated observables.")
    obs = observables
    if obs is None and data.observable_names:
        obs = list(data.observable_names)
    return reference.kalman(
        y=data.observables,
        filter_mode=filter_mode,
        observables=obs,
        x0=x0,
        p0_mode=p0_mode,
        p0_scale=p0_scale,
        jitter=jitter,
        symmetrize=symmetrize,
        return_shocks=return_shocks,
        R=R,
        estimate_R_diag=estimate_R_diag,
        R_scale=R_scale,
    )


def run_wald_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    target: NDF,
    kind: Literal["mean", "covariance", "second_moment"] = "mean",
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> TestResult:
    del reference, dgp, rep_idx
    arr = _resolve_context_array(
        context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    target_arr = np.asarray(target, dtype=np.float64)
    if kind == "mean":
        return wald_mean_hac(
            arr,
            target_arr,
            kernel=kernel,
            bandwidth=bandwidth,
            alpha=alpha,
            _auto_pval=False,
        )
    if kind == "covariance":
        return wald_covariance_hac(
            arr,
            target_arr,
            kernel=kernel,
            bandwidth=bandwidth,
            alpha=alpha,
            _auto_pval=False,
        )
    if kind == "second_moment":
        return wald_second_moment_hac(
            arr,
            target_arr,
            kernel=kernel,
            bandwidth=bandwidth,
            alpha=alpha,
            _auto_pval=False,
        )
    raise ValueError(f"Unsupported Wald test kind: {kind}")


def run_ljung_box_test(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    column: Sequence[int] | int | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 10,
    alpha: float = 0.05,
) -> TestResult:
    del reference, dgp, rep_idx

    col_idx: Sequence[int] | None
    if isinstance(column, int):
        col_idx = [column]
    else:
        col_idx = column

    arr = _resolve_context_array(
        context,
        source=source,
        filter_key=filter_key,
        payload_key=payload_key,
        columns=col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    if arr.shape[1] != 1:
        raise ValueError("Ljung-Box test requires a single column of data.")

    return ljung_box(arr[:, 0], L=lags, alpha=alpha, _auto_pval=False)


def run_regression(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    kind: Literal["ols", "ridge", "ridge_gs"] = "ols",
    y_source: InpSources,
    X_source: InpSources,
    filter_key: str = "filter",
    y_payload_key: str | None = None,
    x_payload_key: str | None = None,
    y_column: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    intercept: bool = True,
    burn_in: int = 0,
    drop_initial: bool = False,
    variables: Sequence[str] | None = None,
    **kind_kwargs: Any,
) -> RegressionResult:
    del reference, dgp, rep_idx

    y_col_idx: Sequence[int] | None
    if isinstance(y_column, int):
        y_col_idx = [y_column]
    else:
        y_col_idx = y_column

    y = _resolve_context_array(
        context,
        source=y_source,
        filter_key=filter_key,
        payload_key=y_payload_key,
        columns=y_col_idx,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )
    X = _resolve_context_array(
        context,
        source=X_source,
        filter_key=filter_key,
        payload_key=x_payload_key,
        columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
    )

    if y.shape[1] != 1:
        raise ValueError(
            "Regression response must resolve to exactly one column. "
            f"Got shape {y.shape}."
        )
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            "Regression response and design matrix must have the same number "
            f"of rows. Got y={y.shape[0]} and X={X.shape[0]}."
        )
    if variables is not None and len(variables) != X.shape[1]:
        raise ValueError(
            "Regression variable names must match the number of design columns. "
            f"Got {len(variables)} names for {X.shape[1]} columns."
        )

    y_vec = np.ascontiguousarray(y[:, 0], dtype=np.float64)
    variable_names = list(variables) if variables is not None else None

    match RegressionKind(kind):
        case RegressionKind.OLS:
            return ols(
                X,
                y_vec,
                intercept=intercept,
                variables=variable_names,
                **kind_kwargs,
            )
        case RegressionKind.RIDGE:
            return ridge(
                X,
                y_vec,
                intercept=intercept,
                variables=variable_names,
                **kind_kwargs,
            )
        case RegressionKind.RIDGE_GS:
            return ridge_gs(
                X,
                y_vec,
                intercept=intercept,
                variables=variable_names,
                **kind_kwargs,
            )
        case _:
            raise ValueError(f"Unsupported regression kind: {kind}")
