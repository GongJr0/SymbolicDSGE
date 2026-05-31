from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from numpy import float64, ndarray

from .._diag_tests.result import TestResult
from .._diag_tests.wald_test import (
    wald_covariance_hac,
    wald_mean_hac,
    wald_second_moment_hac,
)
from .._diag_tests.ljung_box import ljung_box

from ..core.shock_generators import Shock
from ..core.solved_model import SolvedModel
from ..kalman.filter import FilterResult
from ..regression.ols import OLSResult, ols
from .mc_constructs import (
    DataGenReturn,
    MCContext,
    MCData,
    MCStep,
    NDF,
    OpType,
    SeedIncrement,
    ShockMapping,
)

InpSources = Literal[
    "states",
    "observables",
    "x_pred",
    "x_filt",
    "y_pred",
    "y_filt",
    "innov",
    "std_innov",
    "payload",
]


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

    return ljung_box(arr[:, 0], L=lags, alpha=alpha)


def run_regression(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    y_source: InpSources,
    X_source: InpSources,
    filter_key: str = "filter",
    y_payload_key: str | None = None,
    x_payload_key: str | None = None,
    y_column: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    variables: Sequence[str] | None = None,
) -> OLSResult:
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
    return ols(X, y_vec, variables=variable_names)


def simulation_step(name: str = "datagen", **kwargs: Any) -> MCStep:
    return MCStep(name=name, op_type=OpType.DATAGEN, func=simulate_dgp, kwargs=kwargs)


def raw_data_step(name: str = "datagen", **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.DATAGEN, func=raw_data_datagen, kwargs=kwargs
    )


def transform_step(
    name: str,
    func: Callable[..., Any],
    *,
    store_key: str | None = None,
    **kwargs: Any,
) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=func,
        kwargs=kwargs,
        store_key=store_key,
    )


def reference_filter_step(name: str = "filter", **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.FILTER, func=run_reference_filter, kwargs=kwargs
    )


def wald_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(name=name, op_type=OpType.TEST, func=run_wald_test, kwargs=kwargs)


def ljung_box_test_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.TEST, func=run_ljung_box_test, kwargs=kwargs
    )


def regression_step(name: str, **kwargs: Any) -> MCStep:
    return MCStep(
        name=name, op_type=OpType.REGRESSION, func=run_regression, kwargs=kwargs
    )


class MCReferenceConstruct:
    """
    Construct container for Monte Carlo reference data generation via simulation of a solved model.
    """

    def __init__(self, model: SolvedModel, T: int, N: int) -> None:
        self._model = model
        self._T = T
        self._N = N

    def _data_from_sim(
        self,
        *,
        shocks: ShockMapping | None = None,
        shock_scale: float | float64 = 1.0,
        x0: ndarray | None = None,
        observables: bool = True,
    ) -> DataGenReturn:
        data = simulate_dgp(
            reference=self.model,
            dgp=self.model,
            rep_idx=0,
            T=self.T,
            shocks=shocks,
            shock_scale=shock_scale,
            x0=x0,
            observables=observables,
        )
        return DataGenReturn(
            state_mat=data.states,
            obs_mat=data.observables,
            n_exog=data.n_exog,
        )

    @property
    def model(self) -> SolvedModel:
        return self._model

    @property
    def T(self) -> int:
        return self._T

    @property
    def N(self) -> int:
        return self._N

    class DataGeneratingCallable:
        """Simple callable wrapper for external data-generating functions."""

        def __init__(
            self, func: Callable[[int], tuple[NDF, NDF | None]], T: int, N: int
        ) -> None:
            self._func = func
            self._T = T
            self._N = N

        def __call__(self) -> DataGenReturn:
            state_mat, obs_mat = self.func(self.T)
            return DataGenReturn(
                state_mat=state_mat,
                obs_mat=obs_mat,
                n_exog=-1,
            )

        @property
        def func(self) -> Callable[[int], tuple[NDF, NDF | None]]:
            return self._func

        @property
        def T(self) -> int:
            return self._T

        @property
        def N(self) -> int:
            return self._N


def _clone_or_pass_shocks(
    shocks: ShockMapping | None,
    *,
    T: int,
    rep_idx: int,
    seed_increment: SeedIncrement,
) -> Mapping[str, Callable[[float | NDF], NDF] | NDF] | None:
    if shocks is None:
        return None
    out: dict[str, Callable[[float | NDF], NDF] | NDF] = {}
    seed_offset = rep_idx * _resolve_seed_increment(shocks, seed_increment)
    for name, shock in shocks.items():
        if isinstance(shock, Shock):
            if shock.T != T:
                raise ValueError(f"Shock '{name}' has T={shock.T}, expected {T}.")
            if shock.shock_arr is not None:
                raise ValueError(
                    "MC simulation requires generator-style Shock instances."
                )
            if ("," in name) != shock.multivar:
                raise ValueError(
                    f"Shock '{name}' must set multivar={',' in name} to match its specification."
                )
            seed = None if shock.seed is None else int(shock.seed) + seed_offset
            out[name] = Shock(
                T=shock.T,
                dist=shock.dist,  # pyright: ignore
                multivar=shock.multivar,
                seed=seed,
                dist_args=shock.dist_args,
                dist_kwargs=shock.dist_kwargs.copy(),
            ).shock_generator()
        else:
            out[name] = shock
    return out


def _resolve_seed_increment(
    shocks: ShockMapping,
    seed_increment: SeedIncrement,
) -> int:
    if seed_increment == "auto":
        return sum(
            1
            for shock in shocks.values()
            if isinstance(shock, Shock) and shock.seed is not None
        )
    increment = int(seed_increment)
    if increment < 0:
        raise ValueError("seed_increment must be non-negative or 'auto'.")
    return increment


def _resolve_context_array(
    context: MCContext,
    *,
    source: str,
    filter_key: str,
    payload_key: str | None,
    columns: Sequence[int] | slice | None,
    burn_in: int,
    drop_initial: bool,
) -> NDF:
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")
    if source == "states":
        arr = context.require_data().states
        if arr is None:
            raise ValueError("MC context has no generated states.")
        if drop_initial:
            arr = arr[1:]
    elif source == "observables":
        obs = context.require_data().observables
        if obs is None:
            raise ValueError("MC context has no generated observables.")
        arr = obs
    elif source == "payload":
        if payload_key is None:
            raise ValueError("payload_key is required when source='payload'.")
        arr = context.require_payload(payload_key)
    else:
        filter_result = context.require_payload(filter_key)
        if not isinstance(filter_result, FilterResult):
            raise TypeError(f"Payload '{filter_key}' is not a FilterResult.")
        if not hasattr(filter_result, source):
            raise ValueError(f"FilterResult has no array source '{source}'.")
        arr = getattr(filter_result, source)

    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError(f"Selected MC array must be 1D or 2D, got shape {out.shape}.")
    if columns is not None:
        out = out[:, columns]
        if out.ndim == 1:
            out = out.reshape(-1, 1)
    if burn_in:
        out = out[burn_in:]
    return np.ascontiguousarray(out, dtype=np.float64)


def _select_raw_rep_array(
    value: NDF,
    *,
    rep_idx: int,
    name: str,
    allow_vector: bool,
) -> NDF:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 3:
        if rep_idx >= arr.shape[0]:
            raise IndexError(
                f"{name} has {arr.shape[0]} replications, cannot select rep_idx={rep_idx}."
            )
        out = arr[rep_idx]
    elif arr.ndim == 2:
        out = arr
    elif allow_vector and arr.ndim == 1:
        out = arr.reshape(-1, 1)
    else:
        expected = "1D, 2D, or 3D" if allow_vector else "2D or 3D"
        raise ValueError(f"{name} must be {expected}, got shape {arr.shape}.")

    if out.ndim != 2:
        raise ValueError(f"{name} must resolve to a 2D array, got shape {out.shape}.")
    return np.ascontiguousarray(out, dtype=np.float64)
