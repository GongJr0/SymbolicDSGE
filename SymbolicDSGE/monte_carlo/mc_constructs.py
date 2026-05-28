from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, Union

import numpy as np
from numpy import float64, ndarray
from numpy.typing import NDArray

from .._diag_tests.result import MCResult, TestResult
from .._diag_tests.wald_test import (
    wald_covariance_hac,
    wald_mean_hac,
    wald_second_moment_hac,
)
from ..core.shock_generators import Shock
from ..core.solved_model import SolvedModel
from ..kalman.filter import FilterResult

NDF = NDArray[float64]
NDB = NDArray[np.bool_]
ShockValue = Union[Shock, Callable[[float | NDF], NDF], NDF]
ShockMapping = Mapping[str, ShockValue]
SeedIncrement = Union[int, Literal["auto"]]


class OpType(StrEnum):
    DATAGEN = "datagen"
    TRANSFORM = "transform"
    FILTER = "filter"
    TEST = "test"
    POSTPROC = "postproc"


@dataclass(frozen=True)
class MCData:
    """Standard data payload generated for one Monte Carlo replication."""

    states: NDF | None = None
    observables: NDF | None = None
    n_exog: int = -1
    raw: Mapping[str, NDF] = field(default_factory=dict)
    observable_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataGenReturn:
    """Legacy simulation-data container kept for compatibility."""

    state_mat: NDF | None
    obs_mat: NDF | None
    n_exog: int


@dataclass
class MCContext:
    rep_idx: int
    reference: SolvedModel
    dgp: SolvedModel | None
    data: MCData | None = None
    payloads: dict[str, Any] = field(default_factory=dict)
    results: dict[str, TestResult] = field(default_factory=dict)

    def require_data(self) -> MCData:
        if self.data is None:
            raise ValueError(
                "MC context has no generated data. Add a DATAGEN step first."
            )
        return self.data

    def require_payload(self, key: str) -> Any:
        if key not in self.payloads:
            raise KeyError(f"MC context payload '{key}' is not available.")
        return self.payloads[key]


class DataGenOp(Protocol):
    def __call__(
        self,
        *,
        reference: SolvedModel,
        dgp: SolvedModel | None,
        rep_idx: int,
        **kwargs: Any,
    ) -> MCData: ...


class ContextOp(Protocol):
    def __call__(
        self,
        *,
        context: MCContext,
        reference: SolvedModel,
        dgp: SolvedModel | None,
        rep_idx: int,
        **kwargs: Any,
    ) -> Any: ...


class FilterOp(Protocol):
    def __call__(
        self,
        *,
        context: MCContext,
        reference: SolvedModel,
        dgp: SolvedModel | None,
        rep_idx: int,
        **kwargs: Any,
    ) -> FilterResult: ...


class TestOp(Protocol):
    def __call__(
        self,
        *,
        context: MCContext,
        reference: SolvedModel,
        dgp: SolvedModel | None,
        rep_idx: int,
        **kwargs: Any,
    ) -> TestResult: ...


@dataclass(frozen=True)
class MCStep:
    name: str
    op_type: OpType
    func: Callable[..., Any]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    store_key: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MCStep name must be non-empty.")
        object.__setattr__(self, "op_type", OpType(self.op_type))
        object.__setattr__(self, "kwargs", dict(self.kwargs))

    @property
    def output_key(self) -> str:
        return self.store_key if self.store_key is not None else self.name


@dataclass(frozen=True)
class MCFailure:
    rep_idx: int
    step_name: str
    error_type: str
    message: str


@dataclass(frozen=True)
class MCPipelineResult:
    n_rep: int
    n_successful: int
    summaries: Mapping[str, MCResult]
    test_results: Mapping[str, tuple[TestResult, ...]] | None
    payloads: tuple[Mapping[str, Any], ...] | None
    contexts: tuple[MCContext, ...] | None
    failures: tuple[MCFailure, ...] = ()

    @property
    def succeeded(self) -> bool:
        return len(self.failures) == 0

    @property
    def statistic_traces(self) -> Mapping[str, NDF]:
        return {
            name: summary.statistic_trace for name, summary in self.summaries.items()
        }

    @property
    def pval_traces(self) -> Mapping[str, NDF]:
        return {name: summary.pval_trace for name, summary in self.summaries.items()}

    @property
    def rejection_traces(self) -> Mapping[str, NDB]:
        return {
            name: np.asarray(summary.pval_trace < summary.alpha, dtype=bool)
            for name, summary in self.summaries.items()
        }


@dataclass(frozen=True)
class MCPipeline:
    steps: tuple[MCStep, ...]

    def __init__(self, steps: Sequence[MCStep]) -> None:
        if not steps:
            raise ValueError("MCPipeline requires at least one step.")
        step_tuple = tuple(steps)
        self._validate_steps(step_tuple)
        object.__setattr__(self, "steps", step_tuple)

    @staticmethod
    def _validate_steps(steps: tuple[MCStep, ...]) -> None:
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError("MCPipeline step names must be unique.")
        if steps[0].op_type is not OpType.DATAGEN:
            raise ValueError("MCPipeline first step must be a DATAGEN step.")
        for step in steps[1:]:
            if step.op_type is OpType.DATAGEN:
                raise ValueError(
                    "MCPipeline supports only one DATAGEN step, in first position."
                )

    def run(
        self,
        *,
        reference: SolvedModel,
        dgp: SolvedModel | None = None,
        n_rep: int,
        retain_payloads: bool = True,
        retain_test_results: bool = True,
        retain_contexts: bool = False,
        fail_fast: bool = True,
    ) -> MCPipelineResult:
        if n_rep <= 0:
            raise ValueError("n_rep must be positive.")

        contexts: list[MCContext] = []
        payload_traces: list[Mapping[str, Any]] = []
        failures: list[MCFailure] = []
        results_by_step: dict[str, list[TestResult]] = {}

        for rep_idx in range(n_rep):
            context = MCContext(rep_idx=rep_idx, reference=reference, dgp=dgp)
            try:
                for step in self.steps:
                    self._run_step(context, step)
            except Exception as exc:
                if fail_fast:
                    raise
                failures.append(
                    MCFailure(
                        rep_idx=rep_idx,
                        step_name=step.name,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            contexts.append(context)
            if retain_payloads:
                payload_traces.append(dict(context.payloads))
            for name, result in context.results.items():
                results_by_step.setdefault(name, []).append(result)

        summaries = _summarize_results(results_by_step)
        return MCPipelineResult(
            n_rep=n_rep,
            n_successful=len(contexts),
            summaries=summaries,
            test_results=(
                {name: tuple(values) for name, values in results_by_step.items()}
                if retain_test_results
                else None
            ),
            payloads=tuple(payload_traces) if retain_payloads else None,
            contexts=tuple(contexts) if retain_contexts else None,
            failures=tuple(failures),
        )

    def _run_step(self, context: MCContext, step: MCStep) -> None:
        kwargs = dict(step.kwargs)
        if step.op_type is OpType.DATAGEN:
            out = step.func(
                reference=context.reference,
                dgp=context.dgp,
                rep_idx=context.rep_idx,
                **kwargs,
            )
            if isinstance(out, DataGenReturn):
                out = MCData(
                    states=out.state_mat,
                    observables=out.obs_mat,
                    n_exog=out.n_exog,
                )
            if not isinstance(out, MCData):
                raise TypeError("DATAGEN steps must return MCData.")
            context.data = out
            context.payloads[step.output_key] = out
            return

        out = step.func(
            context=context,
            reference=context.reference,
            dgp=context.dgp,
            rep_idx=context.rep_idx,
            **kwargs,
        )
        if step.op_type is OpType.TRANSFORM and isinstance(out, MCData):
            context.data = out
        if step.op_type is OpType.FILTER and not isinstance(out, FilterResult):
            raise TypeError("FILTER steps must return FilterResult.")
        if step.op_type is OpType.TEST:
            if not isinstance(out, TestResult):
                raise TypeError("TEST steps must return TestResult.")
            context.results[step.name] = out
        context.payloads[step.output_key] = out


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
                [np.asarray(raw[name], dtype=np.float64)[1:] for name in obs_names]
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
    source: Literal[
        "states",
        "observables",
        "x_pred",
        "x_filt",
        "y_pred",
        "y_filt",
        "innov",
        "std_innov",
        "payload",
    ],
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
                dist=shock.dist,
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


def _summarize_results(
    results_by_step: Mapping[str, list[TestResult]],
) -> dict[str, MCResult]:
    summaries: dict[str, MCResult] = {}

    for step_name, results in results_by_step.items():
        if not results:
            continue
        first = results[0]
        for result in results[1:]:
            if (
                result.dist is not first.dist
                or result.pval_method is not first.pval_method
                or result.df != first.df
                or result.alpha != first.alpha
            ):
                raise ValueError(
                    f"Test results for step '{step_name}' have incompatible metadata."
                )
        stat_trace = np.asarray(
            [result.statistic for result in results], dtype=np.float64
        )
        summary = MCResult(
            test_name=step_name,
            dist=first.dist,
            df=first.df,
            pval_method=first.pval_method,
            alpha=first.alpha,
            statistic_trace=stat_trace,
        )
        summaries[step_name] = summary

    return summaries
