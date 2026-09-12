"""Public constructors for native Monte Carlo pipeline steps."""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.shock_generators import Shock
from .custom_op import NumbaCustomFunc
from .mc_constructs import ColumnSelector, MCStep, OpType, _compile_source_args
from .postproc import run_kde

NDF = NDArray[np.float64]


def simulation_step(
    name: str = "datagen",
    target: str = "dgp",
    n_retain: int = -1,
    *,
    T: int,
    shocks: Mapping[str, Shock | Callable[[float | NDF], NDF] | NDF] | None = None,
    shock_scale: float = 1.0,
    x0: list[float] | NDF | None = None,
    observables: bool = True,
) -> MCStep:
    """A simulation DATAGEN step, producing a sample of the target model's path.

    Parameters
    ----------
    T : int
        Periods to simulate.
    name : str
        Step name, used to reference the step in later steps and results.
    target : str
        Target model by role. "reference" or "dgp".
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    shocks : Mapping[str, Shock | Callable[[float | NDF], NDF] | NDF] | None
        Shock specification for the simulation. Mirrors :meth:`SolvedModel.sim`.
        A bare array can include a replication dimension (n_rep, T, n_shock)
        for per-rep dispatch.
    shock_scale : float
        Scale factor to multply the shocks by.
    x0 : list[float] | NDF | None
        Initial state for the simulation.
    observables : bool
        Whether to return the observables in the sample.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.

    """
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        kwargs={
            "target": target,
            "T": T,
            "shocks": shocks,
            "shock_scale": shock_scale,
            "x0": x0,
            "observables": observables,
        },
        step_type="simulation",
        n_retain=n_retain,
    )


def raw_model_data_step(
    name: str = "datagen",
    n_retain: int = -1,
    *,
    states: NDF | Sequence[float] | Sequence[Sequence[float]] | None = None,
    shocks: NDF | Sequence[float] | Sequence[Sequence[float]] | None = None,
    observables: NDF | Sequence[float] | Sequence[Sequence[float]] | None = None,
    state_names: Sequence[str] = (),
    shock_names: Sequence[str] = (),
    observable_names: Sequence[str] = (),
) -> MCStep:
    """DATAGEN step to include raw data as model output.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    states : NDF | Sequence[float] | Sequence[Sequence[float]] | None
        State vector data. (T, n_var) for single source, and (n_rep, T, n_var) for per-rep dispatch.
    shocks : NDF | Sequence[float] | Sequence[Sequence[float]] | None
        Shock vector data. (T, n_shock) for single source, and (n_rep, T, n_shock) for per-rep dispatch.
    observables : NDF | Sequence[float] | Sequence[Sequence[float]] | None
        Observable vector data. (T, n_obs) for single source, and (n_rep, T, n_obs) for per-rep dispatch.
    state_names : Sequence[str]
        Names of the state variables, in order. Used to label the output columns.
    shock_names : Sequence[str]
        Names of the shock variables, in order. Used to label the output columns.
    observable_names : Sequence[str]
        Names of the observable variables, in order. Used to label the output columns.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.

    """
    return MCStep(
        name=name,
        op_type=OpType.DATAGEN,
        kwargs={
            "states": states,
            "shocks": shocks,
            "observables": observables,
            "state_names": state_names,
            "shock_names": shock_names,
            "observable_names": observable_names,
        },
        step_type="raw_model_data",
        n_retain=n_retain,
    )


def reference_filter_step(
    name: str = "filter",
    n_retain: int = -1,
    *,
    filter_mode: Literal["linear", "extended", "unscented"] = "linear",
    observables: list[str] | None = None,
    x0: dict[str, float | np.float64] | list[float | np.float64] | NDF | None = None,
    P0: NDF | None = None,
    R: NDF | None = None,
    jitter: float | np.float64 | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = False,
    return_shocks: bool = False,
) -> MCStep:
    """FILTER step to run a Kalman filter on the reference model.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    filter_mode : Literal["linear", "extended", "unscented"]
        Mode of the kalman filter. "linear", "extended", or "unscented".
        - "linear" -> Linear Gaussian KF
        - "extended" -> Linear Gaussian KF with non-linear measurements allowed.
        - "unscented" -> Unscented Kalman Filter (UKF) for second-order non-linearities
        in states and non-linear measurement equations.
    observables : list[str] | None
        Observable names to include in the filter. If ``None``, all observables are used.
    x0 : dict[str, float | np.float64] | list[float | np.float64] | NDF | None
        Initial state for the filter.
    P0 : NDF | None
        Initial state covariance matrix for the filter. If ``None``, the steady-state covariance is used.
    R : NDF | None
        Measurement noise covariance matrix for the filter. If ``None``, and the model has no ``R`` in :class:`KalmanConfig`,
        the filter will raise.
    jitter : float | np.float64 | None
        Jitter to add when a cholesky decomposition fails. If ``None``, no jitter is added.
    symmetrize : bool
        Whether to symmetrize the covariance matrices in the filter kernel.
    joseph_cov : bool
        Whether to use the Joseph form of the covariance update in the filter kernel.
    return_shocks : bool
        Whether to return estimated shocks in the filter output. Unsupported for unscented filters.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.

    """
    return MCStep(
        name=name,
        op_type=OpType.FILTER,
        kwargs={
            "filter_mode": filter_mode,
            "observables": observables,
            "x0": x0,
            "P0": P0,
            "R": R,
            "jitter": jitter,
            "symmetrize": symmetrize,
            "joseph_cov": joseph_cov,
            "return_shocks": return_shocks,
        },
        step_type="filter",
        n_retain=n_retain,
    )


def add_payload_step(
    name: str,
    payload: (
        NDF
        | Sequence[float]
        | Sequence[Sequence[float]]
        | Sequence[Sequence[Sequence[float]]]
    ),
    n_retain: int = -1,
) -> MCStep:
    """TRANSFORM step to add arbitrary data as a payload to the pipeline.
    The payload is stored in the step's output.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    payload : NDF | Sequence[float] | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]
        Arbitrary data as 1D, 2D, or 3D array-like. The data is stored in the step's output.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.

    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        kwargs={"value": payload},
        step_type="payload",
        n_retain=n_retain,
    )


def passthrough_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector,
    burn_in: int = 0,
    drop_initial: bool = False,
) -> MCStep:
    """TRANSFORM step to passthrough a specific output column from a step in the pipeline.
    Allows choosing specific steps to be retained; it's useful for memory management
    in high-replication and/or high-dimensional pipelines.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to passthrough from.
    field : str
        Field to passthrough from the source step.
    columns : ColumnSelector
        Columns of the the field to passthrough. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    burn_in : int
        Number of initial samples to discard from the source step before passthrough.
    drop_initial : bool
        Whether to drop the initial sample from the passthrough.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.

    """
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        kwargs={},
        source_args=(
            _compile_source_args(
                arg="sample",
                source=source,
                field=field,
                columns=columns,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
        ),
        step_type="passthrough",
        n_retain=n_retain,
    )


def _one_source_step(
    name: str,
    op_type: OpType,
    step_type: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector,
    burn_in: int,
    drop_initial: bool,
    kwargs: Mapping[str, Any],
) -> MCStep:
    return MCStep(
        name=name,
        op_type=op_type,
        kwargs=kwargs,
        source_args=(
            _compile_source_args(
                arg="sample",
                source=source,
                field=field,
                columns=columns,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
        ),
        step_type=step_type,
        n_retain=n_retain,
    )


def transform_step(
    name: str,
    func: Callable[..., Any] | NumbaCustomFunc,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    output_shape: tuple[int, int],
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
) -> MCStep:
    """Custom TRANSFORM step with a user-defined callable.
    The callable accepts (input: NDF, output: NDF) and returns
    an integer status. Non-zero means the step failed.
    The function has namespace restrictions through :class:`NumbaCustomFunc`
    and is compiled with ``@numba.cfunc``. See :class:`NumbaCustomFunc` for details.


    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    func : Callable[..., Any] | NumbaCustomFunc
        Callable to apply to the input data.
        If a bare callable is provided, it is wrapped in a :class:`NumbaCustomFunc`.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    output_shape : tuple[int, int]
        Shape of the output array the transform will populate per-replication.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    columns : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the transform.
    drop_initial : bool
        Whether to drop the initial sample from the transform output.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.

    """
    n_out, p_out = output_shape
    if n_out < 0 or p_out < 0:
        raise ValueError("output_shape dimensions must be non-negative.")
    wrapped = func if isinstance(func, NumbaCustomFunc) else NumbaCustomFunc(func)
    return MCStep(
        name=name,
        op_type=OpType.TRANSFORM,
        func=wrapped,
        kwargs={"output_shape": (n_out, p_out)},
        source_args=(
            _compile_source_args(
                arg="sample",
                source=source,
                field=field,
                columns=columns,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
        ),
        step_type="transform:custom",
        n_retain=n_retain,
    )


def standardize_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    ddof: int = 0,
) -> MCStep:
    """TRANSFORM step to standardize an input sample.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    columns : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the transform.
    drop_initial : bool
        Whether to drop the initial sample from the transform output.
    ddof : int
        Delta degrees of freedom for the standard deviation calculation. ``0`` divides by ``n``, ``1`` by ``n - 1``.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "standardize",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"ddof": ddof},
        n_retain=n_retain,
    )


def log_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    offset: float = 0.0,
) -> MCStep:
    """TRANSFORM step to take the logarithm of an input sample, with an optional offset.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    columns : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the transform.
    drop_initial : bool
        Whether to drop the initial sample from the transform output.
    offset : float
        Additive offset to apply before taking the logarithm.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "log",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"offset": offset},
        n_retain=n_retain,
    )


def log_diff_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    offset: float = 0.0,
) -> MCStep:
    """TRANSFORM step to take the logarithmic difference of an input sample, with an optional offset.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    columns : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the transform.
    drop_initial : bool
        Whether to drop the initial sample from the transform output.
    offset : float
        Additive offset to apply before taking the logarithmic difference.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "log_diff",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"offset": offset},
        n_retain=n_retain,
    )


def diff_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    order: int = 1,
) -> MCStep:
    """TRANSFORM step to take the difference of an input sample, with a specified order.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    columns : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the transform.
    drop_initial : bool
        Whether to drop the initial sample from the transform output.
    order : int
        Order of differencing to apply. Must be a positive integer.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "diff",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"order": order},
        n_retain=n_retain,
    )


def rolling_mean_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
) -> MCStep:
    """TRANSFORM step to compute the rolling mean of an input sample over a specified window.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    columns : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the transform.
    drop_initial : bool
        Whether to drop the initial sample from the transform output.
    window : int
        Rolling window size for computing the mean. Must be a positive integer.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "rolling_mean",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"window": window},
        n_retain=n_retain,
    )


def rolling_std_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
    ddof: int = 0,
) -> MCStep:
    """TRANSFORM step to compute the rolling standard deviation of an input sample over a specified window.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    columns : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the transform.
    drop_initial : bool
        Whether to drop the initial sample from the transform output.
    window : int
        Rolling window size for computing the standard deviation. Must be a positive integer.
    ddof : int
        Delta degrees of freedom for the standard deviation calculation. ``0`` divides by ``n``, ``1`` by ``n - 1``.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "rolling_std",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"window": window, "ddof": ddof},
        n_retain=n_retain,
    )


def rolling_var_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    window: int = 10,
    ddof: int = 0,
) -> MCStep:
    """TRANSFORM step to compute the rolling variance of an input sample over a specified window.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    columns : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the transform.
    drop_initial : bool
        Whether to drop the initial sample from the transform output.
    window : int
        Rolling window size for computing the variance. Must be a positive integer.
    ddof : int
        Delta degrees of freedom for the variance calculation. ``0`` divides by ``n``, ``1`` by ``n - 1``.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TRANSFORM,
        "rolling_var",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"window": window, "ddof": ddof},
        n_retain=n_retain,
    )


def regression_step(
    name: str,
    n_retain: int = -1,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    kind: Literal[
        "ols", "ridge", "lasso", "elastic_net", "ridge_gs", "lasso_gs", "elastic_net_gs"
    ] = "ols",
    intercept: bool = True,
    variables: list[str] | None = None,
    **kind_kwargs: Any,
) -> MCStep:
    """REGRESSION step to perform the speified regression of y on X, with optional hyperparameter tuning.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    y_source : str
        Source step name for the dependent variable.
    y_field : str
        Field name for the dependent variable in the source step.
    X_source : str
        Source step name for the regressors variables.
    X_field : str
        Field name for the regressors in the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    y_column : ColumnSelector
        Column selector for the dependent variable in the source step. Can be a single column index.
    X_columns : ColumnSelector
        Columns of the regressors in the source field.
        Can be a single column index, a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source steps before applying the regression.
    drop_initial : bool
        Whether to drop the initial sample from the regression output.
    kind : Literal["ols", "ridge", "lasso", "elastic_net", "ridge_gs", "lasso_gs", "elastic_net_gs"]
        Regression type to perform.
    intercept : bool
        Whether to include an intercept in the regression.
    variables : list[str] | None
        Names of the regressors, in order. If ``None``, default names are used.
    **kind_kwargs : Any
        Additional keyword arguments specific to the regression kind.
        See the documentation for the specific regression type for details.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return MCStep(
        name=name,
        op_type=OpType.REGRESSION,
        kwargs={
            "kind": kind,
            "intercept": intercept,
            "variables": variables,
            **kind_kwargs,
        },
        source_args=(
            _compile_source_args(
                arg="y",
                source=y_source,
                field=y_field,
                columns=y_column,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
            _compile_source_args(
                arg="X",
                source=X_source,
                field=X_field,
                columns=X_columns,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
        ),
        step_type="regression",
        n_retain=n_retain,
    )


def _two_source_test(
    name: str,
    step_type: str,
    n_retain: int = -1,
    *,
    first_source: str,
    first_field: str,
    first_arg: str,
    first_columns: ColumnSelector,
    second_source: str,
    second_field: str,
    second_arg: str,
    second_columns: ColumnSelector,
    burn_in: int,
    drop_initial: bool,
    kwargs: Mapping[str, Any],
) -> MCStep:
    return MCStep(
        name=name,
        op_type=OpType.TEST,
        kwargs=kwargs,
        source_args=(
            _compile_source_args(
                arg=first_arg,
                source=first_source,
                field=first_field,
                columns=first_columns,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
            _compile_source_args(
                arg=second_arg,
                source=second_source,
                field=second_field,
                columns=second_columns,
                burn_in=burn_in,
                drop_initial=drop_initial,
            ),
        ),
        step_type=step_type,
        n_retain=n_retain,
    )


def wald_test_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    kind: Literal["mean", "covariance", "second_moment"] = "mean",
    target: NDF,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> MCStep:
    """TEST step to perform a Wald test on the specified input sample against a target value.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    target : NDF
        Target value, vector, or matrix to test against. Must be compatible with the input sample shape.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    columns : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the test.
    drop_initial : bool
        Whether to drop the initial sample from the test output.
    kind : Literal["mean", "covariance", "second_moment"]
        Kind of wald test to perform.
    kernel : Literal["bartlett", "parzen", "qs"]
        Kernel type for the long-run covariance estimation.
    bandwidth : int | Literal["andrews", "wooldridge", "auto"] | None
        Kernel bandwidth for the long-run covariance estimation.
        ``None`` uses the wooldridge textbook heuristic.
    alpha : float
        Significance level for the test.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TEST,
        "wald",
        source=source,
        field=field,
        columns=columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={
            "kind": kind,
            "target": target,
            "kernel": kernel,
            "bandwidth": bandwidth,
            "alpha": alpha,
        },
        n_retain=n_retain,
    )


def ljung_box_test_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    column: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 10,
    alpha: float = 0.05,
) -> MCStep:
    """TEST step to perform a Ljung-Box test on the specified input sample.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    column : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the test.
    drop_initial : bool
        Whether to drop the initial sample from the test output.
    lags : int
        Number of lags to include in the test. Must be a positive integer.
    alpha : float
        Significance level for the test.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TEST,
        "ljung_box",
        source=source,
        field=field,
        columns=column,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"lags": lags, "alpha": alpha},
        n_retain=n_retain,
    )


def jarque_bera_test_step(
    name: str,
    n_retain: int = -1,
    *,
    source: str,
    field: str,
    column: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep:
    """TEST step to perform a Jarque-Bera test on the specified input sample.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    source : str
        Source step name to input from.
    field : str
        Field to input from the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    column : ColumnSelector
        Columns of the the field to input. Can be a single column index,
        a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source step before applying the test.
    drop_initial : bool
        Whether to drop the initial sample from the test output.
    alpha : float
        Significance level for the test.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _one_source_step(
        name,
        OpType.TEST,
        "jarque_bera",
        source=source,
        field=field,
        columns=column,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"alpha": alpha},
        n_retain=n_retain,
    )


def breusch_pagan_test_step(
    name: str,
    n_retain: int = -1,
    *,
    residuals_source: str,
    residuals_field: str,
    X_source: str,
    X_field: str,
    residual_col: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    robust: bool = False,
    alpha: float = 0.05,
) -> MCStep:
    """TEST step to perform a Breusch-Pagan test for heteroskedasticity on the specified residuals and regressors.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    residuals_source : str
        Source step name for the residuals.
    residuals_field : str
        Field name for the residuals in the source step.
    X_source : str
        Source step name for the regressors.
    X_field : str
        Field name for the regressors in the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    residual_col : ColumnSelector
        Column selector for the residuals in the source step. Can be a single column index.
    X_columns : ColumnSelector
        Columns of the regressors in the source field.
        Can be a single column index, a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source steps before applying the test.
    drop_initial : bool
        Whether to drop the initial sample from the test output.
    robust : bool
        Whether to use a robust version of the Breusch-Pagan test.
    alpha : float
        Significance level for the test.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _two_source_test(
        name,
        "breusch_pagan",
        first_source=residuals_source,
        first_field=residuals_field,
        first_arg="residuals",
        first_columns=residual_col,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"robust": robust, "alpha": alpha},
        n_retain=n_retain,
    )


def breusch_godfrey_test_step(
    name: str,
    n_retain: int = -1,
    *,
    residuals_source: str,
    residuals_field: str,
    X_source: str,
    X_field: str,
    residual_col: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 1,
    alpha: float = 0.05,
) -> MCStep:
    """TEST step to perform a Breusch-Godfrey test for autocorrelation on the specified residuals and independent variables.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    residuals_source : str
        Source step name for the residuals.
    residuals_field : str
        Field name for the residuals in the source step.
    X_source : str
        Source step name for the regressors.
    X_field : str
        Field name for the regressors in the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    residual_col : ColumnSelector
        Column selector for the residuals in the source step. Can be a single column index.
    X_columns : ColumnSelector
        Columns of the regressors in the source field.
        Can be a single column index, a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source steps before applying the test.
    drop_initial : bool
        Whether to drop the initial sample from the test output.
    lags : int
        Number of lags to include in the test. Must be a positive integer.
    alpha : float
        Significance level for the test.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _two_source_test(
        name,
        "breusch_godfrey",
        first_source=residuals_source,
        first_field=residuals_field,
        first_arg="residuals",
        first_columns=residual_col,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"lags": lags, "alpha": alpha},
        n_retain=n_retain,
    )


def cusum_test_step(
    name: str,
    n_retain: int = -1,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep:
    """TEST step to perform a CUSUM test on the specified dependent variable and regressors.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    y_source : str
        Source step name for the dependent variable.
    y_field : str
        Field name for the dependent variable in the source step.
    X_source : str
        Source step name for the regressors.
    X_field : str
        Field name for the regressors in the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    y_column : ColumnSelector
        Column selector for the dependent variable in the source step. Can be a single column index.
    X_columns : ColumnSelector
        Columns of the regressors in the source field.
        Can be a single column index, a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source steps before applying the test.
    drop_initial : bool
        Whether to drop the initial sample from the test output.
    alpha : float
        Significance level for the test.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _two_source_test(
        name,
        "cusum",
        first_source=y_source,
        first_field=y_field,
        first_arg="y",
        first_columns=y_column,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"alpha": alpha},
        n_retain=n_retain,
    )


def cusumsq_test_step(
    name: str,
    n_retain: int = -1,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep:
    """TEST step to perform a CUSUM of squares test on the specified dependent variable and regressors.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    y_source : str
        Source step name for the dependent variable.
    y_field : str
        Field name for the dependent variable in the source step.
    X_source : str
        Source step name for the regressors.
    X_field : str
        Field name for the regressors in the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    y_column : ColumnSelector
        Column selector for the dependent variable in the source step. Can be a single column index.
    X_columns : ColumnSelector
        Columns of the regressors in the source field.
        Can be a single column index, a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source steps before applying the test.
    drop_initial : bool
        Whether to drop the initial sample from the test output.
    alpha : float
        Significance level for the test.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _two_source_test(
        name,
        "cusumsq",
        first_source=y_source,
        first_field=y_field,
        first_arg="y",
        first_columns=y_column,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"alpha": alpha},
        n_retain=n_retain,
    )


def chow_test_step(
    name: str,
    n_retain: int = -1,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: ColumnSelector = None,
    X_columns: ColumnSelector = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    t_break: int = 10,
    alpha: float = 0.05,
) -> MCStep:
    """TEST step to perform a Chow test for structural breaks on the specified dependent variable and regressors.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    y_source : str
        Source step name for the dependent variable.
    y_field : str
        Field name for the dependent variable in the source step.
    X_source : str
        Source step name for the regressors.
    X_field : str
        Field name for the regressors in the source step.
    n_retain : int
        Number of samples to retain. -1 means all, 0 means none.
    y_column : ColumnSelector
        Column selector for the dependent variable in the source step. Can be a single column index.
    X_columns : ColumnSelector
        Columns of the regressors in the source field.
        Can be a single column index, a ``Sequence`` of indices, a ``slice`` object, or ``None``.
        ``None`` means all columns.
    burn_in : int
        Number of initial samples to discard from the source steps before applying the test.
    drop_initial : bool
        Whether to drop the initial sample from the test output.
    t_break : int
        Index of the suspected structural break point in the sample. Must be a positive integer.
    alpha : float
        Significance level for the test.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline`.
    """
    return _two_source_test(
        name,
        "chow",
        first_source=y_source,
        first_field=y_field,
        first_arg="y",
        first_columns=y_column,
        second_source=X_source,
        second_field=X_field,
        second_arg="X",
        second_columns=X_columns,
        burn_in=burn_in,
        drop_initial=drop_initial,
        kwargs={"t_break": t_break, "alpha": alpha},
        n_retain=n_retain,
    )


def postproc_step(name: str, func: Callable[..., Any], **kwargs: Any) -> MCStep:
    """Custom POSTPROC step applying a user-defined function to the run's traces.

    The callable is invoked once after the replication loop as
    ``func(traces=..., **kwargs)``. ``traces`` maps every across-replication trace
    key the pipeline emitted to its stacked array, not only the transform payloads:

    - ``"payload.<name>"`` for a transform step's stacked per-rep payload
    - ``"test.<name>.<sub>"`` with ``sub`` one of ``statistic``, ``pval``, ``status``
    - ``"regression.<name>.<sub>"`` with ``sub`` one of ``coef``, ``ssr``, ``sst``,
      ``se``, ``r2``, ``status``

    A payload key has no sub-channel, since a transform emits a single array.

    Returns must be wrapped in :class:`Raw` for bulk numeric arrays or
    :class:`Summary` for aggregate outputs such as scalars, vectors, matrices and
    ``DataFrame``/``Series``. A step may return one of each, either alone or in an
    order-invariant tuple of length two; two of the same kind is rejected.

    ``func`` is stored as given. Decorating it with :func:`pandas_operation` makes
    it a :class:`PandasCustomFunc`, which is what subjects it to the namespace
    restrictions and makes it shippable inside a bundle; an undecorated callable is
    passed through unwrapped and unvalidated. Either way a POSTPROC op stays
    Python, unlike the compiled custom TRANSFORM callables.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    func : Callable[..., Any]
        Callable to run over the assembled traces. Must accept a ``traces`` keyword
        and return :class:`Raw`, :class:`Summary`, or a tuple holding one of each.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline` as a post-loop step.
        A POSTPROC step cannot be used as a per-replication step.
    """
    return MCStep(
        name=name,
        op_type=OpType.POSTPROC,
        func=func,
        kwargs=kwargs,
        step_type="postproc:custom",
    )


def kde_step(
    name: str,
    *,
    trace: str,
    bandwidth: str | float = "scott",
    grid_points: int = 200,
    kernel: str = "gaussian",
) -> MCStep:
    """POSTPROC step computing a kernel density estimate over one across-rep trace.

    Parameters
    ----------
    name : str
        Step name, used to reference the step in later steps and results.
    trace : str
        Key of the trace to estimate over. A transform payload is keyed
        ``"payload.<name>"``; a test or regression sub-channel is keyed
        ``"test.<name>.<sub>"`` or ``"regression.<name>.<sub>"``. The trace is
        flattened and non-finite values are dropped before estimation, which needs
        at least two finite values.
    bandwidth : str | float
        Bandwidth selection rule or fixed bandwidth, forwarded to
        :class:`scipy.stats.gaussian_kde` as its ``bw_method`` argument. Accepts
        anything that argument does, including a callable.
    grid_points : int
        Number of evenly spaced points spanning the data range at which the density
        is evaluated. Must be a positive integer.
    kernel : str
        Kernel type. Only ``"gaussian"`` is currently implemented and anything else
        raises; the argument exists so a non-Gaussian kernel can be added without a
        signature change.

    Returns
    -------
    MCStep
        Step object to include in a :class:`MCPipeline` as a post-loop step. The
        step produces a :class:`Raw` artifact holding a ``(grid_points, 2)`` array
        of grid values against density, and a :class:`Summary` artifact holding a
        ``DataFrame`` of count, mean, standard deviation, minimum, quartiles,
        median and maximum for the estimated sample.
    """
    return MCStep(
        name=name,
        op_type=OpType.POSTPROC,
        func=run_kde,
        kwargs={
            "trace": trace,
            "bandwidth": bandwidth,
            "grid_points": grid_points,
            "kernel": kernel,
        },
        step_type="kde",
    )
