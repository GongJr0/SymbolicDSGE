---
tags:
    - doc
---
# Built-In Steps

## DGP Simulation

```python
simulation_step(
    name: str = "datagen",
    *,
    T: int,
    shocks: Mapping[str, Shock | Callable | ndarray] | None = None,
    seed_increment: int | Literal["auto"] = "auto",
    shock_scale: float = 1.0,
    x0: ndarray | None = None,
    observables: bool = True,
) -> MCStep
```

`simulation_step` wraps `simulate_dgp(...)`. It requires `MCPipeline.run(..., dgp=...)` and calls `dgp.sim(...)` in each replication.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| T | Number of simulated periods, excluding the initial state. |
| shocks | Shock specification passed into DGP simulation after per-replication seed handling. |
| seed_increment | Integer seed offset per replication, or `"auto"` to increment by the number of seeded `Shock` objects. |
| shock_scale | Shock scaling passed into `SolvedModel.sim(...)`. |
| x0 | Optional initial state. |
| observables | If `True`, observable paths are included in `MCData.observables`. |

???+ info "Seed Convention"
    For generator-style `Shock` objects with integer seeds, replication `rep_idx` receives `shock.seed + rep_idx * seed_increment`. With `seed_increment="auto"`, the increment is the number of seeded `Shock` entries. Array shocks and callable shocks are passed through unchanged.

## Raw Data

```python
raw_data_step(
    name: str = "datagen",
    *,
    states: ndarray | None = None,
    observables: ndarray | None = None,
    n_exog: int = -1,
    raw: Mapping[str, ndarray] | None = None,
    observable_names: Sequence[str] = (),
) -> MCStep
```

`raw_data_step` wraps `raw_data_datagen(...)`. It does not require a DGP model.

__Accepted Shapes:__

| __Field__ | __Accepted Shapes__ | __Description__ |
|:----------|:--------------------:|----------------:|
| states | `(T, n_state)` or `(n_rep, T, n_state)` | A 2D array is reused for every replication. A 3D array selects `arr[rep_idx]`. |
| observables | `(T,)`, `(T, n_obs)`, or `(n_rep, T, n_obs)` | A 1D observable path is reshaped to `(T, 1)`. A 2D array is reused for every replication. |
| raw values | `(T,)`, `(T, n)`, or `(n_rep, T, n)` | Extra raw arrays are selected with the same convention as observables. |

???+ note "State-Only and Observable-Only Runs"
    `raw_data_step` accepts state-only and observable-only payloads. A reference filter step requires observables, while tests using `source="states"` require states.

## Reference Filtering

```python
reference_filter_step(
    name: str = "filter",
    *,
    filter_mode: Literal["linear", "extended"] = "linear",
    observables: list[str] | None = None,
    x0: ndarray | None = None,
    p0_mode: Literal["diag", "eye"] | None = None,
    p0_scale: float | None = None,
    jitter: float | None = None,
    symmetrize: bool | None = None,
    return_shocks: bool = False,
    R: ndarray | None = None,
    estimate_R_diag: bool = False,
    R_scale: float = 1.0,
) -> MCStep
```

`reference_filter_step` wraps `run_reference_filter(...)`. It reads `context.data.observables` and calls `reference.kalman(...)`.

When `observables=None`, generated `MCData.observable_names` are used if available. If names are not available, `reference.kalman(...)` falls back to its normal observable resolution.

## Wald Testing

```python
wald_test_step(
    name: str,
    *,
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
    target: ndarray,
    kind: Literal["mean", "covariance", "second_moment"] = "mean",
    filter_key: str = "filter",
    payload_key: str | None = None,
    columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    kernel: Literal["bartlett", "parzen", "qs"] = "bartlett",
    bandwidth: int | Literal["andrews", "wooldridge", "auto"] | None = "auto",
    alpha: float = 0.05,
) -> MCStep
```

`wald_test_step` wraps `run_wald_test(...)`. It selects a 1D or 2D array from generated data, a `FilterResult`, or a named payload, then runs the requested HAC Wald diagnostic.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |

__Kinds:__

| __Kind__ | __Description__ |
|:---------|----------------:|
| `mean` | Tests `E[g_t] = target`. |
| `covariance` | Tests the vech representation of the covariance matrix against `target`. |
| `second_moment` | Tests the vech representation of the raw second moment against `target`. |

## Transform Steps

```python
transform_step(
    name: str,
    func: Callable[..., Any],
    *,
    store_key: str | None = None,
    **kwargs: Any,
) -> MCStep
```

`transform_step` creates a custom per-replication operation. The callable receives the normalized operation arguments:

```python
func(
    *,
    context: MCContext,
    reference: SolvedModel,
    dgp: SolvedModel | None,
    rep_idx: int,
    **kwargs,
) -> Any
```

If the callable returns an `MCData` object, the pipeline replaces `context.data` with that return value. The return value is stored in `context.payloads[store_key or name]`.
