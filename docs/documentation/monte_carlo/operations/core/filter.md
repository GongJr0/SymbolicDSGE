---
tags:
    - doc
---
# Filtering

```python
filter_step(
    name: str = "filter",
    n_retain: int = -1,
    *,
    target: str,
    obs_source: str,
    obs_field: str,
    obs_columns: ColumnSelector = None,
    filter_mode: Literal["linear", "extended", "unscented"] = "linear",
    observables: list[str] | None = None,
    x0: dict[str, float | float64] | list[float | float64] | NDF | None = None,
    P0: ndarray | None = None,
    R: ndarray | None = None,
    jitter: float | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = False, 
    return_shocks: bool = False,
) -> MCStep
```

`filter_step` runs the target model's Kalman filter over a selected producer field once per replication as a native kernel. It lives in `SymbolicDSGE.monte_carlo.step_factories`.

???+ note "Observation alignment"
    `observables` names the selected data columns in their supplied order; lowering reorders them into the target model's canonical order. With `None`, columns must match the model's full observation order. Producer metadata is not used to infer names.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| name | `"filter"` | Runtime step name. Downstream steps use this as `source`. |
| target | Required | Model name supplying filter components. Must be supplied to `MCPipeline.run(...)`. | 
| obs_source | Required | Producer step supplying observed data. |
| obs_field | Required | Two-dimensional output field to read from the producer. |
| obs_columns | `None` | Column index, sequence of indices, or slice; `None` selects all columns. |
| filter_mode | `"linear"` | Filter mode: `"linear"`, `"extended"`, or `"unscented"`. |
| n_retain | `-1` | Number of replications to retain. `-1` retains all replications. |
| observables | `None` | Target-model observable names corresponding to the selected columns, in order. |
| x0 | `None` | Initial state override in levels. It is the prior for the first observation in linear and extended modes, and the state before the first observation in unscented mode. |
| P0 | `None` | Initial state covariance override with the same timing as `x0`. `None` uses the `P0` matrix from the target model's `KalmanConfig`. |
| R | `None` | Measurement error covariance override. |
| jitter | `None` | Filter jitter override. |
| symmetrize | `True` | Symmetrize covariance matrices during filtering. |
| joseph_cov | `False` | Use the Joseph covariance update for linear and extended filtering. `False` is faster but less robust. |
| return_shocks | `False` | Return shock estimates when supported by the selected filter mode. |

__Downstream Fields:__

| __Field__ | __Description__ |
|:----------|----------------:|
| `x_pred`, `x_filt` | Predicted and filtered model variable paths. |
| `y_pred`, `y_filt` | Predicted and filtered observable paths. |
| `innov`, `std_innov` | Raw and standardized innovations. |
| `eps_hat` | Shock estimates for modes that support `return_shocks=True`. |
| `x1_pred`, `x2_pred`, `x1_filt`, `x2_filt` | Unscented first and second state blocks. |

???+ warning "Unscented shock estimates"
    `return_shocks=True` is not supported with `filter_mode="unscented"`.
