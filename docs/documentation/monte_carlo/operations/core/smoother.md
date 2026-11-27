---
tags:
    - doc
---
# Smoother

```python
smoother_step(
    name: str = "smoother",
    n_retain: int = -1,
    *,
    target: str,
    obs_source: str,
    obs_field: str,
    obs_columns: ColumnSelector = None,
    filter: Filter = LinearKalman(),
    observables: list[str] | None = None,
    x0: dict[str, float | float64] | list[float | float64] | ndarray | None = None,
    P0: ndarray | None = None,
    R: ndarray | None = None,
) -> MCStep
```

`smoother_step` runs the smoother of the selected `Filter` on the `target` model using the selected observable source. It lives in `SymbolicDSGE.monte_carlo.step_factories`. A `smoother_step` provides every output that a `filter_step` does, plus the smoothed counterparts.

???+ note "Observation alignment"
    `observables` names the selected data columns in their supplied order; lowering reorders them into the target model's canonical order. With `None`, columns must match the model's full observation order. Producer metadata is not used to infer names.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| name | `"smoother"` | Runtime step name. Downstream steps use this as `source`. |
| target | Required | Model name supplying filter components. Must be supplied to `MCPipeline.run(...)`. |
| obs_source | Required | Producer step supplying observed data. |
| obs_field | Required | Two-dimensional output field to read from the producer. |
| obs_columns | `None` | Column index, sequence of indices, or slice; `None` selects all columns. |
| filter | `LinearKalman()` | Filter to use. Can be any `Filter` subclass that's suitable with the target model. |
| n_retain | `-1` | Number of replications to retain. `-1` retains all replications. |
| observables | `None` | Target-model observable names corresponding to the selected columns, in order. |
| x0 | `None` | Initial state override in levels. It is the prior for the first observation in linear and extended modes, and the state before the first observation in unscented mode. |
| P0 | `None` | Initial state covariance override with the same timing as `x0`. `None` uses the stationary state covariance. |
| R | `None` | Measurement error covariance override. `None` uses the calibration's `measurement_noise` block to build `R`. |

__Downstream Fields:__

| __Field__ | __Description__ |
|:----------|----------------:|
| `x_pred`, `x_filt` | Predicted and filtered model variable paths. |
| `y_pred`, `y_filt` | Predicted and filtered observable paths. |
| `innov`, `std_innov` | Raw and standardized innovations. |
| `eps_hat` | Shock estimates. |
| `x1_pred`, `x2_pred`, `x1_filt`, `x2_filt` | Unscented first and second state blocks. |
| `x_smooth` | Smoothed model variable paths. |
| `eps_smooth` | Smoothed shock estimates. |
| `x1_smooth`, `x2_smooth` | Unscented first and second smoothed state blocks. |
