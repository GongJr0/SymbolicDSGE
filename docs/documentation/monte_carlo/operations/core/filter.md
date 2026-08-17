---
tags:
    - doc
---
# Reference Filtering

```python
reference_filter_step(
    name: str = "filter",
    *,
    filter_mode: Literal["linear", "extended", "unscented"] = "linear",
    observables: list[str] | None = None,
    x0: ndarray | None = None,
    P0: ndarray | None = None,
    R: ndarray | None = None,
    jitter: float | None = None,
    symmetrize: bool = True,
    joseph_cov: bool = True,
    return_shocks: bool = False,
) -> MCStep
```

`reference_filter_step` runs the reference model's Kalman filter over the data step's observables, once per replication, as a native kernel. It lives in `SymbolicDSGE.monte_carlo.step_factories`.

When `observables=None`, the observable names carried by the data step are used if available. If names are not available, the reference model's normal observable resolution applies.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| name | `"filter"` | Runtime step name. Downstream steps use this as `source`. |
| filter_mode | `"linear"` | Filter mode: `"linear"`, `"extended"`, or `"unscented"`. |
| observables | `None` | Observable names passed to `reference.kalman(...)`. |
| x0 | `None` | Initial state override. It is the prior for the first observation in linear and extended modes, and the state before the first observation in unscented mode. |
| P0 | `None` | Initial state covariance override with the same timing as `x0`. `None` uses the `P0` matrix from the reference model's `KalmanConfig`. |
| R | `None` | Measurement error covariance override. |
| jitter | `None` | Filter jitter override. |
| symmetrize | `True` | Symmetrize covariance matrices during filtering. |
| joseph_cov | `True` | Use the Joseph covariance update for linear and extended filtering. Set it to `False` for the simplified update, which is faster but less numerically robust. It does not affect unscented filtering. |
| return_shocks | `False` | Return shock estimates when supported by the selected filter mode. |

__Downstream Fields:__

| __Field__ | __Description__ |
|:----------|----------------:|
| `x_pred`, `x_filt` | Predicted and filtered model variable paths. |
| `constant` | State offset used by linear and extended results to report levels. It is `NaN` for unscented results, whose kernel forms levels itself. |
| `y_pred`, `y_filt` | Predicted and filtered observable paths. |
| `innov`, `std_innov` | Raw and standardized innovations. |
| `eps_hat` | Shock estimates for modes that support `return_shocks=True`. |
| `x1_pred`, `x2_pred`, `x1_filt`, `x2_filt` | Unscented first and second state blocks. |

???+ warning "Unscented shock estimates"
    `return_shocks=True` is not supported with `filter_mode="unscented"`.
