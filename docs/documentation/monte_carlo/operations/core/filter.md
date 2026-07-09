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

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| name | `"filter"` | Runtime step name. Downstream steps use this as `source`. |
| filter_mode | `"linear"` | Filter mode: `"linear"`, `"extended"`, or `"unscented"`. |
| observables | `None` | Observable names passed to `reference.kalman(...)`. |
| x0 | `None` | Initial state override. |
| p0_mode | `None` | Initial covariance construction mode. |
| p0_scale | `None` | Initial covariance scale override. |
| jitter | `None` | Filter jitter override. |
| symmetrize | `None` | Symmetrization override. |
| return_shocks | `False` | Return shock estimates when supported by the selected filter mode. |
| R | `None` | Measurement error covariance override. |
| estimate_R_diag | `False` | Estimate diagonal measurement error covariance from data. |
| R_scale | `1.0` | Scale applied to measurement error covariance. |

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
