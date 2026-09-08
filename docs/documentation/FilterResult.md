---
tags:
    - doc
---
# FilterResult

```python
@dataclass(frozen=True)
class FilterResult()
```

Stores filter outputs and relevant descriptors of state/observable variables.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| x_pred | `#!python ndarray` | Predicted $x$ states over time. |
| x_filt | `#!python ndarray` | Filtered $x$ states over time. |
| P_pred | `#!python ndarray` | Predicted state covariance $P$ over time. |
| P_filt | `#!python ndarray` | Filtered state covariance $P$ over time. |
| y_pred | `#!python ndarray` | Predicted observables over time. |
| y_filt | `#!python ndarray` | Filtered observables over time. |
| innov | `#!python ndarray` | Observable innovations $y_t - y_{t\mid t-1}$. |
| std_innov | `#!python ndarray` | Innovations standardized by their covariance. |
| S | `#!python ndarray` | Innovation covariance over time. |
| eps_hat | `#!python ndarray | None` | Conditional estimates of structural shocks given observed data (present when `return_shocks=True`). |
| loglik | `#!python float` | log likelihood ($\boldsymbol{\ell}$) of measurements. |

???+ info "State Units"
    `SolvedModel.kalman(...)` returns state paths in levels. The solved steady-state vector is added to `x_pred` and `x_filt` before reporting outputs. Measurements are in levels during the recursion, as required for comparing to observed data. 

    Unscented filtering forms levels inside its kernel by construction; its recursion also necessitates `x_pred` and `x_filt` to be in levels. 

???+ info "Initial State Timing"
    Linear and extended filtering read `x0` and `P0` as the prior mean and covariance for the first observed state. Unscented filtering reads them as the state and covariance before the first observation.

`x_pred` is the state estimate before incorporating its date's observation, while `x_filt` is the estimate after incorporation. `P_pred` and `P_filt` are the matching covariance histories.

&nbsp;

# UnscentedFilterResult

```python
@dataclass(frozen=True)
class UnscentedFilterResult(FilterResult)
```

Returned by `#!python SolvedModel.kalman(filter_mode="unscented")` and `#!python KalmanFilter.run_unscented(...)`. Extends `FilterResult` with the first- and second-order components of the unscented state estimate. All `FilterResult` fields are present; `#!python eps_hat` is `#!python None`, since shock recovery (`return_shocks`) is not supported for the UKF.

__Additional Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| x1_pred | `#!python ndarray` | First-order component of the predicted state over time. |
| x2_pred | `#!python ndarray` | Second-order component of the predicted state over time. |
| x1_filt | `#!python ndarray` | First-order component of the filtered state over time. |
| x2_filt | `#!python ndarray` | Second-order component of the filtered state over time. |
