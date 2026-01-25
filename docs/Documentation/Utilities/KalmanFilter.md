---
tags:
    - doc
---
# KalmanFilter
??? info "Kalman Filter Wiki Page"
    You can refer to the Wikipedia page for derivations and the underlying process. The documentation only includes the user-facing interface.

    [Kalman Filter | Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter){ .md-button }

```python
@dataclass(frozen=True)
class FilterResult()
```

`#!python dataclass` storing the results of a **Kalman Filter** application.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:---------:|-----------------:|
| x_pred | `#!python np.ndarray` | State prediction before observing $y_t$. |
| x_filt | `#!python np.ndarray` | State estimate after observing $y_t$. |
| P_pred | `#!python np.ndarray` | Predicted state covariance $P_{t\mid t-1}$
| P_filt | `#!python np.ndarray` | Filtered state covariance $P_{t\mid t}$. |
| y_pred | `#!python np.ndarray` | Predicted observation. Shape `(T, m)`|
| innov | `#!python np.ndarray` | Innovations (measurement residuals). Shape `(T, m)`.|
| S | `#!python np.ndarray` | Innovation covariance. Shape `(T, m, m)`.
| eps_hat | `#!python np.ndarray`, optional | Estimated shocks. |
| loglik | `#!python float`, optional | Total log-likelihood of the observed data under the filter. |

&nbsp;

```python
class KalmanFilter()
```
Static class containing methods necessary for filter application.

__Methods:__

```python
KalmanFilter.run(
    A: np.ndarray[float64 | complex128],
    B: np.ndarray[float64 | complex128],
    C: np.ndarray[float64 | complex128],
    d: np.ndarray[float64 | complex128],
    Q: np.ndarray[float64 | complex128],
    R: np.ndarray[float64 | complex128],
    y: np.ndarray[float64 | complex128],
    x0: np.ndarray[float64] | None = None,
    P0: np.ndarray[float64] | None = None,
    return_shocks: bool = False,
    symmetrize: bool = True,
    jitter: float = 0.0,
) -> FilterResult
```
Apply a Kalman Filter using the given inputs and return a `#!python FilterResult` object.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| A | State transition matrix. |
| B | Shock loading matrix. |
| C | Observation matrix. |
| d | Observation intercept. |
| Q | Shock covariance matrix. |
| R | Observation noise covariance matrix. |
| y | Observed data over time. |
| x0 (optional) | Initial state mean. Defaults to zero vector. |
| P0 (optional) | Initial state covariance. Defaults to large diagonal. |
| return_shocks | If `#!python True`, compute and return estimated shocks.
| symmetrize | Symmetrize the covariance matrices at each step if `#!python True` |
| jitter | Jitter term added to $S_t$ if **Cholesky** solution fails. |

???+ note "Jitter"
    Though its default is 0.0, running the method with a small jitter is strongly recommended. Using 1e-8 (or similar) can prevent fallback to matrix inversion when Cholesky fails. (Inversion much slower in comparison)

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python FilterResult` | `#!python dataclass` containing results of the Kalman Filter run. |
