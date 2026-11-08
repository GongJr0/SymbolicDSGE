---
tags:
    - doc
---
# LinearKalman

```python
from SymbolicDSGE.filters import LinearKalman
```

```python
@dataclass
class LinearKalman(GaussianFilter[FilterResult])
```

The linear Gaussian Kalman filter. Takes the affine measurement $y_t = C x_t + d$ assembled from the model's observables.

__Fields:__

| __Name__ | __Type__ | __Default__ | __Description__ |
|:---------|:--------:|:-----------:|----------------:|
| jitter | `float` | `0.0` | Amount added to a covariance diagonal when a Cholesky factorization fails. |
| symmetrize | `bool` | `False` | Symmetrize the covariance matrices via their transposes `0.5 * (M + M.T)`. |
| joseph_cov | `bool` | `False` | Use the Joseph form for the covariance update. |
| return_shocks | `bool` | `False` | Recover the structural shocks implied by the filtered path into `eps_hat`. |

???+ note "Joseph Form"
    The Joseph form computes the posterior covariance as $(I - KC) P (I - KC)^\top + K R K^\top$, which stays symmetric and positive semidefinite under rounding where the short form need not. It costs one additional matrix product per period and is disabled by default.

???+ tip "Shock Recovery"
    `return_shocks=True` populates `eps_hat` on the result with the structural shocks implied by the filtered path.
&nbsp;

```python
filter(
    model: SolvedModel,
    y: ndarray | DataFrame,
    observables: list[str] | tuple[str, ...] | None = None,
    R: ndarray | None = None,
    x0: Mapping[str, float] | None = None,
    P0: ndarray | None = None,
) -> FilterResult
```

Run the filter over `y`.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| model | Solved model at first order or higher. |
| y | Observed data. Shape `(T, n_obs)`, or a `DataFrame` whose columns name the observables. |
| observables | Model observables the data corresponds to. Defaults to all of them in declaration order. |
| R | Measurement noise covariance. Shape `(n_obs, n_obs)`. Defaults to the model's `calibration.measurement_noise`. |
| x0 | Prior state mean, in levels. Defaults to the steady state. |
| P0 | Prior state covariance. Shape `(n_var, n_var)`. Defaults to the model's stationary covariance. |


__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `FilterResult` | Predicted and filtered states, covariances, observables, innovations, and the log likelihood. |
