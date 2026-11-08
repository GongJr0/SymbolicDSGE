---
tags:
    - doc
---
# UnscentedKalman

```python
from SymbolicDSGE.filters import UnscentedKalman
```

```python
@dataclass
class UnscentedKalman(GaussianFilter[UnscentedFilterResult])
```

The unscented Kalman filter. Propagates sigma points through the second-order policy and the compiled measurement function.

__Fields:__

| __Name__ | __Type__ | __Default__ | __Description__ |
|:---------|:--------:|:-----------:|----------------:|
| jitter | `float` | `0.0` | Amount added to a covariance diagonal when a Cholesky factorization fails. |
| symmetrize | `bool` | `True` | Symmetrize the covariance matrices via their transposes `0.5 * (M + M.T)`. |
| alpha | `float` | `1.0` | Spread of the sigma points around the mean. |
| beta | `float` | `2.0` | Weight of the central sigma point in the covariance. `2.0` is optimal for a Gaussian. |
| kappa | `float` | `1.0` | Secondary scaling parameter. |

???+ note "Hyperparameter Validity"
    `alpha` and `kappa` jointly determine the sigma-point weights, and not every pair produces positive ones. A combination that does not raises `HyperparameterError` following a kernel rejection.

&nbsp;

```python
filter(
    model: SolvedModel,
    y: ndarray | DataFrame,
    observables: list[str] | tuple[str, ...] | None = None,
    R: ndarray | None = None,
    x0: Mapping[str, float] | None = None,
    P0: ndarray | None = None,
) -> UnscentedFilterResult
```

Run the filter over `y`.

???+ warning "Requires a Second-Order Solution"
    The filter requires second-order policy tensors. Supplying a first-order solved model raises `ValueError`. Solve at order two before filtering.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| model | Solved model at second order. |
| y | Observed data. Shape `(T, n_obs)`, or a `DataFrame` whose columns name the observables. |
| observables | Model observables the data corresponds to. Defaults to all of them in declaration order. |
| R | Measurement noise covariance. Shape `(n_obs, n_obs)`. Defaults to the model's `calibration.measurement_noise`. |
| x0 | Prior state mean, in levels. Only its state block is read. Defaults to the steady state. |
| P0 | Prior state covariance. Shape `(n_var, n_var)`. Defaults to the model's stationary covariance. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `UnscentedFilterResult` | A `FilterResult` extended with the first- and second-order components of the state estimate. |


???+ note "Pruned State Layout"
    The state is carried as a stacked first- and second-order pair of length `2 * n_state`. `x1_pred`, `x2_pred`, `x1_filt`, and `x2_filt` on the result expose the two components separately.
    The `P0` specified on the `filter` call covers the first block only, which is equivalent to the `P0` on a linear or extended filter.
