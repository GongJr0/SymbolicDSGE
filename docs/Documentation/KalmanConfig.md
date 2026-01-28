---
tags:
    - doc
---
# KalmanConfig

```python
@dataclass(frozen=True)
class KalmanConfig()
```

`KalmanConfig` stores the parsed Kalman Filter configuration.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| y_names | `#!python list[str]` | Names of the included observables. |
| R | `#!python NDArray | None` | Covariance matrix of observation noise. R is generated as a matrix from parsing the config and should not be overridden in the config object.|
| jitter | `#!python float` | Jitter term to be added to covariance matrices if Cholesky fails. |
| symmetrize | `#!python bool` | Symmetrize the covariance matrices per iteration if `True`. |
| P0 | `#!python P0Config` | `dataclass` storing the mode and values of the initial $P$ state. |


```python
@dataclass(frozen=True)
class P0Config()
```

`P0Config` stores the required parameters to construct the initial $P$ state.

???+ info "P0 Shape"
    Currently, any `P0` produced by `P0Config` is only populated in the diagonals no matter the configuration. (Zero correlation assumption) A P0 pipeline implementing `std` and `corr` fields to build a complete covariance matrix is a planned implementation.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| mode | `#!python str` | P0 creation mode. `diag` uses given diagonal values, `eye` uses an identity matrix of the appropriate shape. |
| scale | `#!python float` | Scaling factor of the P0 matrix. (`#!python P0 = P0 * scale`)  |
| diag | `#!python dict[str, float]` | Variable names and their diagonals (variances, not standard deviation) in the $P$ matrix. |
