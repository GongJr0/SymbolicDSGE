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
| R | `#!python NDArray \| None` | Numeric observation-noise covariance matrix (full matrix) built from config parameters. |
| jitter | `#!python float \| None` | Jitter term for Cholesky-failure fallback (`None` defers to runtime defaults). |
| symmetrize | `#!python bool \| None` | Symmetrization option (`None` defers to runtime defaults). |
| P0 | `#!python P0Config` | `dataclass` storing the mode and values of the initial $P$ state. |
| R_symbolic | `#!python sympy.Matrix \| None` | Symbolic expression of the configured full `R` matrix. |
| R_param_symbols | `#!python list[sympy.Symbol] \| None` | Symbols required to build `R_symbolic`. |
| R_param_names | `#!python list[str] \| None` | Parameter names (ordered) passed to `R_builder`. |
| R_builder | `#!python Callable[..., NDArray] \| None` | Lambdified builder that reconstructs full `R` from `R_param_names`. |

??? info "Symbolic `R` Metadata"
    `R_symbolic`/`R_builder` are used by estimation pipelines (e.g. iterative MCMC updates) to rebuild `R` from the current parameter draw when needed.


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
| diag | `#!python dict[str, float] \| None` | Variable names and their diagonals (variances, not standard deviation) in the $P$ matrix. |
