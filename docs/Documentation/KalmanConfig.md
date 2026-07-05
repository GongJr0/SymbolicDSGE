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
| R | `#!python NDArray \| None` | Numeric observation noise covariance matrix built from config parameters. |
| P0 | `#!python P0Config` | `dataclass` storing the mode and values of the initial $P$ state. |
| R_symbolic | `#!python sympy.Matrix \| None` | Symbolic expression of the configured full `R` matrix. |
| R_param_symbols | `#!python list[sympy.Symbol] \| None` | Symbols required to build `R_symbolic`. |
| R_param_names | `#!python list[str] \| None` | Parameter names (ordered) passed to `R_builder`. |
| R_builder | `#!python Callable[..., NDArray] \| None` | Lambdified builder that reconstructs full `R` from `R_param_names`. |
| R_std_param_map | `#!python dict[str, str] \| None` | Observable to measurement standard deviation parameter name map. |
| R_corr_param_map | `#!python dict[frozenset[str], str \| None] \| None` | Observable pair to measurement correlation parameter name map. Missing pairs are stored with `None`. |

??? info "Symbolic `R` Metadata"
    `R_symbolic`/`R_builder` are used by estimation pipelines (e.g. iterative MCMC updates) to rebuild `R` from the current parameter draw when needed.


```python
@dataclass(frozen=True)
class P0Config()
```

`P0Config` stores the required parameters to construct the initial $P$ state.

???+ info "P0 Shape"
    `P0Config` supports diagonal and scaled identity initialization. It does not carry correlation fields.

__Fields:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| mode | `#!python str` | P0 creation mode. `diag` uses given diagonal values, `eye` uses an identity matrix of the appropriate shape. |
| scale | `#!python float` | Scaling factor of the P0 matrix. (`#!python P0 = P0 * scale`)  |
| diag | `#!python dict[str, float] \| None` | Variable names and their diagonals (variances, not standard deviation) in the $P$ matrix. |
