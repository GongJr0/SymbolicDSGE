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
| R | `#!python NDArray | None` | Numeric observation noise covariance matrix built from config parameters. |
| P0 | `#!python NDArray` | Diagonal initial-state covariance $P_0$ as a numeric matrix, built at parse time from the YAML `mode`/`diag` fields and stored in canonical (compiled) variable order. |
| R_symbolic | `#!python sympy.Matrix | None` | Symbolic expression of the configured full `R` matrix. |
| R_param_symbols | `#!python list[sympy.Symbol] | None` | Symbols required to build `R_symbolic`. |
| R_param_names | `#!python list[str] | None` | Parameter names (ordered) passed to `R_builder`. |
| R_builder | `#!python Callable[..., NDArray] | None` | Lambdified builder that reconstructs full `R` from `R_param_names`. |
| R_std_param_map | `#!python dict[str, str] | None` | Observable to measurement standard deviation parameter name map. |
| R_corr_param_map | `#!python dict[frozenset[str], str | None] | None` | Observable pair to measurement correlation parameter name map. Missing pairs are stored with `None`. |

??? info "Symbolic `R` Metadata"
    `R_symbolic`/`R_builder` are used by estimation pipelines (e.g. iterative MCMC updates) to rebuild `R` from the current parameter draw when needed.

???+ info "P0 Shape"
    `P0` is resolved to a numeric diagonal matrix at parse time. The YAML `kalman.P0` block accepts `mode` (`diag` uses the given `diag` values, `eye` uses an identity of the appropriate shape) and, for `diag` mode, a `diag` map of variable names to variances (not standard deviations). It carries no correlation fields.
