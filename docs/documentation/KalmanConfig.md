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
| R_param_names | `#!python list[str] | None` | Parameter names (ordered) passed to `R_builder`. |
| R_std_param_map | `#!python dict[str, str] | None` | Observable to measurement standard deviation parameter name map. |
| R_corr_param_map | `#!python dict[frozenset[str], str | None] | None` | Observable pair to measurement correlation parameter name map. Missing pairs are stored with `None`. |

??? info "Symbolic `R` Metadata"
    `R_symbolic`/`R_builder` are used by estimation pipelines (e.g. iterative MCMC updates) to rebuild `R` from the current parameter draw when needed.

