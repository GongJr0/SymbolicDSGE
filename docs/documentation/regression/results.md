---
tags:
    - doc
---
# Regression Results

```python
from SymbolicDSGE.regression import RegressionResult
```

&nbsp;

```python
@dataclass(frozen=True)
class RegressionResult(
    variables: list[str],
    coefficients: ndarray,
    y: ndarray,
    X: ndarray,
    status: RegressionStatus,
)
```

`RegressionResult` is the shared result abstraction for standard linear regression outputs. Concrete result types inherit the common fitted-data diagnostics and add method-specific quantities.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| variables | `#!python list[str]` | Names of the design columns represented in `coefficients`. |
| coefficients | `#!python ndarray` | Estimated coefficient vector. Shape `(k,)`. |
| y | `#!python ndarray` | Response vector. Shape `(n,)`. |
| X | `#!python ndarray` | Design matrix used by the regression. Shape `(n, k)`. |
| x | `#!python ndarray` | Alias for `X`. |
| status | `#!python RegressionStatus` | Solver status. |
| n | `#!python int` | Number of observations. |
| k | `#!python int` | Number of design columns. |
| y_hat | `#!python ndarray` | Fitted response vector. |
| residuals | `#!python ndarray` | Response residuals, `y - y_hat`. |
| ssr | `#!python float64` | Sum of squared residuals. |
| sst | `#!python float64` | Total sum of squares around the sample mean of `y`. |
| mse | `#!python float64` | Mean squared error, `ssr / n`. |
| rmse | `#!python float64` | Root mean squared error. |
| r2 | `#!python float64` | Coefficient of determination. |
| r2_adj | `#!python float64` | Adjusted coefficient of determination. |
| `to_dict()` | `#!python dict` | Dataclass dictionary representation. |

???+ warning "Shape Contract"
    `RegressionResult` expects a one-dimensional response vector and a two-dimensional design matrix. Multivariate response regressions should be represented as separate result objects.

???+ note "Monte Carlo aggregates"
    `MCRegressionResult`, the container a Monte Carlo regression step produces, is documented alongside `MCTestResult` in [Monte Carlo Results](../monte_carlo/results.md).
