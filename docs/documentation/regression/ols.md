---
tags:
    - doc
---
# OLS

```python
from SymbolicDSGE.regression.ols import OLSResult, ols
```

```python
ols(
    x: ndarray,
    y: ndarray,
    variables: list[str] | None = None,
    intercept: bool = True,
) -> OLSResult
```

Run Ordinary Least Squares regression on a one-dimensional response and a two-dimensional design matrix.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| x | Design matrix. Shape `(n, k)`. |
| y | Response vector. Shape `(n,)`. |
| variables | Optional names for the columns of `x`. Defaults to `x0`, `x1`, ... |
| intercept | If `True`, prepend an intercept column to the design matrix. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python OLSResult` | Regression result with common fitted diagnostics and OLS inference outputs. |

&nbsp;

```python
@dataclass(frozen=True)
class OLSResult(RegressionResult)
```

`OLSResult` extends `RegressionResult` with standard errors, coefficient tests, confidence intervals, and an F-test.

__Additional Fields and Methods:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| se | `#!python ndarray` | Coefficient standard errors. |
| t_stat | `#!python ndarray` | Coefficient t-statistics. |
| partial_r2 | `#!python ndarray` | Partial R-squared values implied by each t-statistic. |
| p_values | `#!python ndarray` | Two-sided coefficient p-values under the t reference distribution. |
| `confidence_intervals(alpha=0.05)` | `#!python ndarray` | Lower and upper coefficient bounds. Shape `(k, 2)`. |
| `summary(alpha=0.05)` | `#!python pandas.DataFrame` | Coefficient, interval, t-statistic, p-value, and partial R-squared table. |
| `F_test(alpha=0.05)` | `#!python TestResult` | Regression F-test against the relevant F reference distribution. |

???+ note "Rank-Deficient Designs"
    OLS first attempts a Cholesky solve and falls back to least squares when needed. The `status` field records the solver status used by the result.

