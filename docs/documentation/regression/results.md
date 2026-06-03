---
tags:
    - doc
---
# Regression Results

```python
from SymbolicDSGE.regression import RegressionResult
from SymbolicDSGE.regression.ols import MCRegressionResult
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

&nbsp;

```python
@dataclass(frozen=True)
class MCRegressionResult(
    variables: list[str],
    results: tuple[RegressionResult, ...],
)
```

`MCRegressionResult` aggregates per-replication `RegressionResult` objects produced by Monte Carlo regression steps.

__Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| variables | `#!python list[str]` | Shared variable ordering across replications. |
| results | `#!python tuple[RegressionResult, ...]` | Per-replication regression outputs. |
| coef_trace | `#!python ndarray` | Coefficients stacked by replication. Shape `(n_rep, k)`. |
| coefficients | `#!python ndarray` | Alias for `coef_trace`. |
| status_trace | `#!python tuple[RegressionStatus, ...]` | Solver status for each replication. |
| n_rep | `#!python int` | Number of stored regression results. |
| n | `#!python int` | Shared number of observations per replication. |
| k | `#!python int` | Shared number of design columns. |
| y_trace | `#!python ndarray` | Response vectors stacked by replication. |
| x_trace | `#!python ndarray` | Design matrices stacked by replication. |
| y_hat_trace | `#!python ndarray` | Fitted responses stacked by replication. |
| residual_trace | `#!python ndarray` | Residual vectors stacked by replication. |
| ssr_trace | `#!python ndarray` | Per-replication SSR values. |
| sst_trace | `#!python ndarray` | Per-replication SST values. |
| mse_trace | `#!python ndarray` | Per-replication MSE values. |
| rmse_trace | `#!python ndarray` | Per-replication RMSE values. |
| r2_trace | `#!python ndarray` | Per-replication R-squared values. |
| r2_adj_trace | `#!python ndarray` | Per-replication adjusted R-squared values. |
| `summary(alpha=0.05)` | `#!python pandas.DataFrame` | Coefficient trace summary. OLS results include inference columns. |
| `to_dict()` | `#!python dict` | Compact dictionary representation. |

__OLS-Only Aggregate Diagnostics:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| se_trace | `#!python ndarray` | OLS standard-error trace. |
| t_stat_trace | `#!python ndarray` | OLS t-statistic trace. |
| partial_r2_trace | `#!python ndarray` | OLS partial R-squared trace. |
| pval_trace | `#!python ndarray` | OLS coefficient p-value trace. |
| F_stat_trace | `#!python ndarray` | OLS F-statistic trace. |
| F_pval_trace | `#!python ndarray` | OLS F-test p-value trace. |
| `confidence_intervals(alpha=0.05)` | `#!python ndarray` | Per-replication OLS coefficient intervals. |
| `F_test(alpha=0.05)` | `#!python MCResult` | Aggregate F-test result container. |

???+ note "OLS-Specific Diagnostics"
    OLS aggregate diagnostics require every stored result to be an `OLSResult`. For ridge, lasso, and elastic-net aggregates, `summary()` returns coefficient traces without OLS inference columns.
