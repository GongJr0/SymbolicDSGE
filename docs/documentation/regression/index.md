---
tags:
    - doc
---
# Regression

```python
from SymbolicDSGE import regression
from SymbolicDSGE.regression import (
    RegressionKind,
    RegressionResult,
    RegressionStatus,
)
```

The `regression` subpackage contains the standard linear regression interfaces used directly by users and by Monte Carlo regression steps.

__Public Modules:__

| __Module__ | __Description__ |
|:-----------|----------------:|
| `regression.ols` | Ordinary Least Squares regression and OLS inference diagnostics. |
| `regression.ridge` | Ridge regression and L2 grid search. |
| `regression.lasso` | Lasso regression and Lasso grid/path utilities. |
| `regression.elastic_net` | Elastic Net regression and grid search. |
| `regression.sr` | Symbolic-regression utilities. |

__RegressionKind Values:__

| __Member__ | __String__ | __Dispatch Target__ |
|:-----------|:----------:|--------------------:|
| `OLS` | `"ols"` | `regression.ols.ols` |
| `RIDGE` | `"ridge"` | `regression.ridge.ridge` |
| `RIDGE_GS` | `"ridge_gs"` | `regression.ridge.ridge_gs` |
| `LASSO` | `"lasso"` | `regression.lasso.lasso` |
| `LASSO_GS` | `"lasso_gs"` | `regression.lasso.lasso_gs` |
| `ELASTIC_NET` | `"elastic_net"` | `regression.elastic_net.elastic_net` |
| `ELASTIC_NET_GS` | `"elastic_net_gs"` | `regression.elastic_net.elastic_net_gs` |

__RegressionStatus Values:__

| __Member__ | __Code__ | __Description__ |
|:-----------|:--------:|----------------:|
| `OK` | `0` | Regression solver completed normally. |
| `RANK_DEFICIENT` | `-1` | The primary linear solver detected rank deficiency and a fallback or failure path was used. |
| `NON_CONVERGENT` | `-2` | An iterative solver exhausted its iteration budget before satisfying tolerance. |

???+ note "Intercept Convention"
    Regression functions use `intercept=True` by default. When enabled, the returned design matrix contains an `Intercept` column as the first variable. Penalized methods do not penalize this intercept term.

