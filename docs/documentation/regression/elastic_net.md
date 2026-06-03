---
tags:
    - doc
---
# Elastic Net

```python
from SymbolicDSGE.regression.elastic_net import (
    ElasticNetResult,
    elastic_net,
    elastic_net_gs,
)
```

```python
elastic_net(
    X: ndarray,
    y: ndarray,
    alpha: float,
    l1_ratio: float,
    variables: list[str] | None = None,
    intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> ElasticNetResult
```

Run Elastic Net regression with a combined L1/L2 penalty.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| X | Design matrix. Shape `(n, k)`. |
| y | Response vector. Shape `(n,)`. |
| alpha | Non-negative total penalty weight. |
| l1_ratio | Penalty split between L1 and L2. Must be in `[0, 1]`. |
| variables | Optional names for the columns of `X`. Defaults to `x0`, `x1`, ... |
| intercept | If `True`, fit an unpenalized intercept by centering and restore it in the returned design. |
| max_iter | Maximum coordinate-descent iterations. |
| tol | Coordinate-descent convergence tolerance. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python ElasticNetResult` | Regression result with combined penalty diagnostics. |

&nbsp;

```python
elastic_net_gs(
    X: ndarray,
    y: ndarray,
    start: float,
    stop: float,
    num: int,
    l1_ratio: float,
    criterion: Literal["aic", "bic", "loss"] = "loss",
    variables: list[str] | None = None,
    intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> ElasticNetResult
```

Run Elastic Net regression over a logarithmic alpha grid and return the best result under the selected criterion.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| start | Positive lower endpoint for the alpha grid. |
| stop | Positive upper endpoint for the alpha grid. |
| num | Number of grid points. |
| criterion | Grid-search objective: AIC, BIC, or residual loss. |
| X, y, l1_ratio, variables, intercept, max_iter, tol | Same contract as `elastic_net(...)`. |

&nbsp;

```python
@dataclass(frozen=True)
class ElasticNetResult(RegressionResult)
```

__Additional Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| alpha | `#!python float64` | Selected total penalty weight. |
| l1_ratio | `#!python float64` | L1 share of the total penalty. |
| effective_dof | `#!python float64` | Effective degrees of freedom for the selected solution. |
| intercept | `#!python bool` | Whether the returned design includes an intercept column. |
| alpha_grid | `#!python ndarray | None` | Grid of alpha values evaluated by `elastic_net_gs(...)`. |
| coefficient_path | `#!python ndarray | None` | Coefficients evaluated on `alpha_grid`. |
| objective_trace | `#!python ndarray | None` | Grid-search criterion trace. |
| rss_trace | `#!python ndarray | None` | Residual-sum-of-squares trace over `alpha_grid`. |
| effective_dof_trace | `#!python ndarray | None` | Effective degrees of freedom over `alpha_grid`. |
| status_trace | `#!python ndarray | None` | Solver status code over `alpha_grid`. |
| penalized_coefficients | `#!python ndarray` | Coefficients subject to the penalty. |
| active_mask | `#!python ndarray` | Boolean mask over penalized coefficients. |
| n_active | `#!python int` | Number of active penalized coefficients. |
| selected_variables | `#!python list[str]` | Variable names with active penalized coefficients. |
| l1_norm | `#!python float64` | L1 norm of penalized coefficients. |
| l2_norm_sq | `#!python float64` | Squared L2 norm of penalized coefficients. |
| l1_penalty | `#!python float64` | Realized L1 penalty component. |
| l2_penalty | `#!python float64` | Realized L2 penalty component. |
| penalty | `#!python float64` | Combined realized penalty. |

???+ note "Endpoint Dispatch"
    `l1_ratio=0.0` delegates to ridge logic and `l1_ratio=1.0` delegates to lasso logic where the selected criterion allows it. The returned object is still normalized to `ElasticNetResult`.

