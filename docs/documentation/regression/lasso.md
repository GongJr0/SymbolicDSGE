---
tags:
    - doc
---
# Lasso

```python
from SymbolicDSGE.regression.lasso import LassoResult, lasso, lasso_gs
```

```python
lasso(
    X: ndarray,
    y: ndarray,
    alpha: float,
    variables: list[str] | None = None,
    intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> LassoResult
```

Run Lasso regression with an L1 penalty using coordinate descent on the normalized Gram system.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| X | Design matrix. Shape `(n, k)`. |
| y | Response vector. Shape `(n,)`. |
| alpha | Non-negative L1 penalty weight. |
| variables | Optional names for the columns of `X`. Defaults to `x0`, `x1`, ... |
| intercept | If `True`, fit an unpenalized intercept by centering and restore it in the returned design. |
| max_iter | Maximum coordinate-descent iterations. |
| tol | Coordinate-descent convergence tolerance. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python LassoResult` | Regression result with L1 sparsity diagnostics. |

&nbsp;

```python
lasso_gs(
    X: ndarray,
    y: ndarray,
    start: float,
    stop: float,
    num: int,
    variables: list[str] | None = None,
    intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> LassoResult
```

Evaluate a logarithmic alpha grid using the LARS-Lasso path and select the coefficient vector with the lowest residual loss.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| start | Positive lower endpoint for the alpha grid. |
| stop | Positive upper endpoint for the alpha grid. |
| num | Number of grid points. |
| X, y, variables, intercept, max_iter, tol | Same contract as `lasso(...)`. |

&nbsp;

```python
@dataclass(frozen=True)
class LassoResult(RegressionResult)
```

__Additional Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| alpha | `#!python float64` | Selected L1 penalty weight. |
| effective_dof | `#!python float64` | Active penalized coefficient count plus the intercept, when present. |
| intercept | `#!python bool` | Whether the returned design includes an intercept column. |
| alpha_grid | `#!python ndarray | None` | Grid of alpha values evaluated by `lasso_gs(...)`. |
| coefficient_path | `#!python ndarray | None` | Coefficients evaluated on `alpha_grid`. |
| objective_trace | `#!python ndarray | None` | Residual-loss trace over `alpha_grid`. |
| knot_lambdas | `#!python ndarray | None` | LARS path knot locations from grid-search construction. |
| knot_coefficients | `#!python ndarray | None` | LARS path coefficients aligned to `knot_lambdas`. |
| penalized_coefficients | `#!python ndarray` | Coefficients subject to the L1 penalty. |
| active_mask | `#!python ndarray` | Boolean mask over penalized coefficients. |
| n_active | `#!python int` | Number of active penalized coefficients. |
| selected_variables | `#!python list[str]` | Variable names with active penalized coefficients. |
| l1_norm | `#!python float64` | L1 norm of penalized coefficients. |
| l1_penalty | `#!python float64` | Realized L1 penalty value. |

???+ note "Path Output"
    Path fields are populated by `lasso_gs(...)`. Direct `lasso(...)` calls return only the selected coefficient vector and scalar L1 diagnostics.

