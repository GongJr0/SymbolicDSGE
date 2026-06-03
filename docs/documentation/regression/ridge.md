---
tags:
    - doc
---
# Ridge

```python
from SymbolicDSGE.regression.ridge import RidgeResult, ridge, ridge_gs
```

```python
ridge(
    x: ndarray,
    y: ndarray,
    alpha: float,
    variables: list[str] | None = None,
    intercept: bool = True,
) -> RidgeResult
```

Run ridge regression with an L2 penalty.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| x | Design matrix. Shape `(n, k)`. |
| y | Response vector. Shape `(n,)`. |
| alpha | Non-negative L2 penalty weight. |
| variables | Optional names for the columns of `x`. Defaults to `x0`, `x1`, ... |
| intercept | If `True`, prepend an unpenalized intercept column to the returned design matrix. |

__Returns:__

| __Type__ | __Description__ |
|:---------|----------------:|
| `#!python RidgeResult` | Regression result with ridge penalty diagnostics. |

&nbsp;

```python
ridge_gs(
    x: ndarray,
    y: ndarray,
    start: float,
    stop: float,
    num: int,
    criterion: Literal["aic", "bic", "loss"] = "aic",
    variables: list[str] | None = None,
    intercept: bool = True,
) -> RidgeResult
```

Run ridge regression over a logarithmic alpha grid and return the best result under the selected criterion.

__Inputs:__

| __Name__ | __Description__ |
|:---------|----------------:|
| start | Positive lower endpoint for the alpha grid. |
| stop | Positive upper endpoint for the alpha grid. |
| num | Number of grid points. |
| criterion | Grid-search objective: AIC, BIC, or residual loss. |
| x, y, variables, intercept | Same contract as `ridge(...)`. |

&nbsp;

```python
@dataclass(frozen=True)
class RidgeResult(RegressionResult)
```

__Additional Fields and Properties:__

| __Name__ | __Type__ | __Description__ |
|:---------|:--------:|----------------:|
| alpha | `#!python float64` | Selected L2 penalty weight. |
| effective_dof | `#!python float64` | Effective degrees of freedom used by grid-search criteria. |
| intercept | `#!python bool` | Whether the returned design includes an intercept column. |
| objective | `#!python RidgeObjective | None` | Grid-search criterion used to select `alpha`, if applicable. |
| objective_value | `#!python float64 | None` | Realized grid-search criterion value, if applicable. |
| l2_penalty | `#!python float64` | Realized ridge penalty excluding the intercept term. |

???+ note "Penalty Convention"
    When `intercept=True`, the intercept is not included in `l2_penalty` and is not regularized by the solver.
