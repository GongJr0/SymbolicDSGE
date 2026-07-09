---
tags:
    - doc
---

# Regression Steps

```python
regression_step(
    name: str,
    *,
    kind: Literal[
        "ols",
        "ridge",
        "ridge_gs",
        "lasso",
        "lasso_gs",
        "elastic_net",
        "elastic_net_gs",
    ] = "ols",
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: int | Sequence[int] | slice | ndarray | None = None,
    X_columns: int | Sequence[int] | slice | ndarray | None = None,
    intercept: bool = True,
    burn_in: int = 0,
    drop_initial: bool = False,
    variables: Sequence[str] | None = None,
    **kind_kwargs: Any,
) -> MCStep
```

`regression_step` conducts a regression of `y` on `X` in each replication and stores the aggregate summary in `MCPipelineResult.regression_summaries[name]`.

__Kind Dispatch:__

| __kind__ | __Result Type__ | __Required `kind_kwargs`__ |
|:---------|:----------------|---------------------------:|
| `"ols"` | `#!python OLSResult` | none |
| `"ridge"` | `#!python RidgeResult` | `alpha` |
| `"ridge_gs"` | `#!python RidgeResult` | `start`, `stop`, `num` |
| `"lasso"` | `#!python LassoResult` | `alpha` |
| `"lasso_gs"` | `#!python LassoResult` | `start`, `stop`, `num` |
| `"elastic_net"` | `#!python ElasticNetResult` | `alpha`, `l1_ratio` |
| `"elastic_net_gs"` | `#!python ElasticNetResult` | `start`, `stop`, `num`, `l1_ratio` |

???+ warning "Target must be 1D"
    `y_source` and `y_field` must resolve to one column. Add separate `regression_step` components for multiple targets.

???+ note "Forwarded Regression Arguments"
    `kind_kwargs` are passed directly to the selected regression function. Grid-search kinds also accept the underlying regression options such as `criterion`, `max_iter`, and `tol` when supported by that method.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| kind | `"ols"` | Regression estimator. |
| y_source | required | Producer step for the response. |
| y_field | required | Response producer field. |
| X_source | required | Producer step for regressors. |
| X_field | required | Regressor producer field. |
| y_column | `None` | Response column selector. It must resolve to one series. |
| X_columns | `None` | Optional regressor column subset. |
| intercept | `True` | Add an intercept column. |
| burn_in | `0` | Rows dropped from both inputs before fitting. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| variables | `None` | Optional names for design columns. |
| kind_kwargs | none | Extra estimator arguments, such as `alpha`, `l1_ratio`, or grid settings. |
