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
    y_source: InpSources,
    X_source: InpSources,
    filter_key: str = "filter",
    y_payload_key: str | None = None,
    x_payload_key: str | None = None,
    y_column: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    intercept: bool = True,
    burn_in: int = 0,
    drop_initial: bool = False,
    variables: Sequence[str] | None = None,
    **kind_kwargs: Any,
) -> MCStep
```

`regression_step` conducts a regression of `y_source` on `X_source` and stores the regression summary in `MCPipelineResult.regression_summaries` under the key `name`. The regression is conducted separately for each replication, and summary statistics are stored as traces across replications.

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
    `regression_step` does not support multivariate targets. Multiple target values are to be regressed on the same set of regressors, separate `regression_step` components should be appended to the pipeline for each target variable.

???+ note "Forwarded Regression Arguments"
    `kind_kwargs` are passed directly to the selected regression function. Grid-search kinds also accept the underlying regression options such as `criterion`, `max_iter`, and `tol` when supported by that method.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |
