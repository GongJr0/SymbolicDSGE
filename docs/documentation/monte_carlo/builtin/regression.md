---
tags:
    - doc
---

# Regression Steps

```python
regression_step(
    name: str,
    *,
    y_source: InpSources,
    X_source: InpSources,
    filter_key: str = "filter",
    y_payload_key: str | None = None,
    x_payload_key: str | None = None,
    y_column: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    variables: Sequence[str] | None = None,
) -> MCStep
```

`regression_step` conducts an Ordinary Least Squares (OLS) regression of `y_source` on `X_source` and stores the regression summary in `MCPipelineResult.regression_summaries` under the key `name`. The regression is conducted separately for each replication, and the summary statistics are stored as traces across replications.

__Sources:__

???+ warning "Target must be 1D"
    `regression_step` does not support multivariate targets. Multiple target values are to be regressed on the same set of regressors, separate `regression_step` components should be appended to the pipeline for each target variable.

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |