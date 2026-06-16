---
tags:
    - doc
---

# CUSUM of Squares Tests

```python
cusumsq_test_step(
    name: str,
    *,
    y_source: InpSources,
    x_source: InpSources,
    filter_key: str = "filter",
    payload_key: str | None = None,
    y_column: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep
```

`cusumsq_test_step` wraps `run_cusumsq_test(...)`. It resolves a single response column from `y_source` and a regressor matrix from `x_source`, then tests the regression variance for stability using the cumulative sum of *squared* recursive (Brown-Durbin-Evans) residuals. Where the [CUSUM test](cusum.md) targets shifts in the coefficients, the CUSUM of squares targets shifts in the residual variance.

???+ info "Reference Distribution"
    The statistic is the maximum departure of the normalized squared-residual partial sums from their expected line, compared against a Kolmogorov-type survival function parameterized by the recursive-residual count $n = T - p$ (reported as the degrees of freedom).

???+ warning "Response must be 1D"
    `y_source` must resolve to exactly one column, and the response and regressor arrays must share the same number of rows.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |
