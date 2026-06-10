---
tags:
    - doc
---

# CUSUM Tests

```python
cusum_test_step(
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

`cusum_test_step` wraps `run_cusum_test(...)`. It resolves a single response column from `y_source` and a regressor matrix from `x_source`, then tests the regression coefficients for stability using the cumulative sum of standardized recursive (Brown-Durbin-Evans) residuals.

???+ info "Reference Distribution"
    The test is parameter-free: the boundary-crossing probability is Durbin's closed-form approximation evaluated directly on the statistic, so the reported degrees of freedom are not applicable (rendered as `N/A`).

???+ warning "Response must be 1D"
    `y_source` must resolve to exactly one column, and the response and regressor arrays must share the same number of rows.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |
