---
tags:
    - doc
---

# Breusch-Godfrey Tests

```python
breusch_godfrey_test_step(
    name: str,
    *,
    residual_source: InpSources,
    X_source: InpSources,
    filter_key: str = "filter",
    residual_payload_key: str | None = None,
    x_payload_key: str | None = None,
    residual_col: Sequence[int] | int | None = None,
    X_columns: Sequence[int] | slice | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 1,
    alpha: float = 0.05,
) -> MCStep
```

`breusch_godfrey_test_step` wraps `run_breusch_godfrey_test(...)`. It resolves a single residual column from `residual_source` and the original regressor matrix from `X_source`, then tests the residuals for serial correlation up to order `lags` via the auxiliary regression of the residuals on the regressors and their own lags.

???+ info "Reference Distribution"
    The Lagrange-multiplier statistic is compared against a $\chi^2(\text{lags})$ distribution.

???+ warning "Residuals must be 1D"
    `residual_source` must resolve to exactly one column, and the residual and regressor arrays must share the same number of rows.

__Sources:__

| __Source__ | __Description__ |
|:-----------|----------------:|
| `states` | Use `context.data.states`. Set `drop_initial=True` to remove the initial state row. |
| `observables` | Use `context.data.observables`. |
| `x_pred`, `x_filt`, `y_pred`, `y_filt`, `innov`, `std_innov` | Read arrays from the `FilterResult` stored under `filter_key`. |
| `payload` | Read an array-like object from `context.payloads[payload_key]`. |
