---
tags:
    - doc
---

# Breusch-Godfrey Tests

```python
breusch_godfrey_test_step(
    name: str,
    *,
    residuals_source: str,
    residuals_field: str,
    X_source: str,
    X_field: str,
    residual_col: int | Sequence[int] | slice | ndarray | None = None,
    X_columns: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    lags: int = 1,
    alpha: float = 0.05,
) -> MCStep
```

`breusch_godfrey_test_step` wraps `run_breusch_godfrey_test(...)`. It resolves residuals and regressors from explicit producer and field pairs, then tests residual serial correlation up to order `lags`.

???+ info "Reference Distribution"
    The Lagrange-multiplier statistic is compared against a $\chi^2(\text{lags})$ distribution.

???+ warning "Residuals must be 1D"
    `residuals_source` and `residuals_field` must resolve to exactly one column. The residual and regressor arrays must share the same number of rows.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| residuals_source | required | Producer step for residuals. |
| residuals_field | required | Residual producer field. |
| X_source | required | Producer step for regressors. |
| X_field | required | Regressor producer field. |
| residual_col | `None` | Residual column selector. It must resolve to one series. |
| X_columns | `None` | Optional regressor column subset. |
| burn_in | `0` | Rows dropped from both inputs before the test. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| lags | `1` | Serial correlation lag order. |
| alpha | `0.05` | Test size. |
