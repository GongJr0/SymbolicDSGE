---
tags:
    - doc
---

# CUSUM Tests

```python
cusum_test_step(
    name: str,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: int | Sequence[int] | slice | ndarray | None = None,
    X_columns: int | Sequence[int] | slice | ndarray | None = None,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep
```

`cusum_test_step` wraps `run_cusum_test(...)`. It resolves the response and regressors from explicit producer and field pairs, then tests coefficient stability using the cumulative sum of standardized recursive residuals.

???+ info "Reference Distribution"
    The test is parameter-free: the boundary-crossing probability is Durbin's closed-form approximation evaluated directly on the statistic, so the reported degrees of freedom are not applicable (rendered as `N/A`).

???+ warning "Response must be 1D"
    `y_source` must resolve to exactly one column, and the response and regressor arrays must share the same number of rows.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| y_source | required | Producer step for the response. |
| y_field | required | Response producer field. |
| X_source | required | Producer step for regressors. |
| X_field | required | Regressor producer field. |
| y_column | `None` | Response column selector. It must resolve to one series. |
| X_columns | `None` | Optional regressor column subset. |
| burn_in | `0` | Rows dropped from both inputs before the test. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| alpha | `0.05` | Test size. |
