---
tags:
    - doc
---

# CUSUM of Squares Tests

```python
cusumsq_test_step(
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

`cusumsq_test_step` wraps `run_cusumsq_test(...)`. It resolves the response and regressors from explicit producer and field pairs, then tests variance stability using the cumulative sum of squared recursive residuals.

???+ info "Reference Distribution"
    The statistic is the maximum departure of the normalized squared-residual partial sums from their expected line, compared against a Kolmogorov-type survival function parameterized by the recursive-residual count $n = T - p$ (reported as the degrees of freedom).

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
