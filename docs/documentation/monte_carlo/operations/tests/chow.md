---
tags:
    - doc
---

# Chow Tests

```python
chow_test_step(
    name: str,
    *,
    y_source: str,
    y_field: str,
    X_source: str,
    X_field: str,
    y_column: int | Sequence[int] | slice | ndarray | None = None,
    X_columns: int | Sequence[int] | slice | ndarray | None = None,
    t_break: int = 10,
    burn_in: int = 0,
    drop_initial: bool = False,
    alpha: float = 0.05,
) -> MCStep
```

`chow_test_step` wraps `run_chow_test(...)`. It resolves the response and regressors from explicit producer and field pairs, then tests for a structural break at `t_break` by comparing the pooled residual sum of squares against the two subsample fits.

???+ info "Reference Distribution"
    The statistic is compared against an $F(p,\ T - 2p)$ distribution, where $p$ is the number of regressor columns and $T$ is the number of observations.

???+ warning "Break point and sample size"
    `t_break` must satisfy `0 < t_break < T`, and the sample must provide enough observations for two separate fits (`T > 2p`); otherwise the step reports a non-`OK` status. `y_source` must resolve to exactly one column, and the response and regressor arrays must share the same number of rows.

__Inputs:__

| __Name__ | __Default__ | __Description__ |
|:---------|:-----------:|----------------:|
| y_source | required | Producer step for the response. |
| y_field | required | Response producer field. |
| X_source | required | Producer step for regressors. |
| X_field | required | Regressor producer field. |
| y_column | `None` | Response column selector. It must resolve to one series. |
| X_columns | `None` | Optional regressor column subset. |
| t_break | `10` | Break point row after selection and row trimming. |
| burn_in | `0` | Rows dropped from both inputs before the test. |
| drop_initial | `False` | Start at row `1` when `burn_in=0`. |
| alpha | `0.05` | Test size. |
